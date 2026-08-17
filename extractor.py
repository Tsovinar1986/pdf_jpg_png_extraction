#!/usr/bin/env python3
"""Simple text extractor for PDFs and images.

Usage:
  python extractor.py /path/to/file_or_dir

Features:
- If a PDF contains extractable text, use pdfminer.six.
- Otherwise convert PDF pages to images (pdf2image) and run OCR (pytesseract).
- For image files, run OCR via pytesseract.
"""
import argparse
import io
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    from PIL import Image, ImageFilter, ImageOps, ImageStat
except Exception:
    Image = None
    ImageFilter = None
    ImageOps = None
    ImageStat = None

try:
    import pytesseract
except Exception:
    pytesseract = None

if pytesseract is not None:
    _tesseract_cmd = os.getenv("TESSERACT_CMD")
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
    elif sys.platform == "win32":
        # The Windows installer doesn't add tesseract.exe to PATH by
        # default; fall back to its default install location if present.
        _default_tesseract = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe"
        if _default_tesseract.exists():
            pytesseract.pytesseract.tesseract_cmd = str(_default_tesseract)

try:
    import pymupdf
except Exception:
    pymupdf = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import openpyxl
except Exception:
    openpyxl = None

try:
    import docx as python_docx
except Exception:
    python_docx = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
TEXT_EXTS = {".txt", ".csv"}
SPREADSHEET_EXTS = {".xlsx", ".xlsm"}
DOCX_EXTS = {".docx"}
OFFICE_ZIP_EXTS = SPREADSHEET_EXTS | DOCX_EXTS

try:
    # Adds HEIC/HEIF decoding support to Pillow (common iOS/macOS screenshot
    # and photo format that Pillow can't open on its own).
    import pillow_heif

    pillow_heif.register_heif_opener()
    IMAGE_EXTS |= {".heic", ".heif"}
except Exception:
    pass

# Tesseract language packs to run together (Armenian docs are often mixed
# with Russian/English headers and stamps, like the NAIRI medical reports).
# "hye-calfa-n" (not Tesseract's own "hye") is used for Armenian: it's a
# community model trained on Classical, Western, *and* Eastern Armenian,
# vs. stock "hye" which only covers Eastern Armenian orthography — see
# https://github.com/calfa-co/hye-tesseract. Requires manually placing
# hye-calfa-n.traineddata in Tesseract's tessdata folder (README/
# requirements.txt have install steps); if it isn't installed, set
# OCR_LANGS=hye+rus+eng to fall back to the stock Eastern-Armenian-only
# model, or OCR_LANGS to whatever languages you do have.
# Override with the OCR_LANGS env var, e.g. "eng" for Latin-only scans.
DEFAULT_OCR_LANGS = os.getenv("OCR_LANGS", "hye-calfa-n+rus+eng")

# On Windows, poppler (pdftoppm/pdftocairo) usually isn't on PATH unless
# manually added; point pdf2image at its bin/ folder via this env var.
POPPLER_PATH = os.getenv("POPPLER_PATH") or None


def _installed_ocr_langs() -> Optional[set]:
    """List Tesseract's installed language/model codes.

    Deliberately doesn't use pytesseract.get_languages(): it filters
    tesseract's own `--list-langs` output through a strict `^[a-z_]+$`
    regex that rejects any code with a hyphen or digit — which silently
    drops legitimate custom models (e.g. the community "hye-calfa-n"
    Armenian model) that Tesseract itself lists and runs just fine. That
    would make _resolve_ocr_langs below wrongly report a real, working
    language pack as missing.
    """
    if pytesseract is None:
        return None
    try:
        result = subprocess.run(
            [pytesseract.pytesseract.tesseract_cmd, "--list-langs"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    # First line is a header ("List of available languages in ..."), the
    # rest is one language/model code per line.
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    langs = set(lines[1:]) if lines and lines[0].lower().startswith("list of") else set(lines)
    return langs or None


def _resolve_ocr_langs(lang: str) -> str:
    """Validate the '+'-joined language codes against what Tesseract
    actually has trained data for.

    A missing language pack (most commonly Armenian on Windows, where the
    installer's language checkboxes are easy to miss) doesn't reliably
    make tesseract error out — depending on the build, it can silently OCR
    the script using only the languages it does have, misreading Armenian
    letter shapes as vaguely similar Latin/Cyrillic ones and returning
    fluent-looking but meaningless gibberish instead of failing. Checking
    up front turns that into a clear, actionable error.
    """
    requested = [code for code in lang.split("+") if code]
    installed = _installed_ocr_langs()
    if not installed:
        # get_languages() itself failed (e.g. tesseract not found at all);
        # let the OCR call raise its own, more specific error.
        return lang

    missing = [code for code in requested if code not in installed]
    if not missing:
        return lang

    usable = [code for code in requested if code in installed]
    fix_hint = (
        "Install the missing Tesseract language data (see the README's per-OS "
        "instructions — on Windows, rerun the Tesseract installer and check the "
        "language you need)"
    )
    if not usable:
        raise RuntimeError(
            f"None of the requested OCR languages ({'+'.join(requested)}) are installed "
            f"in Tesseract ({len(installed)} other language(s) found). {fix_hint}, "
            "or set OCR_LANGS to a language you do have installed."
        )

    raise RuntimeError(
        f"Missing Tesseract language data for: {', '.join(missing)}. OCR would silently "
        "misread that script using only the installed languages instead of failing "
        f"clearly, so refusing to run. {fix_hint}, or set OCR_LANGS={'+'.join(usable)} "
        "to proceed with only what's installed."
    )


def _preprocess_for_ocr(img: "Image.Image") -> "Image.Image":
    """Upscale small images and boost contrast before OCR.

    Tesseract's layout analysis and character recognition are unreliable
    on the low-resolution, low-contrast images typical of screenshots
    (e.g. small white text on a colored chat-bubble background) — it can
    silently drop whole text blocks rather than misread them.
    """
    img = img.convert("RGB")
    if max(img.size) < 2000:
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    # Decorative/script fonts common on posters and graphics are often
    # anti-aliased into soft edges that blur together at OCR resolution;
    # a light unsharp mask crisps the strokes back up before thresholding.
    return gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# A single character repeated 3+ times in a row (e.g. "AAAAAAA") is
# essentially never real text. It's the classic misread of a repeating
# decorative graphic — a row of pine-tree triangles, dots, snowflakes, a
# striped ribbon edge — into the one Latin/Cyrillic/etc. letter that
# shape resembles. Tesseract is often unnervingly *confident* about that
# misread since the shape repeats cleanly, so left unfiltered it can
# rack up a higher confidence-weighted score than the real (shorter,
# less certain) text elsewhere in a busy image and win the "best of"
# comparison outright.
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


def _box_iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _otsu_threshold(gray_img: "Image.Image") -> int:
    hist = gray_img.histogram()
    total = sum(hist)
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_bg = 0.0
    weight_bg = 0
    best_variance = 0.0
    threshold = 127
    for i, h in enumerate(hist):
        weight_bg += h
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += i * h
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = i
    return threshold


def _adaptive_threshold_variants(gray_img: "Image.Image") -> List["Image.Image"]:
    """Locally-thresholded binarizations of a grayscale image via OpenCV.

    Otsu's threshold is global: it picks one cutoff for the whole image, so
    it only works when text/background contrast is roughly uniform
    everywhere. Poster-style graphics with gradients, multi-colored text,
    and illustration behind the words routinely have different local
    contrast in different regions — a word that's legible against a
    lighter corner of the background can vanish into a darker corner
    under one global cutoff. Gaussian-weighted adaptive thresholding picks
    a cutoff per neighborhood instead, so it can pick up text a global
    threshold misses. Returns [] if OpenCV/NumPy aren't installed.
    """
    if cv2 is None or np is None:
        return []
    arr = np.array(gray_img)
    variants = []
    for block_size in (25, 51):
        binarized = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10
        )
        variants.append(Image.fromarray(binarized))
    return variants


def _cv2_clahe_variant(img: "Image.Image") -> Optional["Image.Image"]:
    """A second, independently-tuned preprocessing pipeline via OpenCV:
    cubic-upscale → auto-invert dark images → CLAHE → bilateral denoise →
    adaptive threshold.

    The PIL-based pipeline above (autocontrast + unsharp mask) is cheap
    and works for most images, but CLAHE equalizes local contrast far
    better than one global autocontrast pass on unevenly-lit poster
    photos, and a bilateral filter smooths illustration/photo noise while
    keeping text edges sharp — cases where the simpler pipeline still
    comes up short. Runs off the original (not the PIL-preprocessed)
    image so the two pipelines stay independent. Returns None if
    OpenCV/NumPy aren't installed.
    """
    if cv2 is None or np is None:
        return None
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if max(gray.shape) < 2000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if gray.mean() < 127:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)


def _shading_corrected_variant(img: "Image.Image") -> Optional["Image.Image"]:
    """Illumination/shading-corrected adaptive-threshold binarization.

    CLAHE and Otsu/adaptive thresholding correct *local* contrast, but a
    large-scale gradient or vignette across a poster's background (soft
    lighting, a color fade) can still bias them since it changes the
    baseline brightness a whole region sits on. Dividing the image by a
    heavily morphologically-opened (i.e. large-scale-blurred) version of
    itself cancels that gradient out first, so the adaptive threshold
    that follows only has to deal with the actual text strokes on an
    already-flattened background. Returns None if OpenCV/NumPy aren't
    installed.
    """
    if cv2 is None or np is None:
        return None
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if max(gray.shape) < 2000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    background = np.where(background == 0, 1, background).astype(np.uint8)
    normalized = cv2.divide(gray, background, scale=255)
    thresh = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(thresh)


# Engine/page-segmentation combos to try alongside the default. --oem 3
# (LSTM, Tesseract's default) is pinned explicitly for clarity. --psm 3
# ("fully automatic") assumes a page laid out in coherent blocks/columns;
# poster graphics scatter short text fragments around illustrations with
# no such structure, which can make automatic layout analysis miss them
# entirely. --psm 6 ("single uniform block") suits a stacked caption like
# "March / Women's / Day", while --psm 11 ("sparse text") looks for words
# anywhere in the image in no particular order — between the three, most
# poster layouts are covered.
_OCR_CONFIGS = ("--oem 3", "--oem 3 --psm 6", "--oem 3 --psm 11")


def _ocr_candidate_images(raw_img: "Image.Image") -> List[tuple]:
    """Build several different renderings of the same image for OCR to try,
    each paired with its (scale_x, scale_y) relative to raw_img.

    Tesseract is tuned for solid dark text on a light background. Posters,
    dark-mode screenshots, and images with busy illustrated backgrounds
    can confuse its thresholding badly enough that a plain grayscale pass
    reads nothing, or hallucinates junk from the artwork. Two independent
    preprocessing pipelines (PIL autocontrast/unsharp, and OpenCV
    CLAHE/bilateral-denoise), each with plain/inverted/Otsu/adaptive-
    threshold binarization, cover the common failure modes without having
    to guess which one applies ahead of time.
    """
    pil_pre = _preprocess_for_ocr(raw_img)
    sx, sy = pil_pre.width / raw_img.width, pil_pre.height / raw_img.height
    threshold = _otsu_threshold(pil_pre)
    binarized = pil_pre.point(lambda p: 255 if p > threshold else 0)
    candidates = [
        (pil_pre, sx, sy),
        (ImageOps.invert(pil_pre), sx, sy),
        (binarized, sx, sy),
        (ImageOps.invert(binarized), sx, sy),
    ]
    candidates += [(v, sx, sy) for v in _adaptive_threshold_variants(pil_pre)]

    cv2_variant = _cv2_clahe_variant(raw_img)
    if cv2_variant is not None:
        candidates.append((cv2_variant, cv2_variant.width / raw_img.width, cv2_variant.height / raw_img.height))

    shading_variant = _shading_corrected_variant(raw_img)
    if shading_variant is not None:
        candidates.append((shading_variant, shading_variant.width / raw_img.width, shading_variant.height / raw_img.height))

    return candidates


def _collect_ocr_detections(raw_img: "Image.Image", lang: str, min_conf: float) -> List[tuple]:
    """Run every candidate rendering × page-segmentation config and return
    every accepted (box, conf, word) detection, in raw_img's own pixel
    coordinates, with duplicates from overlapping candidates collapsed.

    A single "best overall candidate" doesn't exist for busy multi-color
    graphics: one thresholding pass may read a ribbon banner's yellow-on-
    red text well but miss a heading entirely, while another does the
    reverse. Collecting detections from every candidate and keeping the
    highest-confidence read of each *region* (rather than picking one
    candidate's full-image output) lets each region be read by whichever
    rendering actually suits it.
    """
    detections = []
    for candidate, sx, sy in _ocr_candidate_images(raw_img):
        for config in _OCR_CONFIGS:
            try:
                data = pytesseract.image_to_data(candidate, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            except Exception:
                continue
            for i, word in enumerate(data.get("text", [])):
                word = word.strip()
                if not word or _REPEATED_CHAR_RE.search(word):
                    continue
                try:
                    conf = float(data["conf"][i])
                except (ValueError, TypeError):
                    conf = -1.0
                if conf < min_conf:
                    continue
                box = (
                    data["left"][i] / sx,
                    data["top"][i] / sy,
                    data["width"][i] / sx,
                    data["height"][i] / sy,
                )
                detections.append((box, conf, word))

    detections.sort(key=lambda d: -d[1])
    kept = []
    for box, conf, word in detections:
        x, y, w, h = box
        # A misread graphic (a ribbon edge, an arrow tip, a row of icons)
        # can get labeled as one "word" spanning a box far wider per
        # character than real text ever is — Tesseract's confidence
        # reflects how sure it is about the character shapes it guessed,
        # not whether the box is a plausible word at all.
        if w > len(word) * h * 1.3:
            continue
        if any(_box_iou(box, kb) > 0.3 for kb, _, _ in kept):
            continue
        kept.append((box, conf, word))
    return kept


def _cluster_lines(detections: List[tuple]) -> List[dict]:
    """Group detections into horizontal lines by vertical overlap AND
    horizontal proximity to the nearest word already in that line,
    top-to-bottom.

    Vertical proximity alone isn't enough: a decorative glyph in a
    border far to the side of the page can share a similar y-coordinate
    with a real text line purely by coincidence and get merged into it,
    which then throws off anything built on top of these lines (reading
    order, paragraph-block detection). Requiring the new word to actually
    sit close to its nearest neighbor in the line — not just anywhere in
    a wide y-band — keeps unrelated same-row content separate.
    """
    by_top = sorted(detections, key=lambda d: d[0][1])
    lines: List[dict] = []
    for det in by_top:
        x, y, w, h = det[0]
        line = None
        for ln in lines:
            if abs(y - ln["y"]) >= h * 0.6:
                continue
            nearest_gap = min(max(x - (ox + ow), ox - (x + w)) for (ox, _oy, ow, _oh), _c, _wd in ln["dets"])
            if nearest_gap < h * 8:
                line = ln
                break
        if line is None:
            lines.append({"y": y, "dets": [det]})
        else:
            line["dets"].append(det)
            line["y"] = (line["y"] + y) / 2
    lines.sort(key=lambda ln: ln["y"])
    return lines


def _detections_to_text(detections: List[tuple]) -> str:
    """Assemble kept detections into reading-order text: cluster into
    lines by vertical overlap, then sort each line left-to-right."""
    lines = _cluster_lines(detections)
    return "\n".join(
        " ".join(word for (x, _y, _w, _h), _conf, word in sorted(ln["dets"], key=lambda d: d[0][0]))
        for ln in lines
    )


def _prefer_dominant_paragraph_block(detections: List[tuple]) -> List[tuple]:
    """If the detections contain one clearly dominant multi-line
    paragraph block, apply a stricter confidence bar to everything
    outside it.

    Ornate decorative borders/frames (lace patterns, a ring of small
    icons around a greeting card) can generate dozens of scattered,
    medium-confidence word-shaped misreads that individually clear the
    normal bar. A real paragraph is reliably several consecutive,
    closely-spaced lines with multiple words each — border noise almost
    never clusters that way. When such a block clearly exists, trust it
    and demand more confidence from everything else. Posters/captions
    made of a few separate short text elements (no dominant block) are
    left untouched, since this would otherwise wrongly gut them.
    """
    lines = _cluster_lines(detections)
    if len(lines) < 3:
        return detections

    heights = sorted(d[0][3] for ln in lines for d in ln["dets"])
    median_h = heights[len(heights) // 2]
    if median_h <= 0:
        return detections

    # Group lines into blocks by *both* vertical proximity and horizontal
    # overlap. Y-proximity alone isn't enough: a column of single-glyph
    # border noise running down one edge sits at roughly the same
    # y-positions as a paragraph's lines and would otherwise chain
    # together into a tall fake "block" purely from tight vertical
    # spacing, even though each of its "lines" has only one word. Also
    # requiring horizontal overlap keeps a side column of noise from ever
    # joining the paragraph's block, since they occupy disjoint x-ranges.
    blocks: List[dict] = []
    for ln in lines:
        x0 = min(d[0][0] for d in ln["dets"])
        x1 = max(d[0][0] + d[0][2] for d in ln["dets"])
        placed = False
        for blk in blocks:
            last = blk["lines"][-1]
            overlaps = x0 <= blk["x1"] and blk["x0"] <= x1
            if overlaps and (ln["y"] - last["y"]) < median_h * 1.8:
                blk["lines"].append(ln)
                blk["x0"], blk["x1"] = min(blk["x0"], x0), max(blk["x1"], x1)
                placed = True
                break
        if not placed:
            blocks.append({"lines": [ln], "x0": x0, "x1": x1})

    def word_count(blk):
        return sum(len(ln["dets"]) for ln in blk["lines"])

    # A real paragraph has multiple words on nearly every line; scattered
    # border noise averages close to one word per "line" even when it
    # does chain together. Requiring a minimum density here is what
    # actually excludes it, not line count alone.
    paragraph_like = [b for b in blocks if len(b["lines"]) >= 3 and word_count(b) / len(b["lines"]) >= 3]
    if not paragraph_like:
        return detections  # nothing clearly paragraph-shaped; leave as-is

    dominant = max(paragraph_like, key=lambda b: (len(b["lines"]), word_count(b)))
    if word_count(dominant) < 9:
        return detections

    dominant_ids = {id(d) for ln in dominant["lines"] for d in ln["dets"]}
    return [det for det in detections if id(det) in dominant_ids or det[1] >= 65]


def _ocr_best_of(raw_img: "Image.Image", lang: str) -> str:
    detections = _collect_ocr_detections(raw_img, lang, min_conf=40)
    detections = _prefer_dominant_paragraph_block(detections)
    return _detections_to_text(detections)


def extract_text_from_pdf(path: Path, lang: str = DEFAULT_OCR_LANGS) -> str:
    # First try to extract text directly (for digitally generated PDFs)
    if pdf_extract_text is not None:
        try:
            text = pdf_extract_text(str(path))
            if text and text.strip():
                return text
        except Exception:
            pass

    # Fallback: render pages to images and OCR
    if convert_from_path is None or pytesseract is None or Image is None:
        raise RuntimeError("Missing pdf->image OCR dependencies: install pdf2image, pytesseract, pillow")

    lang = _resolve_ocr_langs(lang)
    pages = convert_from_path(str(path), poppler_path=POPPLER_PATH)
    out_lines = []
    for i, page in enumerate(pages, start=1):
        txt = pytesseract.image_to_string(_preprocess_for_ocr(page), lang=lang)
        out_lines.append(f"\n--- PAGE {i} ---\n")
        out_lines.append(txt)
    return "\n".join(out_lines)


def extract_text_from_image(path: Path, lang: str = DEFAULT_OCR_LANGS) -> str:
    if pytesseract is None or Image is None:
        raise RuntimeError("Missing image OCR dependencies: install pytesseract and pillow")
    # Closing the handle (rather than leaving it open until GC) matters on
    # Windows, where the caller's temp-file cleanup would otherwise fail
    # with "file in use by another process".
    lang = _resolve_ocr_langs(lang)
    with Image.open(path) as img:
        return _ocr_best_of(img, lang)


def normalize_dark_image_to_paper(img: "Image.Image") -> Optional["Image.Image"]:
    """If the image looks like light text on a dark background, convert it
    to a normal-looking scanned-paper style: white background, black text.

    Dark-mode screenshots and dark poster/flyer designs are readable to a
    person but visually the inverse of a normal document; producing a
    black-on-white version alongside the extracted text gives something
    that looks like what people expect a "cleaned up" text image to look
    like. Returns None if the image isn't dark-background to begin with
    (nothing to normalize) or if Pillow isn't installed.
    """
    if Image is None or ImageOps is None or ImageStat is None:
        return None

    gray = ImageOps.grayscale(img.convert("RGB"))
    if ImageStat.Stat(gray).mean[0] >= 127:
        return None  # already a light background

    threshold = _otsu_threshold(gray)
    binarized = gray.point(lambda p: 255 if p > threshold else 0)

    # Otsu doesn't know which side of the cutoff is "background" vs
    # "text" — assume whichever value covers more pixels is the
    # background, and normalize so background is always white (255) and
    # text is always black (0), regardless of the source's original
    # polarity or color.
    hist = binarized.histogram()
    if hist[0] > hist[255]:
        binarized = ImageOps.invert(binarized)

    return binarized.convert("RGB")


def extract_text_from_textfile(path: Path) -> str:
    # Text/CSV files already are text — read them as-is rather than OCR-ing.
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def extract_text_from_spreadsheet(path: Path) -> str:
    if openpyxl is None:
        raise RuntimeError("Missing spreadsheet dependency: install openpyxl")
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    out_lines = []
    try:
        for ws in wb.worksheets:
            out_lines.append(f"\n--- SHEET: {ws.title} ---\n")
            for row in ws.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                if any(cells):
                    out_lines.append("\t".join(cells))
    finally:
        wb.close()
    return "\n".join(out_lines)


def extract_text_from_docx(path: Path) -> str:
    if python_docx is None:
        raise RuntimeError("Missing docx dependency: install python-docx")
    document = python_docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _strip_text_from_image(img: "Image.Image", lang: str = DEFAULT_OCR_LANGS) -> "Image.Image":
    """Remove OCR-detected text from an image via inpainting, leaving just
    the underlying artwork/photo.

    "Extract images" is for pulling out the picture asset itself — a
    poster's illustration, a product photo — not a duplicate of what
    "extract text" already returns baked into the picture. Reuses the
    same multi-candidate detection battery as text extraction (plain
    single-pass detection reliably misses decorative/stylized fonts on
    busy backgrounds, leaving the original text untouched) but with a
    much lower confidence floor: finding *where* text roughly is only
    needs to be right enough to mask it, not right enough to read it, so
    it's fine to be far more liberal here than when extracting text.
    Falls back to returning the image unchanged if OpenCV isn't
    installed, the language pack is missing, or OCR/inpainting fails for
    any reason — stripping is a best-effort enhancement, not something
    that should block image extraction.
    """
    if cv2 is None or np is None or pytesseract is None:
        return img

    try:
        resolved_lang = _resolve_ocr_langs(lang)
    except RuntimeError:
        return img

    rgb = img.convert("RGB")
    try:
        # A near-zero floor (below ~15) starts admitting misreads of
        # repeating decorative patterns (a row of pine-tree/dot/snowflake
        # shapes) as one long, low-confidence "word" spanning the whole
        # row — genuine text, even faint or stylized, reliably scores
        # well above that.
        detections = _collect_ocr_detections(rgb, resolved_lang, min_conf=15)
    except Exception:
        return img

    if not detections:
        return img

    mask = np.zeros((rgb.height, rgb.width), dtype=np.uint8)
    for (x, y, w, h), _conf, _word in detections:
        pad = max(3, int(0.3 * h))
        x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
        x1, y1 = min(rgb.width, int(x + w + pad)), min(rgb.height, int(y + h + pad))
        mask[y0:y1, x0:x1] = 255

    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))


def _to_png_bytes(img: "Image.Image") -> bytes:
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_images_from_pdf(path: Path) -> List[bytes]:
    """Return every embedded image in the PDF as PNG bytes, in document order.

    Falls back to rendering full pages when the PDF has no embedded images
    (e.g. it's a scanned PDF stored as page-level vector/XObject content).
    """
    if pymupdf is None:
        raise RuntimeError("Missing PDF image extraction dependency: install pymupdf")

    doc = pymupdf.open(str(path))
    images: List[bytes] = []
    try:
        for page in doc:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    extracted = doc.extract_image(xref)
                    raw = extracted.get("image")
                    if not raw:
                        continue
                    if Image is not None:
                        images.append(_to_png_bytes(_strip_text_from_image(Image.open(io.BytesIO(raw)))))
                    else:
                        images.append(raw)
                except Exception:
                    continue

        if not images and convert_from_path is not None:
            for page in convert_from_path(str(path), poppler_path=POPPLER_PATH):
                images.append(_to_png_bytes(page))
    finally:
        doc.close()

    return images


def extract_images_from_office(path: Path) -> List[bytes]:
    """xlsx/docx are zip archives; pull out whatever pictures are embedded
    under their media/ folder."""
    images: List[bytes] = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if "/media/" in n):
            raw = z.read(name)
            if Image is not None:
                try:
                    images.append(_to_png_bytes(_strip_text_from_image(Image.open(io.BytesIO(raw)))))
                    continue
                except Exception:
                    pass
            images.append(raw)
    return images


def extract_images_from_file(path: Path) -> List[bytes]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_images_from_pdf(path)
    if suffix in IMAGE_EXTS:
        if Image is None:
            raise RuntimeError("Missing image dependency: install pillow")
        with Image.open(path) as img:
            return [_to_png_bytes(_strip_text_from_image(img))]
    if suffix in OFFICE_ZIP_EXTS:
        return extract_images_from_office(path)
    if suffix in TEXT_EXTS:
        return []
    raise ValueError(f"Unsupported file type: {suffix}")


ALL_SUPPORTED_EXTS = IMAGE_EXTS | TEXT_EXTS | SPREADSHEET_EXTS | DOCX_EXTS | {".pdf"}


def extract_text_from_path(p: Path) -> Optional[str]:
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(p)
    if suffix in IMAGE_EXTS:
        return extract_text_from_image(p)
    if suffix in TEXT_EXTS:
        return extract_text_from_textfile(p)
    if suffix in SPREADSHEET_EXTS:
        return extract_text_from_spreadsheet(p)
    if suffix in DOCX_EXTS:
        return extract_text_from_docx(p)
    return None


def process_path(p: Path) -> Optional[str]:
    if not p.exists():
        print(f"Not found: {p}", file=sys.stderr)
        return None

    if p.is_dir():
        outputs = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in ALL_SUPPORTED_EXTS:
                txt = process_path(f)
                if txt:
                    outputs.append(f"\n===== {f} =====\n")
                    outputs.append(txt)
        return "\n".join(outputs)

    # single file
    if p.suffix.lower() in ALL_SUPPORTED_EXTS:
        return extract_text_from_path(p)

    print(f"Unsupported file type: {p}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs and images (with OCR fallback).")
    parser.add_argument("path", help="File or directory path to process")
    parser.add_argument("-o", "--out", help="Output file. If omitted prints to stdout")
    args = parser.parse_args()

    p = Path(args.path)
    try:
        result = process_path(p)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if not result:
        print("No text extracted.")
        sys.exit(0)

    if args.out:
        outp = Path(args.out)
        outp.write_text(result, encoding="utf-8")
        print(f"Wrote: {outp}")
    else:
        print(result)


if __name__ == "__main__":
    main()
