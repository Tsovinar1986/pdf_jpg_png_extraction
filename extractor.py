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
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

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
# Override with the OCR_LANGS env var, e.g. "eng" for Latin-only scans.
DEFAULT_OCR_LANGS = os.getenv("OCR_LANGS", "hye+rus+eng")

# On Windows, poppler (pdftoppm/pdftocairo) usually isn't on PATH unless
# manually added; point pdf2image at its bin/ folder via this env var.
POPPLER_PATH = os.getenv("POPPLER_PATH") or None


def _installed_ocr_langs() -> Optional[set]:
    if pytesseract is None:
        return None
    try:
        return set(pytesseract.get_languages(config=""))
    except Exception:
        return None


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
    return ImageOps.autocontrast(ImageOps.grayscale(img))


_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _ocr_score(text: str) -> int:
    # Counting raw alnum characters lets noise win: busy illustrations can
    # OCR into long runs of junk symbols/short fragments that outscore a
    # short-but-correct reading. Score by letters inside word-like runs of
    # 3+ characters instead — garbage rarely forms those consistently.
    return sum(len(w) for w in _WORD_RE.findall(text))


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


def _ocr_best_of(img: "Image.Image", lang: str) -> str:
    """Run OCR against a few different renderings of the same (grayscale)
    image and keep whichever recognized the most real words.

    Tesseract is tuned for solid dark text on a light background. Posters,
    dark-mode screenshots, and images with busy illustrated backgrounds
    can confuse its thresholding badly enough that a plain grayscale pass
    reads nothing, or hallucinates junk from the artwork. Trying the
    original, its color inversion, and a binarized (pure black/white,
    Otsu-thresholded) version of both covers the common failure modes
    without having to guess which one applies ahead of time.
    """
    threshold = _otsu_threshold(img)
    binarized = img.point(lambda p: 255 if p > threshold else 0)
    candidates = [img, ImageOps.invert(img), binarized, ImageOps.invert(binarized)]

    best_text, best_score = "", -1
    for candidate in candidates:
        try:
            text = pytesseract.image_to_string(candidate, lang=lang)
        except Exception:
            continue
        score = _ocr_score(text)
        if score > best_score:
            best_text, best_score = text, score
    return best_text


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
        return _ocr_best_of(_preprocess_for_ocr(img), lang)


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
                        images.append(_to_png_bytes(Image.open(io.BytesIO(raw))))
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
                    images.append(_to_png_bytes(Image.open(io.BytesIO(raw))))
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
            return [_to_png_bytes(img)]
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
