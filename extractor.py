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
import sys
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
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import pymupdf
except Exception:
    pymupdf = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

try:
    # Adds HEIC/HEIF decoding support to Pillow (common iOS/macOS screenshot
    # and photo format that Pillow can't open on its own).
    import pillow_heif

    pillow_heif.register_heif_opener()
    IMAGE_EXTS |= {".heic", ".heif"}
except Exception:
    pass


def extract_text_from_pdf(path: Path) -> str:
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

    pages = convert_from_path(str(path))
    out_lines = []
    for i, page in enumerate(pages, start=1):
        txt = pytesseract.image_to_string(page)
        out_lines.append(f"\n--- PAGE {i} ---\n")
        out_lines.append(txt)
    return "\n".join(out_lines)


def extract_text_from_image(path: Path) -> str:
    if pytesseract is None or Image is None:
        raise RuntimeError("Missing image OCR dependencies: install pytesseract and pillow")
    img = Image.open(path)
    # pytesseract only recognizes a handful of PIL format tags (PNG, JPEG,
    # BMP, ...); formats like HEIF/WEBP fall outside that list and get
    # rejected, so force it to re-encode as PNG instead.
    img.format = None
    return pytesseract.image_to_string(img)


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
            for page in convert_from_path(str(path)):
                images.append(_to_png_bytes(page))
    finally:
        doc.close()

    return images


def extract_images_from_file(path: Path) -> List[bytes]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_images_from_pdf(path)
    if suffix in IMAGE_EXTS:
        if Image is None:
            raise RuntimeError("Missing image dependency: install pillow")
        return [_to_png_bytes(Image.open(path))]
    raise ValueError(f"Unsupported file type: {suffix}")


def process_path(p: Path) -> Optional[str]:
    if not p.exists():
        print(f"Not found: {p}", file=sys.stderr)
        return None

    if p.is_dir():
        outputs = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and (f.suffix.lower() in IMAGE_EXTS or f.suffix.lower() == ".pdf"):
                txt = process_path(f)
                if txt:
                    outputs.append(f"\n===== {f} =====\n")
                    outputs.append(txt)
        return "\n".join(outputs)

    # single file
    if p.suffix.lower() == ".pdf":
        return extract_text_from_pdf(p)
    if p.suffix.lower() in IMAGE_EXTS:
        return extract_text_from_image(p)

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
