Text Extractor (PDF + Images)
=================================

Extract text from PDFs and common image formats (including screenshots), with OCR fallback for scanned/image-based PDFs. Ships as both a CLI tool and a small web app with a drag-and-drop upload UI.

Features
--------
- PDF text extraction via `pdfminer.six`, with OCR fallback (`pdf2image` + `pytesseract`) for scanned/image-only PDFs.
- Image OCR for PNG, JPG/JPEG, TIFF, BMP, GIF, WEBP, and HEIC/HEIF (iOS/macOS screenshot format).
- Extract embedded images out of a PDF, or the image itself for a plain image upload, as downloadable PNGs.
- Web UI (FastAPI backend + static frontend) with drag-and-drop upload, extracted-text panel, and an image gallery.

Installation
------------

1. Install system dependencies (macOS):

```sh
brew install tesseract poppler
```

2. Install Python dependencies (prefer a virtualenv):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CLI usage
---------

Extract text from a single file:

```sh
python extractor.py /path/to/file.pdf
```

Extract a directory recursively and write output to a file:

```sh
python extractor.py /path/to/folder -o output.txt
```

Notes
-----
- For best OCR results ensure `tesseract` is installed and on PATH.
- `pdf2image` requires `poppler` (installed via Homebrew on macOS).

Web app (upload UI + API)
--------------------------

A FastAPI backend serves the extraction API and the frontend:

```sh
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000 in your browser. Upload a PDF or image and:
- **Extract text** — runs OCR/text extraction and shows the result (with an image preview when the upload is an image).
- **Extract images** — pulls out embedded images from a PDF (or the image itself), downloadable individually from the gallery.

### API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/extract-text` | POST | `multipart/form-data` file upload → `{ text, filename }` |
| `/api/extract-images` | POST | `multipart/form-data` file upload → `{ images: [{ filename, data_url }], filename }` |

License
-------
[MIT](LICENSE)
