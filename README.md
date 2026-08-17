Text Extractor (PDF + Images)
=================================

Extract text from PDFs and common image formats (including screenshots), with OCR fallback for scanned/image-based PDFs. Ships as both a CLI tool and a small web app with a drag-and-drop upload UI.

Features
--------
- PDF text extraction via `pdfminer.six`, with OCR fallback (`pdf2image` + `pytesseract`) for scanned/image-only PDFs.
- Image OCR for PNG, JPG/JPEG, TIFF, BMP, GIF, WEBP, and HEIC/HEIF (iOS/macOS screenshot format), recognizing Armenian (Classical, Western, and Eastern), Russian, and English by default (`OCR_LANGS` env var to change).
- XLSX/XLSM, DOCX, CSV, and TXT are read directly (not OCR'd) — spreadsheets are converted to tab-separated text per sheet, DOCX paragraphs/tables are read as-is.
- Extract embedded images out of a PDF/XLSX/DOCX, or the image itself for a plain image upload, as downloadable PNGs.
- Web UI (FastAPI backend + static frontend) with drag-and-drop upload, extracted-text panel, and an image gallery.

Installation
------------

1. Install system dependencies (Tesseract for OCR, Poppler for PDF rendering):

**macOS**

```sh
brew install tesseract poppler
brew install tesseract-lang   # Russian/etc. language packs
```

**Linux (Debian/Ubuntu)**

```sh
sudo apt-get install tesseract-ocr tesseract-ocr-rus poppler-utils
```

**Windows**

- Install Tesseract from the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki). **During setup, expand "Additional language data" and check Russian** (`rus`) — it's not installed by default, only English is. This step is easy to miss; if you already installed without it, rerun the installer and check "Modify" to add it.
  - Alternatively, download `rus.traineddata` directly from the [tessdata repo](https://github.com/tesseract-ocr/tessdata) and drop it into Tesseract-OCR's `tessdata` folder (e.g. `C:\Program Files\Tesseract-OCR\tessdata`) — useful if you can't rerun the installer.
- Install Poppler for Windows from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases).
- Either add both tools' folders to your `PATH`, or point the app at them directly with environment variables (no PATH changes needed):

  ```powershell
  setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
  setx POPPLER_PATH "C:\poppler\Library\bin"
  ```

  (If Tesseract was installed to its default location, `TESSERACT_CMD` is auto-detected and can be skipped.)

**Armenian (all platforms)**

Armenian isn't in the standard Tesseract language packs above, and the app doesn't use Tesseract's own `hye` for it. Instead it defaults to [`hye-calfa-n`](https://github.com/calfa-co/hye-tesseract), a community model trained on Classical, Western, *and* Eastern Armenian — `hye` only covers Eastern Armenian orthography and reads Western/Classical text noticeably worse.

1. Download `hye-calfa-n.traineddata` from the [hye-tesseract repo](https://github.com/calfa-co/hye-tesseract).
2. Place it in Tesseract's `tessdata` folder, next to `hye.traineddata`:
   - macOS (Homebrew): `/opt/homebrew/share/tessdata/` (Intel Macs: `/usr/local/share/tessdata/`)
   - Linux: `/usr/share/tesseract-ocr/*/tessdata/` (path varies by distro — check with `find / -name hye.traineddata 2>/dev/null`)
   - Windows: `C:\Program Files\Tesseract-OCR\tessdata\`

Don't install/enable stock `hye` *alongside* `hye-calfa-n` in `OCR_LANGS` — two models for the same script compete rather than help. If you'd rather skip this extra step and use stock Eastern-Armenian-only `hye` (`brew install tesseract-lang` / `apt install tesseract-ocr-hye` / check it during the Windows installer), set `OCR_LANGS=hye+rus+eng` to override the default.

To check which languages/models are actually installed:

```sh
tesseract --list-langs                                         # macOS/Linux
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs   # Windows
```

If a language the app needs (`OCR_LANGS`, default `hye-calfa-n+rus+eng`) isn't listed, extraction now fails with a clear error naming exactly what's missing, instead of silently misreading that script as English/Cyrillic gibberish.

2. Install Python dependencies (prefer a virtualenv):

**macOS/Linux**

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
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
- OCR defaults to Armenian + Russian + English (`hye-calfa-n+rus+eng`). See the "Armenian (all platforms)" section above for the Armenian model; `brew install tesseract-lang` covers Russian; check what's installed with `tesseract --list-langs`.

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
