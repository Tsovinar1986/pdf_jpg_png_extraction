import base64
import contextlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from extractor import (
    ALL_SUPPORTED_EXTS,
    IMAGE_EXTS,
    extract_images_from_file,
    extract_text_from_path,
    normalize_dark_image_to_paper,
)

app = FastAPI(title="PDF/Image Text Extraction Service")

FRONTEND_DIR = ROOT_DIR / "frontend"

# 50MB was too tight for a book-length scanned PDF (hundreds of
# high-res pages easily exceeds that). 500MB comfortably covers that
# case while still refusing truly unbounded uploads; override per-
# deployment with MAX_UPLOAD_MB if you need something else.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

SUPPORTED_EXTS = ALL_SUPPORTED_EXTS


class ExtractionResponse(BaseModel):
    text: str
    filename: str
    normalized_image: Optional[str] = None


class ExtractedImage(BaseModel):
    filename: str
    data_url: str


class ImageExtractionResponse(BaseModel):
    images: List[ExtractedImage]
    filename: str


@contextlib.asynccontextmanager
async def _save_upload_to_temp(file: UploadFile) -> Iterator[Path]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        size = 0
        while chunk := await file.read(CHUNK_SIZE):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(status_code=413, detail="File too large")
            tmp.write(chunk)

    try:
        yield Path(tmp_path)
    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/extract-text", response_model=ExtractionResponse)
async def extract_text(file: UploadFile = File(...)):
    async with _save_upload_to_temp(file) as tmp_path:
        try:
            text = extract_text_from_path(tmp_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        normalized_data_url = None
        if tmp_path.suffix.lower() in IMAGE_EXTS:
            try:
                with Image.open(tmp_path) as img:
                    normalized = normalize_dark_image_to_paper(img)
                if normalized is not None:
                    buf = io.BytesIO()
                    normalized.save(buf, format="PNG")
                    normalized_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception:
                pass  # the normalized preview is a bonus, not required for a successful extraction

    return ExtractionResponse(text=text or "", filename=file.filename or "", normalized_image=normalized_data_url)


@app.post("/api/extract-images", response_model=ImageExtractionResponse)
async def extract_images(file: UploadFile = File(...)):
    async with _save_upload_to_temp(file) as tmp_path:
        try:
            images = extract_images_from_file(tmp_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    stem = Path(file.filename or "image").stem
    payload = [
        ExtractedImage(
            filename=f"{stem}_{i + 1}.png",
            data_url="data:image/png;base64," + base64.b64encode(data).decode("ascii"),
        )
        for i, data in enumerate(images)
    ]
    return ImageExtractionResponse(images=payload, filename=file.filename or "")


def _safe_download_filename(name: str, ext: str) -> str:
    stem = Path(name or "extracted-text").stem or "extracted-text"
    stem = re.sub(r'[\r\n"/\\]+', "_", stem)
    return f"{stem}.{ext}"


@app.post("/api/download-text")
async def download_text(text: str = Form(...), filename: str = Form("extracted-text.txt")):
    safe_name = _safe_download_filename(filename, "txt")
    return PlainTextResponse(
        text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@app.post("/api/download-docx")
async def download_docx(text: str = Form(...), filename: str = Form("extracted-text.docx")):
    safe_name = _safe_download_filename(filename, "docx")
    document = Document()
    for line in (text or "").split("\n"):
        document.add_paragraph(line)
    buf = io.BytesIO()
    document.save(buf)
    return Response(
        buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


class NoCacheStaticFiles(StaticFiles):
    """Force revalidation on every request so frontend edits (HTML/CSS/JS)
    show up on a normal reload instead of needing a hard-refresh to bust
    the browser's cache."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
