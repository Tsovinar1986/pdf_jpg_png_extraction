import base64
import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from extractor import ALL_SUPPORTED_EXTS, extract_images_from_file, extract_text_from_path

app = FastAPI(title="PDF/Image Text Extraction Service")

FRONTEND_DIR = ROOT_DIR / "frontend"

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

SUPPORTED_EXTS = ALL_SUPPORTED_EXTS


class ExtractionResponse(BaseModel):
    text: str
    filename: str


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

    return ExtractionResponse(text=text or "", filename=file.filename or "")


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


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
