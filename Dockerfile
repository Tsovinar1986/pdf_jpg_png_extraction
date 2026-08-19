FROM python:3.12-slim

# Tesseract (+ Armenian/Russian language data) and Poppler are the same
# system dependencies documented per-OS in the README's Installation
# section — baking them into the image is what makes the container behave
# identically everywhere, instead of depending on what the host happens to
# have on PATH. libglib2.0-0 covers a common opencv-python-headless import
# failure on slim Debian images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hye \
    tesseract-ocr-rus \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY extractor.py .
COPY frontend/ frontend/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
