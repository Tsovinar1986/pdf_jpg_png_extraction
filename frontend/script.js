const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const clearFile = document.getElementById("clearFile");
const submitBtn = document.getElementById("submitBtn");
const submitLabel = document.getElementById("submitLabel");
const imagesBtn = document.getElementById("imagesBtn");
const imagesLabel = document.getElementById("imagesLabel");
const progress = document.getElementById("progress");
const progressText = document.getElementById("progressText");
const errorBox = document.getElementById("errorBox");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");
const resultImagePreview = document.getElementById("resultImagePreview");
const copyBtn = document.getElementById("copyBtn");

const IMAGE_EXTS = new Set([
  ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".heic", ".heif",
]);

function isImageFile(file) {
  const dot = file.name.lastIndexOf(".");
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  return IMAGE_EXTS.has(ext);
}

let previewObjectUrl = null;

function clearPreview() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
  resultImagePreview.classList.add("hidden");
  resultImagePreview.src = "";
}
const imagesBox = document.getElementById("imagesBox");
const imageGallery = document.getElementById("imageGallery");
const imageCount = document.getElementById("imageCount");

let selectedFile = null;

function setFile(file) {
  selectedFile = file;
  if (file) {
    fileName.textContent = file.name;
    fileInfo.classList.remove("hidden");
    submitBtn.disabled = false;
    imagesBtn.disabled = false;
  } else {
    fileInfo.classList.add("hidden");
    submitBtn.disabled = true;
    imagesBtn.disabled = true;
  }
  hideError();
  resultBox.classList.add("hidden");
  imagesBox.classList.add("hidden");
  imageGallery.innerHTML = "";
  clearPreview();
}

fileInput.addEventListener("change", () => {
  setFile(fileInput.files[0] || null);
});

clearFile.addEventListener("click", (e) => {
  e.preventDefault();
  fileInput.value = "";
  setFile(null);
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    setFile(file);
  }
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function setLoading(isLoading, mode) {
  submitBtn.disabled = isLoading || !selectedFile;
  imagesBtn.disabled = isLoading || !selectedFile;
  progress.classList.toggle("hidden", !isLoading);
  submitLabel.textContent = isLoading && mode === "text" ? "Extracting..." : "Extract text";
  imagesLabel.textContent = isLoading && mode === "images" ? "Extracting..." : "Extract images";
  if (isLoading) {
    progressText.textContent =
      mode === "images"
        ? "Pulling images out… this can take a moment for large PDFs"
        : "Extracting… this can take a moment for scanned pages";
  }
}

submitBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  resultBox.classList.add("hidden");
  setLoading(true, "text");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/api/extract-text", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Error (${response.status})`);
    }

    const data = await response.json();
    resultText.value = data.text || "";

    clearPreview();
    if (isImageFile(selectedFile)) {
      previewObjectUrl = URL.createObjectURL(selectedFile);
      resultImagePreview.src = previewObjectUrl;
      resultImagePreview.classList.remove("hidden");
    }

    resultBox.classList.remove("hidden");
    if (!data.text || !data.text.trim()) {
      showError("No text was found in this file.");
    }
  } catch (err) {
    showError(err.message || "An unknown error occurred.");
  } finally {
    setLoading(false);
  }
});

imagesBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  imagesBox.classList.add("hidden");
  imageGallery.innerHTML = "";
  setLoading(true, "images");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/api/extract-images", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Error (${response.status})`);
    }

    const data = await response.json();
    const images = data.images || [];
    imageCount.textContent = String(images.length);
    images.forEach((img) => {
      const tile = document.createElement("a");
      tile.className = "image-tile";
      tile.href = img.data_url;
      tile.download = img.filename;

      const thumb = document.createElement("img");
      thumb.src = img.data_url;
      thumb.alt = img.filename;

      const label = document.createElement("span");
      label.textContent = img.filename;

      tile.appendChild(thumb);
      tile.appendChild(label);
      imageGallery.appendChild(tile);
    });

    imagesBox.classList.remove("hidden");
    if (images.length === 0) {
      showError("No images were found in this file.");
    }
  } catch (err) {
    showError(err.message || "An unknown error occurred.");
  } finally {
    setLoading(false);
  }
});

copyBtn.addEventListener("click", async () => {
  if (!resultText.value) return;
  try {
    await navigator.clipboard.writeText(resultText.value);
    const original = copyBtn.textContent;
    copyBtn.textContent = "Copied ✓";
    setTimeout(() => (copyBtn.textContent = original), 1500);
  } catch {
    resultText.select();
    document.execCommand("copy");
  }
});
