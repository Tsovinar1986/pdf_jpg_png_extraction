const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const clearFile = document.getElementById("clearFile");
const submitBtn = document.getElementById("submitBtn");
const submitLabel = document.getElementById("submitLabel");
const progress = document.getElementById("progress");
const progressText = document.getElementById("progressText");
const progressBarFill = document.getElementById("progressBarFill");
const errorBox = document.getElementById("errorBox");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");
const resultImagePreview = document.getElementById("resultImagePreview");
const normalizedPreviewBox = document.getElementById("normalizedPreviewBox");
const normalizedPreviewImg = document.getElementById("normalizedPreviewImg");
const normalizedPreviewLink = document.getElementById("normalizedPreviewLink");
const copyBtn = document.getElementById("copyBtn");
const copyBtnLabel = document.getElementById("copyBtnLabel");
const downloadBtn = document.getElementById("downloadBtn");
const downloadForm = document.getElementById("downloadForm");
const downloadTextInput = document.getElementById("downloadTextInput");
const downloadFilenameInput = document.getElementById("downloadFilenameInput");
const downloadDocxBtn = document.getElementById("downloadDocxBtn");
const downloadDocxForm = document.getElementById("downloadDocxForm");
const downloadDocxTextInput = document.getElementById("downloadDocxTextInput");
const downloadDocxFilenameInput = document.getElementById("downloadDocxFilenameInput");

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
  normalizedPreviewBox.classList.add("hidden");
  normalizedPreviewImg.src = "";
  normalizedPreviewLink.href = "#";
}
let selectedFile = null;

function setFile(file) {
  selectedFile = file;
  if (file) {
    fileName.textContent = file.name;
    fileInfo.classList.remove("hidden");
    submitBtn.disabled = false;
  } else {
    fileInfo.classList.add("hidden");
    submitBtn.disabled = true;
  }
  hideError();
  resultBox.classList.add("hidden");
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

function setLoading(isLoading) {
  submitBtn.disabled = isLoading || !selectedFile;
  progress.classList.toggle("hidden", !isLoading);
  submitLabel.textContent = isLoading ? "Extracting..." : "Extract text";
  if (isLoading) {
    setUploadProgress(0);
  }
}

function setUploadProgress(fraction) {
  const pct = Math.round(fraction * 100);
  progressBarFill.classList.remove("indeterminate");
  progressBarFill.style.width = `${pct}%`;
  progressText.textContent = `Uploading… ${pct}%`;
}

function setProcessing() {
  progressBarFill.classList.add("indeterminate");
  progressText.textContent = "Extracting… this can take a moment for scanned pages";
}

function uploadFile(url, file, onUploadProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) onUploadProgress(e.loaded / e.total);
    });
    xhr.addEventListener("load", () => {
      const body = xhr.response || {};
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body);
      } else {
        reject(new Error(body.detail || `Error (${xhr.status})`));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("Network error")));
    const formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
  });
}

let lastResultFilename = "";

submitBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  resultBox.classList.add("hidden");
  setLoading(true);

  try {
    const data = await uploadFile("/api/extract-text", selectedFile, (fraction) => {
      if (fraction >= 1) {
        setProcessing();
      } else {
        setUploadProgress(fraction);
      }
    });
    resultText.value = data.text || "";
    lastResultFilename = data.filename || selectedFile.name;

    clearPreview();
    if (isImageFile(selectedFile)) {
      previewObjectUrl = URL.createObjectURL(selectedFile);
      resultImagePreview.src = previewObjectUrl;
      resultImagePreview.classList.remove("hidden");
    }

    if (data.normalized_image) {
      normalizedPreviewImg.src = data.normalized_image;
      normalizedPreviewLink.href = data.normalized_image;
      normalizedPreviewLink.download = `${lastResultFilename.replace(/\.[^.]+$/, "")}-cleaned.png`;
      normalizedPreviewBox.classList.remove("hidden");
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

copyBtn.addEventListener("click", async () => {
  if (!resultText.value) return;
  try {
    await navigator.clipboard.writeText(resultText.value);
    const original = copyBtnLabel.textContent;
    copyBtnLabel.textContent = "Copied ✓";
    setTimeout(() => (copyBtnLabel.textContent = original), 1500);
  } catch {
    resultText.select();
    document.execCommand("copy");
  }
});

downloadBtn.addEventListener("click", () => {
  if (!resultText.value) return;
  // Submitted as a real form POST (not a scripted click on a blob: <a>)
  // so the download works consistently across browsers, including
  // Safari/WebKit, which doesn't reliably honor scripted blob downloads.
  downloadTextInput.value = resultText.value;
  downloadFilenameInput.value = lastResultFilename || "extracted-text.txt";
  downloadForm.submit();
});

downloadDocxBtn.addEventListener("click", () => {
  if (!resultText.value) return;
  downloadDocxTextInput.value = resultText.value;
  downloadDocxFilenameInput.value = lastResultFilename || "extracted-text.docx";
  downloadDocxForm.submit();
});
