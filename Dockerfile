# ─────────────────────────────────────────────────────────────
# SafeScan — container image for the Flask web app.
# Serves: SVM text classifier (+ SHAP), model comparison, fairness audit,
#         AND the zero-shot CLIP meme tab (pytesseract OCR + Grad-CAM).
# Deep-learning deps are CPU-only wheels to keep the image small enough for a
# free Hugging Face Space (cpu-basic). BERT/Bi-LSTM live serving is not included
# here — their metrics show in the comparison tab once you train them.
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps: tesseract = meme OCR; libgl1/libglib = OpenCV runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dir for the CLIP weights (writable by any runtime UID on HF Spaces).
ENV HF_HOME=/app/hf_cache \
    HOST=0.0.0.0 PORT=5000 PYTHONUNBUFFERED=1
RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

# 1) Core dependencies (Flask + SVM + fairness).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) Multimodal deps for the live meme tab — CPU-only torch (≈200 MB vs ≈2 GB CUDA).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1
RUN pip install --no-cache-dir \
        transformers==4.46.3 pytesseract==0.3.13 \
        opencv-python==4.10.0.84 Pillow==10.4.0

# 3) Pre-bake CLIP so the first meme request is instant (no runtime download).
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Application code + data.
COPY src/ ./src/
COPY templates/ ./templates/
COPY app.py ./
COPY Data/ ./Data/
RUN mkdir -p models && chmod -R 777 models

EXPOSE 5000

# Train the SVM + fairness audit at build so the app has something to serve.
RUN python -m src.train_svm && python -m src.fairness

CMD ["python", "app.py"]
