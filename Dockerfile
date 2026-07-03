# ─────────────────────────────────────────────────────────────
# SafeScan — container image for the Flask web app + SVM/fairness.
# Builds the lightweight core by default. To include the multimodal
# extras (BERT/CLIP/OCR), uncomment the requirements-ml.txt line below
# (large image, GPU recommended at runtime).
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# tesseract-ocr is the system dependency for the meme OCR pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir -r requirements-ml.txt   # enable for BERT/CLIP/OCR

# Copy the application code and data.
COPY src/ ./src/
COPY templates/ ./templates/
COPY app.py ./
COPY Data/ ./Data/

# Model artifacts are produced by training; mount ./models at runtime,
# or run `python -m src.run_training` inside the container to populate it.
RUN mkdir -p models

ENV HOST=0.0.0.0 PORT=5000 PYTHONUNBUFFERED=1
EXPOSE 5000

# Train the SVM + fairness audit at build time so the app has something to serve.
# (Comment out if you prefer to mount pre-trained models instead.)
RUN python -m src.train_svm && python -m src.fairness

CMD ["python", "app.py"]
