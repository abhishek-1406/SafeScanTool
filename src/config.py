"""Central configuration, loaded from environment / .env.

Nothing secret is hard-coded (that was one of the audit's biggest findings).
Model paths and runtime knobs all come from the environment, with sensible
defaults that point at the local ``models/`` directory. ``.env.example`` documents
every variable.
"""
from __future__ import annotations

import os
from pathlib import Path

# Optional: load a local .env if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional; env vars still work without it
    pass

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT / "Data" / "updated_hatexplain_data.csv"))

# Text (SVM) artifacts — the model the web app serves for /analyze/text.
SVM_MODEL_PATH = Path(os.environ.get("SVM_MODEL_PATH", MODELS_DIR / "svm_model.pkl"))
SVM_VECTORIZER_PATH = Path(os.environ.get("SVM_VECTORIZER_PATH", MODELS_DIR / "svm_vectorizer.pkl"))

# Registry files produced by training / auditing (read by the comparison tabs).
METRICS_PATH = Path(os.environ.get("METRICS_PATH", MODELS_DIR / "metrics.json"))
FAIRNESS_PATH = Path(os.environ.get("FAIRNESS_PATH", MODELS_DIR / "fairness.json"))

# Flask runtime.
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

# Upload limits for the meme route.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
