"""SafeScan Flask web app — serves the UI and the analysis API on :5000.

Routes
------
GET  /                 → the single-page frontend (templates/index.html)
POST /analyze/text     → {label, probabilities, words:[{word, weight}], ...}   (SHAP-style)
POST /analyze/meme     → {label, confidence, ocr_text, overlay_png_base64}     (Grad-CAM)
GET  /api/metrics      → models/metrics.json  (Model-comparison tab)
GET  /api/fairness     → models/fairness.json (Fairness-audit tab)
GET  /health           → liveness + which models are loaded

Heavy models are loaded lazily and cached: the SVM loads on first text request,
CLIP on first meme request. Missing artifacts/deps produce a clear JSON error
(HTTP 503) instead of crashing the server.
"""
from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src import config
from src.explainability import explain_text, gradcam_clip

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
CORS(app)


# --------------------------------------------------------------------------- #
# Lazy, cached model loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_svm():
    if not config.SVM_MODEL_PATH.exists():
        raise FileNotFoundError(
            "SVM model not found. Train it first: python -m src.train_svm"
        )
    return joblib.load(config.SVM_MODEL_PATH), joblib.load(config.SVM_VECTORIZER_PATH)


@lru_cache(maxsize=1)
def _load_meme_classifier():
    from src.clip_meme_pipeline import get_classifier

    return get_classifier()


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze/text", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Provide a non-empty 'text' field."}), 400
    try:
        model, vectorizer = _load_svm()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    top_k = min(max(int(data.get("top_k", 15)), 1), 40)
    result = explain_text(model, vectorizer, text, top_k=top_k)
    result["input"] = text
    return jsonify(result)


@app.route("/analyze/meme", methods=["POST"])
def analyze_meme():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image under the 'image' form field."}), 400
    file = request.files["image"]
    ext = Path(secure_filename(file.filename or "")).suffix.lower()
    if ext not in config.ALLOWED_IMAGE_EXTS:
        return jsonify({"error": f"Unsupported file type '{ext}'."}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        file.save(tmp.name)
        try:
            classifier = _load_meme_classifier()
        except ImportError as e:
            return jsonify({
                "error": "Meme analysis needs the multimodal extras "
                         "(torch, transformers, pytesseract). "
                         f"Missing: {e.name}. See requirements.txt / README."
            }), 503
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"Failed to load CLIP model: {e}"}), 503
        result = gradcam_clip(classifier, tmp.name)
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    metrics = _read_json(config.METRICS_PATH)
    if metrics is None:
        return jsonify({"error": "No metrics yet. Run: python -m src.run_training"}), 404
    return jsonify(metrics)


@app.route("/api/fairness")
def api_fairness():
    fairness = _read_json(config.FAIRNESS_PATH)
    if fairness is None:
        return jsonify({"error": "No audit yet. Run: python -m src.fairness"}), 404
    return jsonify(fairness)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "svm_available": config.SVM_MODEL_PATH.exists(),
        "metrics_available": config.METRICS_PATH.exists(),
        "fairness_available": config.FAIRNESS_PATH.exists(),
    })


if __name__ == "__main__":
    print(f"🛡️  SafeScan running on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
