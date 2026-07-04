---
title: SafeScan
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 5000
pinned: false
license: other
short_description: Explainable multimodal hate-speech detection (text + memes)
---

# 🛡️ SafeScan — Explainable Multimodal Hate Speech Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.46-FFD21E)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blue)
[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-yellow)](https://casanovaabhishek14-safescan.hf.space)

SafeScan classifies **text** (Normal / Offensive / Hate speech) and **memes**
(Hateful / Non-hateful), and — crucially — *explains itself*: SHAP-style
word-level attributions for text, Grad-CAM heatmaps for images, plus a
counterfactual **fairness audit** that measures gender bias. Three text models
(SVM, Bi-LSTM, BERT) and a zero-shot CLIP meme pipeline are trained through one
reproducible command and served through a single-page Flask web app.

> MSc Data Science Research Project · University of Leicester · Module MA7443
> Supervisor: Dr. Hammad Afzal

---

## 🌐 Live demo

### ▶️ **[Try it live → casanovaabhishek14-safescan.hf.space](https://casanovaabhishek14-safescan.hf.space)**

Hosted free on Hugging Face Spaces (Docker). **All four tabs are live** — a text classifier
with a **SVM / BERT** model selector (SHAP for SVM, occlusion attribution for BERT), the
CLIP + OCR meme tab (Grad-CAM), model comparison, and the fairness audit. Retrain the models
for free on a GPU — see [docs/TRAINING.md](docs/TRAINING.md).

Or run it locally (see [Quick start](#-quick-start-3-commands)) and open **http://localhost:5000**.

> On macOS, port 5000 is taken by the AirPlay Receiver, so the app serves at
> **http://localhost:5001** there (`PORT=5001 python app.py`).

---

## 🖼️ Screenshot

![SafeScan UI](docs/screenshot.png)
<!-- Placeholder — run the app (see below), then drop a screenshot at docs/screenshot.png -->

---

## 🚀 Quick start (3 commands)

```bash
pip install -r requirements.txt        # 1. core deps (Flask + SVM + fairness)
python -m src.run_training             # 2. train the SVM & write metrics/fairness
python app.py                          # 3. serve the app  →  http://localhost:5000
```

That's the lightweight path: it trains the **SVM** and powers all four UI tabs
(text analysis, model comparison, fairness audit; the meme tab shows a friendly
"install extras" message).

To enable **Bi-LSTM, BERT, and the CLIP meme pipeline** (larger, GPU-recommended):

```bash
pip install -r requirements-ml.txt     # torch, tensorflow, transformers, pytesseract…
# macOS: brew install tesseract   |   Ubuntu: sudo apt-get install -y tesseract-ocr
python -m src.run_training             # now trains all three text models
```

> **macOS note:** port 5000 is often taken by the AirPlay Receiver. Either disable
> it (System Settings → General → AirDrop & Handoff) or run `PORT=5001 python app.py`.

---

## 🗂️ Project structure

```
SafeScanTool/
├── app.py                     # Flask app — serves UI + /analyze/text, /analyze/meme, APIs
├── templates/index.html       # Single-page frontend (4 tabs, no build step)
├── src/
│   ├── data_utils.py          # Shared load / clean / 80-10-10 split / class weights
│   ├── eval_utils.py          # Metrics + models/metrics.json registry
│   ├── train_svm.py           # SVM (TF-IDF word+char + LinearSVC)
│   ├── train_bilstm.py        # Bi-LSTM (GloVe optional, early stopping, checkpointing)
│   ├── train_bert.py          # BERT fine-tune (weighted loss, early stopping)
│   ├── run_training.py        # Train all three sequentially
│   ├── clip_meme_pipeline.py  # pytesseract OCR + CLIP zero-shot prompt ensembles
│   ├── explainability.py      # SVM SHAP + BERT occlusion + CLIP Grad-CAM
│   ├── fairness.py            # Counterfactual gender-bias audit
│   └── config.py             # Env-driven config (no hard-coded secrets)
├── scripts/upload_models.py   # Push trained models/ to a Hugging Face model repo
├── docs/
│   ├── TRAINING.md            # Train BERT / Bi-LSTM free on a GPU (Colab/Kaggle)
│   └── DEPLOYMENT.md          # HF Spaces architecture + how to deploy your own
├── Data/updated_hatexplain_data.csv   # 44,931 labelled posts (3-class)
├── notebooks/                 # Original research notebooks (secrets scrubbed)
├── models/                    # Trained artifacts + metrics.json / fairness.json (git-ignored)
├── requirements.txt           # Core (pinned)
├── requirements-ml.txt        # Deep-learning / multimodal extras (pinned)
├── Dockerfile                 # Container build (pulls trained models from HF at build)
└── .env.example               # Configuration template
```

---

## 🧠 Models & results

Text is a 3-class problem on a merged **44,931-post** corpus
(offensive 54.9% / normal 28.7% / hate speech 16.4% — imbalanced, so every model
uses **balanced class weights**). All models share an identical stratified
**80/10/10** train/val/test split.

| Model | Accuracy | Macro F1 | Weighted F1 | Serving |
|---|---|---|---|---|
| **BERT** (bert-base-uncased, weighted loss) | **0.80** | **0.77** | **0.80** | live (text) |
| Bi-LSTM (trainable emb, early stopping) | 0.77 | 0.75 | 0.78 | comparison only¹ |
| SVM (TF-IDF word+char, LinearSVC) | 0.77 | 0.75 | 0.78 | live (text) |
| CLIP + OCR (zero-shot, memes) | — | — | — | live (memes)² |

> ¹ Bi-LSTM's metrics show in the comparison tab but it isn't served for inference —
> that would bundle TensorFlow into the container. BERT is the stronger text model.
> ² CLIP is zero-shot (binary hateful/non-hateful) — no supervised accuracy is reported.
>
> All numbers are measured on the held-out **test** split (seed 42). The web app's
> comparison tab reads `models/metrics.json` **live**, so it only ever shows what
> training actually produced — never hard-coded values.

**Fairness (measured on the SVM):** a counterfactual gender-swap audit over 3,000
real posts gives an overall **label flip-rate of ~4.8%** (normal 6.9%,
offensive 2.9%, hate speech 6.8%) — i.e. how often a prediction changes when only
gendered terms are swapped. Regenerate with `python -m src.fairness`.

---

## 🔬 What each piece does

- **Text analysis** (`/analyze/text`, `model` = `svm` | `bert`) → predicted label,
  class probabilities, and signed word-level attributions.
  - **SVM** — linear over TF-IDF, so each token's contribution is computed exactly
    (`coef × tf-idf`): the SHAP value of a linear model, with no sampling. Fast.
  - **BERT** — non-linear, so words are attributed by **occlusion** (mask each word,
    measure the drop in the predicted-class probability). Most accurate model.
- **Meme analysis** (`/analyze/meme`) → pytesseract extracts the caption, CLIP
  scores the image against **engineered prompt ensembles** (many "hateful" vs
  "non-hateful" templates, averaged) plus caption-aware prompts, and Grad-CAM
  overlays the image regions driving the "hateful" score.
- **Fairness audit** → counterfactual gender-term substitution + flip-rate.

---

## ⚙️ Tech stack

**ML/NLP:** scikit-learn · TensorFlow · PyTorch · HuggingFace Transformers · CLIP
`openai/clip-vit-base-patch32` · GloVe
**Explainability & fairness:** linear SHAP attributions · Grad-CAM · counterfactual testing
**OCR:** pytesseract (local, no API keys)
**App:** Flask · vanilla-JS single-page UI

---

## 🚢 Deployment

The live demo is a **Docker Space** on Hugging Face (free `cpu-basic`). The `Dockerfile`
downloads the trained weights (SVM + BERT + `metrics.json`) from a separate **HF model
repo** at build time — so code (GitHub) and model artifacts version independently, and
nothing large is committed to git.

Full details — build flow, deploying your own, and refreshing models on a running
Space — are in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

---

## 📊 Datasets

The bundled `Data/updated_hatexplain_data.csv` is the merged text corpus. The
Facebook Hateful Memes images require registration with Meta AI and are **not**
included — see [`Data/README.md`](Data/README.md).

---

## 🔐 Security note

Earlier notebook versions contained hard-coded API keys. These were removed **and
scrubbed from the full git history** (`git filter-repo`, force-pushed); the exposed
keys must still be treated as **compromised and rotated**. All runtime configuration
now flows through `.env` (see `.env.example`), which is git-ignored. The default meme
pipeline is fully local (pytesseract) and needs no keys.

---

## ⚠️ Limitations

Automated hate-speech detection is imperfect and context-dependent. These models
carry dataset biases (documented by the fairness audit), can misfire on reclaimed
language, irony, and dialect, and should support — not replace — human moderation.

---

## 👥 Authors

| Name | Email |
|---|---|
| Nikhil Ayyappan Nair | nan16@student.le.ac.uk |
| Abhishek Kumar Pal | akp33@student.le.ac.uk |
| Sahil Dinesh Padwal | sdp20@student.le.ac.uk |

**Supervisor:** Dr. Hammad Afzal · School of Computing and Mathematical Sciences,
University of Leicester

## 📄 License

For academic purposes. Please cite appropriately if building upon this work.
