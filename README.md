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

# SafeScan: Explainable Multimodal Hate Speech Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
[![Live Demo](https://img.shields.io/badge/demo-Hugging%20Face%20Spaces-yellow?logo=huggingface)](https://casanovaabhishek14-safescan.hf.space)

SafeScan classifies social-media text as normal, offensive, or hate speech, and
classifies memes as hateful or non-hateful. Alongside each prediction it shows why the
decision was made (word-level attributions for text, a heatmap for images) and reports a
gender-bias audit, so the models can be inspected rather than trusted blindly.

It was built as an MSc Data Science research project at the University of Leicester
(module MA7443, supervised by Dr. Hammad Afzal). Four models are benchmarked across two
modalities, and the whole thing is served through a small Flask web app.

## Live demo

Deployed on Hugging Face Spaces: **https://casanovaabhishek14-safescan.hf.space**

The app has four tabs: text analysis (pick SVM or BERT), meme analysis (OCR + CLIP with a
Grad-CAM overlay), a model-comparison table, and the fairness audit. To run it yourself,
see the quick start below.

## Quick start

Three commands run the app with the SVM model and the fairness audit:

```bash
pip install -r requirements.txt
python -m src.run_training
python app.py                     # http://localhost:5000
```

For BERT, the Bi-LSTM, and the CLIP meme pipeline, install the extra dependencies first:

```bash
pip install -r requirements-ml.txt
# tesseract for OCR:  brew install tesseract (macOS)  /  apt-get install -y tesseract-ocr (Ubuntu)
python -m src.run_training
```

On macOS the AirPlay Receiver holds port 5000, so run `PORT=5001 python app.py` there.

## Models and results

The text task is three-class classification on a merged corpus of 44,931 posts
(55% offensive, 29% normal, 16% hate speech). The imbalance is handled with balanced class
weights, and every model uses the same stratified 80/10/10 train/validation/test split.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| BERT (bert-base-uncased, weighted loss) | 0.80 | 0.77 | 0.80 |
| Bi-LSTM (trainable embeddings, early stopping) | 0.77 | 0.75 | 0.78 |
| SVM (TF-IDF word+char, LinearSVC) | 0.77 | 0.75 | 0.78 |
| CLIP + OCR (zero-shot, memes) | — | — | — |

Scores are on the held-out test split (seed 42). BERT is the strongest text model. CLIP
runs zero-shot for the binary meme task, so no supervised accuracy is reported. The web
app reads these figures from `models/metrics.json` at runtime, so the table always matches
what training actually produced.

The gender-bias audit swaps gendered terms (he/she, man/woman, and so on) in 3,000 real
posts and checks how often the prediction changes. For the SVM that is 4.8% overall
(6.9% normal, 2.9% offensive, 6.8% hate speech). Regenerate it with `python -m src.fairness`.

## How the explanations work

- **Text.** The SVM is linear over TF-IDF features, so a word's contribution is exactly
  `coefficient × tf-idf` (no sampling needed). BERT is non-linear, so words are scored by
  occlusion: mask one word at a time and measure the drop in the predicted class.
- **Memes.** pytesseract reads the caption, CLIP scores the image against a set of "hateful"
  and "non-hateful" prompt templates, and Grad-CAM highlights the regions behind the score.
- **Fairness.** Counterfactual gender-term substitution, reported as a label flip-rate.

## Project structure

```
SafeScanTool/
├── app.py                     # Flask app: UI + /analyze/text, /analyze/meme, /api/*
├── templates/index.html       # Single-page frontend (4 tabs, no build step)
├── src/
│   ├── data_utils.py          # Load, clean, 80/10/10 split, class weights
│   ├── eval_utils.py          # Metrics + the models/metrics.json registry
│   ├── train_svm.py           # SVM (TF-IDF word+char + LinearSVC)
│   ├── train_bilstm.py        # Bi-LSTM (GloVe optional, early stopping)
│   ├── train_bert.py          # BERT fine-tune (weighted loss, early stopping)
│   ├── run_training.py        # Train all three text models, then run the audit
│   ├── clip_meme_pipeline.py  # pytesseract OCR + CLIP zero-shot classification
│   ├── explainability.py      # SVM SHAP, BERT occlusion, CLIP Grad-CAM
│   ├── fairness.py            # Counterfactual gender-bias audit
│   └── config.py             # Environment-driven config (no hard-coded secrets)
├── scripts/upload_models.py   # Push trained models/ to a Hugging Face model repo
├── docs/
│   ├── TRAINING.md            # Train BERT / Bi-LSTM on a free GPU
│   └── DEPLOYMENT.md          # How the Hugging Face Space is built and updated
├── Data/updated_hatexplain_data.csv   # 44,931 labelled posts (3-class)
├── notebooks/                 # Original research notebooks
├── models/                    # Trained artifacts + metrics.json (git-ignored)
├── requirements.txt           # Core dependencies (pinned)
├── requirements-ml.txt        # Deep-learning / multimodal extras (pinned)
├── Dockerfile                 # Container build (pulls trained models from HF)
└── .env.example               # Configuration template
```

## Training and deployment

- Training BERT and the Bi-LSTM on a free GPU (Colab or Kaggle): [docs/TRAINING.md](docs/TRAINING.md)
- How the Hugging Face Space is built and how to update it: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Dataset

The merged text corpus ships with the repo (`Data/updated_hatexplain_data.csv`). The
Facebook Hateful Memes images require registration with Meta AI and are not included; see
[Data/README.md](Data/README.md).

## Tech stack

Python, scikit-learn, PyTorch, TensorFlow, Hugging Face Transformers, CLIP
(`openai/clip-vit-base-patch32`), Flask, and Docker. OCR runs locally through pytesseract,
so no API keys are needed.

## Security note

Earlier notebook versions contained hard-coded API keys. They have been removed and
scrubbed from the git history, and should be treated as compromised and rotated. Runtime
configuration goes through `.env` (see `.env.example`), which is git-ignored.

## Limitations

Automated hate-speech detection is imperfect and highly context-dependent. The models
inherit biases from their training data (measured by the fairness audit), can misread
reclaimed language, irony, and dialect, and are meant to support human moderation, not
replace it.

## Authors

| Name | Email |
|---|---|
| Nikhil Ayyappan Nair | nan16@student.le.ac.uk |
| Abhishek Kumar Pal | akp33@student.le.ac.uk |
| Sahil Dinesh Padwal | sdp20@student.le.ac.uk |

Supervised by Dr. Hammad Afzal, School of Computing and Mathematical Sciences,
University of Leicester.

## License

For academic use. Please cite if you build on this work.
