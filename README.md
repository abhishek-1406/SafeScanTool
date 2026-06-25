# 🛡️ SafeScan — Explainable Multimodal Hate Speech Detection

> MSc Data Science Group Research Project · University of Leicester · August 2025  
> Supervisor: Dr. Hammad Afzal · Module: MA7443

---

## 📌 Overview

SafeScan is an end-to-end hate speech detection system capable of classifying both **text** (tweets and social media posts) and **multimodal content** (memes). The system goes beyond accuracy — it integrates **explainability** (SHAP & Grad-CAM) and a **fairness audit** to address the ethical dimension of automated content moderation.

The project was built as part of the MSc Data Science Research Project at the University of Leicester and benchmarks four models across two modalities, culminating in an interactive web application.

---

## 🔍 Problem Statement

Manual content moderation at scale is unsustainable. Hate speech on social media contributes to psychological harm, civil discourse erosion, and real-world violence. Memes pose a unique challenge — combining text and images in ways that are individually innocuous but collectively harmful. This project addresses that gap with a multi-model, multi-modal approach.

---

## 🧠 Models Implemented

### Text-Based Classification (3-class: Normal / Offensive / Hate Speech)

| Model | Accuracy | Macro F1 |
|---|---|---|
| SVM (TF-IDF + Linear SVC) | 0.75 | 0.71 |
| Bi-LSTM (GloVe embeddings) | 0.75 | 0.72 |
| **BERT (fine-tuned bert-base-uncased)** | **0.81** | **0.81** |

### Multimodal Classification (Binary: Hateful / Non-Hateful)

| Model | Accuracy | F1 Score |
|---|---|---|
| CLIP + OCR (Google Vision API) | 0.73 | 0.57 |

---

## 📁 Repository Structure

```
safescan/
│
├── data/
│   └── README.md                  # Dataset sources and access instructions
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_SVM.ipynb               # SVM baseline + fairness audit
│   ├── 03_BiLSTM.ipynb            # Bi-LSTM with GloVe embeddings
│   ├── 04_BERT.ipynb              # BERT fine-tuning + occlusion explainability
│   └── 05_CLIP.ipynb              # CLIP multimodal model + Grad-CAM
│
├── backend/
│   ├── app.py                     # Flask API (model serving + CORS)
│   ├── models/                    # Serialised model files (.pkl, .h5, .pt)
│   └── utils/
│       ├── preprocessing.py       # Text cleaning, tokenisation, OCR
│       ├── explainability.py      # SHAP and Grad-CAM logic
│       └── fairness.py            # Counterfactual testing + bias metrics
│
├── frontend/
│   ├── src/
│   │   ├── components/            # React components (TextInput, ImageUpload, Results)
│   │   └── pages/                 # Next.js pages
│   ├── package.json
│   └── tsconfig.json
│
├── report/
│   └── HateSpeech_Detection_Final_Report.pdf   # Full academic report
│
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

| Dataset | Type | Size | Source |
|---|---|---|---|
| HateXplain | Text (3-class) | ~10,000 posts | [GitHub](https://github.com/hate-alert/HateXplain) |
| Hate Speech & Offensive Tweets | Text (3-class) | ~25,000 tweets | [GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language) |
| Merged Corpus (combined) | Text | 44,931 instances | Derived |
| Facebook Hateful Memes | Image + Text | 10,000+ memes | [Meta AI](https://ai.meta.com/tools/hatefulmemes/) |

> ⚠️ **Note:** The Facebook Hateful Memes dataset requires registration via Meta AI. Raw data files are not included in this repository. See `data/README.md` for access and setup instructions.

---

## ⚙️ Tech Stack

**ML / NLP**
- Python, scikit-learn, TensorFlow, PyTorch
- HuggingFace Transformers (`bert-base-uncased`)
- OpenAI CLIP (`openai/clip-vit-base-patch32`)
- GloVe Embeddings (Stanford NLP)

**Explainability & Fairness**
- SHAP (SHapley Additive Explanations)
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Counterfactual pronoun-swap testing
- SMOTE for class imbalance mitigation

**OCR & APIs**
- Google Cloud Vision API (primary OCR)
- EasyOCR + OpenCV (fallback)
- Clarifai API (hate symbol detection)

**Deployment**
- Flask (REST API backend)
- React + Next.js + TypeScript (frontend)
- Tailwind CSS, Plotly.js, Axios

---

## 🔬 Key Findings

- **BERT significantly outperforms** both traditional and recurrent models (Macro F1: 0.81 vs 0.71/0.72), confirming the superiority of contextual transformer representations for nuanced hate speech detection.
- **SVM is a strong baseline** — competitive with Bi-LSTM despite being far simpler and more computationally efficient.
- **Bi-LSTM suffered from overfitting** — training accuracy exceeded 90% while validation plateaued at ~75%, illustrating the performance-complexity paradox.
- **CLIP proves multimodal reasoning is essential** — 73% accuracy on the Hateful Memes benchmark (which is specifically designed to defeat unimodal systems).
- **All text models exhibit measurable gender bias** via counterfactual testing (pronoun swapping). Applying SMOTE-based rebalancing partially mitigated this, with a minor accuracy trade-off (F1: 0.72 → 0.71).
- **SHAP and Grad-CAM** provide actionable transparency — revealing which tokens or image regions drive predictions, and exposing spurious correlations.

---

## 🚀 Getting Started

### Prerequisites
```
Python 3.9+
Node.js 18+
```

### Backend Setup
```bash
git clone https://github.com/your-username/safescan.git
cd safescan
pip install -r requirements.txt
cd backend
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The app will run at `http://localhost:3000`, connecting to the Flask API at `http://localhost:5000`.

> **Note:** Large model files (BERT, CLIP) are not included in this repo due to size. Download instructions are in `backend/models/README.md`.

---

## 📖 Report

The full academic report covering methodology, results, fairness analysis, and deployment is available in the `/report` folder.

---

## 👥 Authors

| Name | Email |
|---|---|
| Nikhil Ayyappan Nair | nan16@student.le.ac.uk |
| Abhishek Kumar Pal | akp33@student.le.ac.uk |
| Sahil Dinesh Padwal | sdp20@student.le.ac.uk |

**Supervisor:** Dr. Hammad Afzal  
**School of Computing and Mathematical Sciences, University of Leicester**

---

## 📄 License

This project is for academic purposes. Please cite appropriately if building upon this work.
