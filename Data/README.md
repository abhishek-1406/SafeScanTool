# Datasets

This project uses three publicly available datasets. Raw data files are **not included** in this repository. Follow the instructions below to download each one.

---

## 1. HateXplain

- **Type:** Text — 3-class (Normal / Offensive / Hate Speech)
- **Size:** ~10,000 annotated social media posts
- **Source:** [https://github.com/hate-alert/HateXplain](https://github.com/hate-alert/HateXplain)

**Download:**
```bash
git clone https://github.com/hate-alert/HateXplain.git
```
Use the file `Data/dataset.json` from the cloned repo.

---

## 2. Hate Speech & Offensive Language (Davidson et al.)

- **Type:** Text — 3-class (Normal / Offensive / Hate Speech)
- **Size:** ~25,000 tweets
- **Source:** [https://github.com/t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language)

**Download:**
```bash
git clone https://github.com/t-davidson/hate-speech-and-offensive-language.git
```
Use the file `data/labeled_data.csv` from the cloned repo.

---

## 3. Facebook Hateful Memes Dataset

- **Type:** Multimodal — Binary (Hateful / Non-Hateful) — images + text
- **Size:** 10,000+ memes
- **Source:** [https://ai.meta.com/tools/hatefulmemes/](https://ai.meta.com/tools/hatefulmemes/)

> ⚠️ **This dataset requires registration.** You must apply for access via Meta AI and agree to their terms of use. The dataset is provided strictly for non-commercial research purposes.

**Steps to access:**
1. Visit [https://ai.meta.com/tools/hatefulmemes/](https://ai.meta.com/tools/hatefulmemes/)
2. Fill in the access request form with your institutional/research details
3. Once approved, download the dataset and place it under `data/hateful_memes/`

---

## Combined Corpus (Text)

The two text datasets (HateXplain + Davidson) were merged and preprocessed to create a combined corpus of **44,931 instances** used for text model training. Preprocessing steps are documented in `notebooks/01_EDA.ipynb`.

---

## Expected Folder Structure

After downloading, organise files as follows:

```
data/
├── hatexplain/
│   └── dataset.json
├── davidson/
│   └── labeled_data.csv
├── hateful_memes/
│   ├── img/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── README.md               ← this file
```
