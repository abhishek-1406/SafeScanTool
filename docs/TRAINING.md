# Training the heavy models for free (Kaggle GPU)

**CLIP needs no training** — it's zero-shot, already live on the Space. This guide
trains **BERT** and **Bi-LSTM** on a free GPU and uploads them (plus the merged
`metrics.json`) to a Hugging Face model repo the Space can pull from.

> Kaggle gives **~30 free GPU hours/week** with 12-hour sessions — the most
> generous free option. Google Colab works too but disconnects sooner.

---

## 1. Set up the Kaggle notebook
1. New notebook at [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. Right sidebar → **Settings → Accelerator → GPU T4 x2** (free).
3. **Add-ons → Secrets** → add a secret named **`HF_TOKEN`** = your HF **write** token.

## 2. Run these cells

**Cell 1 — clone + install**
```python
!git clone https://github.com/abhishek-1406/SafeScanTool.git
%cd SafeScanTool
# Colab/Kaggle already ship torch, tensorflow, scikit-learn, pandas — adding the
# pinned requirements-ml.txt fights their versions (ResolutionImpossible), so only
# install the few extras the training actually needs. tf-keras gives the Bi-LSTM
# the Keras-2 backend it needs (Colab's default Keras 3 removed the APIs it uses):
!pip install -q -U transformers datasets accelerate tf-keras
```
> On your **own machine** (not Colab), use the pinned set instead:
> `pip install -r requirements.txt -r requirements-ml.txt`.

**Cell 2 — train all text models** (SVM + Bi-LSTM + BERT)
```python
!python -m src.run_training
```
- BERT: ~20–40 min on a T4 · Bi-LSTM: a few min (early-stops) · SVM: seconds.
- This writes `models/bert/`, `models/bilstm_model.keras`, the SVM `.pkl`s, and a
  merged `models/metrics.json` containing **all** models' test scores.

**Cell 3 — upload the trained models to Hugging Face**
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!python scripts/upload_models.py --repo-id casanovaabhishek14/safescan-models
```
This creates (if needed) and fills `huggingface.co/casanovaabhishek14/safescan-models`.

## 3. Refresh the live Space
The deployed Space already pulls this model repo at build time (see [`Dockerfile`](../Dockerfile))
and exposes a SVM/BERT selector + a live comparison table. To make it pick up the
**newly uploaded** models, trigger a factory rebuild (the download is a cached layer):

- **UI:** Space → **Settings → Factory rebuild**, or
- **API:**
  ```bash
  curl -X POST -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/spaces/casanovaabhishek14/SafeScan/restart?factory=true"
  ```

After the rebuild, `/api/metrics` and the **Model comparison** tab show all trained models.
See [DEPLOYMENT.md](DEPLOYMENT.md) for the full deployment architecture.

---

### Notes
- **GloVe for Bi-LSTM (optional):** by default the Bi-LSTM uses a trainable embedding
  layer (no download). For pretrained GloVe, add a cell to fetch `glove.6B.100d.txt`
  and run `!python -m src.train_bilstm --glove glove.6B.100d.txt`.
- **CLIP meme model:** nothing to train — the live Space already does zero-shot CLIP + OCR.
- **Colab alternative:** same cells; enable GPU via *Runtime → Change runtime type → T4 GPU*,
  and set `os.environ["HF_TOKEN"]` manually instead of using Kaggle Secrets.
