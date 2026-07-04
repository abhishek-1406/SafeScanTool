# Deployment

SafeScan runs as a **Docker** app on **Hugging Face Spaces** (free `cpu-basic`
tier). This doc explains how the pieces fit together and how to deploy your own.

## Three repositories, three jobs

| Where | Holds | Role |
|---|---|---|
| **GitHub** `abhishek-1406/SafeScanTool` | source code | canonical repo you develop in |
| **HF Space** `casanovaabhishek14/SafeScan` | a mirror of the code + `Dockerfile` | builds & serves the live app |
| **HF model repo** `casanovaabhishek14/safescan-models` | trained weights + `metrics.json` | the Space downloads these at build |

Weights are **not** committed to git (they're large and `.gitignore`d). Instead the
Space pulls them from the model repo — so code and model artifacts version separately.

## Build flow (the `Dockerfile`)

1. Install core deps + CPU-only PyTorch, Transformers, pytesseract/OpenCV.
2. Pre-download the CLIP weights (so the first meme request is instant).
3. `snapshot_download` the trained models (SVM + BERT + `metrics.json`) from
   `$MODELS_REPO_ID` into `models/`.
4. Run the fairness audit (`python -m src.fairness`) from the downloaded SVM.
5. `CMD python app.py` → Flask serves on port 5000 (declared via `app_port` in the
   README front-matter).

If the model download fails, the build falls back to training the SVM in-container
(no BERT/Bi-LSTM then) so it never hard-breaks.

## Deploy your own

1. **Create a Docker Space** at [huggingface.co/new-space](https://huggingface.co/new-space)
   (SDK: Docker, blank template, Public).
2. **Push the code** to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/<space-name>
   git push space main --force        # paste an HF write token as the git password
   ```
3. If you use your **own model repo**, set it on the Space:
   `Settings → Variables → MODELS_REPO_ID = <you>/<your-models-repo>` (the Dockerfile
   reads this as a build arg), then Factory rebuild.

The README's YAML front-matter (`sdk: docker`, `app_port: 5000`, …) configures the
Space — keep it at the top of `README.md`.

## Update the models on a running Space

After retraining (see [TRAINING.md](TRAINING.md)) and re-running
`scripts/upload_models.py`, the Space's cached download layer must be refreshed:

```bash
curl -X POST -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/<you>/<space-name>/restart?factory=true"
```

Verify with `curl https://<you>-<space-name>.hf.space/api/metrics`.

## Notes

- **macOS local runs:** port 5000 is used by the AirPlay Receiver — run
  `PORT=5001 python app.py` (or disable AirPlay Receiver).
- **Secrets:** nothing secret is required at runtime. The optional legacy OCR keys
  live in `.env` (see `.env.example`), never committed.
- **Free-tier resources:** 2 vCPU / 16 GB RAM — comfortable for SVM + BERT + CLIP
  inference on CPU (BERT ~0.5–1 s/request, CLIP ~1–2 s/image).
