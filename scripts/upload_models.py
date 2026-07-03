"""Upload trained model artifacts to a Hugging Face model repo.

After training on a free GPU (see docs/TRAINING.md), this pushes the whole
``models/`` folder — BERT checkpoint, Bi-LSTM ``.keras``, SVM ``.pkl``s, and the
merged ``metrics.json`` — to a public HF model repo. The Space then downloads
these at build time (much cheaper than retraining in the container).

Usage:
    export HF_TOKEN=hf_xxx            # a WRITE token
    python scripts/upload_models.py --repo-id <your-hf-username>/safescan-models
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Upload models/ to a Hugging Face model repo")
    ap.add_argument("--repo-id", required=True,
                    help="Target HF model repo, e.g. casanovaabhishek14/safescan-models")
    ap.add_argument("--models-dir", default="models", help="Local folder to upload")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF write token (defaults to $HF_TOKEN)")
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("No token. Set $HF_TOKEN or pass --token (needs WRITE scope).")
    models_dir = Path(args.models_dir)
    if not models_dir.exists() or not any(models_dir.iterdir()):
        raise SystemExit(f"{models_dir} is empty — train first (python -m src.run_training).")

    from huggingface_hub import HfApi, create_repo

    create_repo(args.repo_id, repo_type="model", exist_ok=True, token=args.token)
    HfApi(token=args.token).upload_folder(
        folder_path=str(models_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload SafeScan trained models",
        ignore_patterns=["*.tmp", "**/checkpoint-*/**"],  # skip HF Trainer intermediate checkpoints
    )
    print(f"✅ Uploaded {models_dir}/ -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
