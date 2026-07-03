"""Shared data utilities for the SafeScan text pipelines.

Every text model (SVM, Bi-LSTM, BERT) loads its data through this module so the
label mapping, text cleaning, and the 80/10/10 train/val/test split are
*identical* across models. That is what makes the reported metrics comparable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------------------------------- #
# Label space (3-class text classification)
# --------------------------------------------------------------------------- #
LABEL2ID: Dict[str, int] = {"normal": 0, "offensive": 1, "hatespeech": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(len(ID2LABEL))]  # ["normal", ...]

# Repo-root-relative default location of the dataset.
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "updated_hatexplain_data.csv"

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WS_RE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    """Light, dependency-free normalisation used at both train and serve time.

    The bundled dataset is already lemmatised/stop-worded, so this stays
    deliberately conservative: lowercase, strip URLs/@mentions, keep letters and
    spaces, collapse whitespace. Using the *same* function for training and for
    the live API is what keeps SHAP token attributions aligned with the model.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def load_dataframe(path: str | Path = DEFAULT_DATA_PATH, clean: bool = True) -> pd.DataFrame:
    """Load the corpus, map string labels to ids, and (optionally) clean text.

    Returns a frame with columns: ``text`` (cleaned), ``label`` (int 0-2).
    Rows with unknown labels or empty text are dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See Data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    if not {"majority_label", "cleaned_text"}.issubset(df.columns):
        raise ValueError(
            f"Expected columns 'majority_label' and 'cleaned_text', got {list(df.columns)}"
        )

    df = df.rename(columns={"cleaned_text": "text"})
    df["label"] = df["majority_label"].map(LABEL2ID)
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(int)

    if clean:
        df["text"] = df["text"].apply(basic_clean)

    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["text", "label"]]


def train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 split (defaults) with a fixed seed.

    Done in two stages so the class balance is preserved in *all three* splits.
    """
    assert 0 < val_size < 1 and 0 < test_size < 1 and (val_size + test_size) < 1

    # First carve off the test set...
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    # ...then split the remainder into train/val, re-scaling val_size accordingly.
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=train_val["label"]
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def class_weight_dict(y) -> Dict[int, float]:
    """Balanced class weights as an ``{class_id: weight}`` dict.

    Consumed directly by sklearn (``class_weight=``), Keras (``class_weight=``),
    and our weighted-loss BERT trainer.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def describe_splits(train, val, test) -> str:
    """Human-readable summary of split sizes and per-class counts (for logs)."""
    lines = []
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = split["label"].value_counts().sort_index()
        pretty = ", ".join(f"{ID2LABEL[i]}={counts.get(i, 0)}" for i in range(len(ID2LABEL)))
        lines.append(f"  {name:5s} n={len(split):6d}  [{pretty}]")
    return "\n".join(lines)
