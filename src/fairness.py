"""Counterfactual gender-bias audit for the text classifier.

Method (counterfactual token substitution): take real posts that mention a
gendered term, swap that term for its opposite-gender counterpart (he↔she,
man↔woman, …), and re-classify. A *fair* model should almost never change its
prediction just because a gendered word was swapped. The "flip rate" — the
fraction of pairs whose predicted label changes — is our bias metric.

Run:  python -m src.fairness            # audits the trained SVM, writes JSON
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from .data_utils import ID2LABEL, basic_clean, load_dataframe
from .eval_utils import MODELS_DIR

# Bidirectional gender term pairs used for the counterfactual swap.
GENDER_PAIRS: List[tuple[str, str]] = [
    ("he", "she"), ("him", "her"), ("his", "her"), ("himself", "herself"),
    ("man", "woman"), ("men", "women"), ("male", "female"), ("boy", "girl"),
    ("boys", "girls"), ("father", "mother"), ("brother", "sister"),
    ("son", "daughter"), ("husband", "wife"), ("king", "queen"),
    ("gentleman", "lady"), ("guy", "gal"), ("mr", "mrs"),
]
FAIRNESS_PATH = MODELS_DIR / "fairness.json"


def _build_swap_map() -> Dict[str, str]:
    swap = {}
    for a, b in GENDER_PAIRS:
        swap[a] = b
        swap.setdefault(b, a)  # don't overwrite (his->her already set)
    return swap


def _swap_gender(text: str, swap: Dict[str, str]) -> tuple[str, bool]:
    changed = False
    out = []
    for tok in text.split():
        low = tok.lower()
        if low in swap:
            out.append(swap[low])
            changed = True
        else:
            out.append(tok)
    return " ".join(out), changed


def _predict(model, vectorizer, texts: List[str]) -> np.ndarray:
    X = vectorizer.transform([basic_clean(t) for t in texts])
    return np.asarray(model.predict(X))


def run_audit(model_path: Path | None = None, vectorizer_path: Path | None = None,
              max_samples: int = 3000, seed: int = 42) -> Dict:
    model_path = model_path or MODELS_DIR / "svm_model.pkl"
    vectorizer_path = vectorizer_path or MODELS_DIR / "svm_vectorizer.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — train the SVM first (python -m src.train_svm)."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    swap = _build_swap_map()

    # Keep only posts that actually contain a gendered term.
    df = load_dataframe()
    gender_re = re.compile(r"\b(?:" + "|".join(swap.keys()) + r")\b")
    mask = df["text"].str.contains(gender_re)
    subset = df[mask]
    if len(subset) > max_samples:
        subset = subset.sample(max_samples, random_state=seed)
    originals = subset["text"].tolist()

    counterfactuals, kept = [], []
    for t in originals:
        cf, changed = _swap_gender(t, swap)
        if changed:
            counterfactuals.append(cf)
            kept.append(t)

    orig_pred = _predict(model, vectorizer, kept)
    cf_pred = _predict(model, vectorizer, counterfactuals)
    flipped = orig_pred != cf_pred

    # Per-original-class flip rate.
    per_class = {}
    for cid, name in ID2LABEL.items():
        cls_mask = orig_pred == cid
        n = int(cls_mask.sum())
        per_class[name] = {
            "n": n,
            "flips": int(flipped[cls_mask].sum()),
            "flip_rate": round(float(flipped[cls_mask].mean()) if n else 0.0, 4),
        }

    # A few concrete examples of predictions that flipped after a gender swap.
    examples = []
    for i in np.nonzero(flipped)[0][:8]:
        examples.append({
            "original": kept[i],
            "counterfactual": counterfactuals[i],
            "original_label": ID2LABEL[int(orig_pred[i])],
            "counterfactual_label": ID2LABEL[int(cf_pred[i])],
        })

    result = {
        "method": "counterfactual gender-term substitution",
        "n_pairs": len(kept),
        "overall_flip_rate": round(float(flipped.mean()) if len(kept) else 0.0, 4),
        "per_class": per_class,
        "examples": examples,
        "gender_pairs": [list(p) for p in GENDER_PAIRS],
    }

    FAIRNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAIRNESS_PATH.write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description="Gender-bias counterfactual audit (SVM)")
    ap.add_argument("--max-samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    res = run_audit(max_samples=args.max_samples, seed=args.seed)
    print(f"Audited {res['n_pairs']} counterfactual pairs")
    print(f"Overall flip rate: {res['overall_flip_rate']:.2%}")
    for name, s in res["per_class"].items():
        print(f"  {name:11s} n={s['n']:5d}  flip_rate={s['flip_rate']:.2%}")
    print(f"Written -> {FAIRNESS_PATH}")


if __name__ == "__main__":
    main()
