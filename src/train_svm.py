"""Train the SVM (TF-IDF + LinearSVC) text classifier.

Design choices vs. the original notebook:
  * proper 80/10/10 train/val/test split (was 80/20),
  * class-weight balancing instead of SMOTE (no synthetic-sample leakage),
  * the regularisation strength ``C`` is tuned on the *validation* split and the
    final score is reported on the untouched *test* split,
  * a single ``FeatureUnion`` (word + char n-grams) is pickled so the serving
    code transforms text with one ``.transform()`` call.

Run:  python -m src.train_svm
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .data_utils import (
    CLASS_NAMES,
    class_weight_dict,
    describe_splits,
    load_dataframe,
    train_val_test_split,
)
from .eval_utils import MODELS_DIR, compute_metrics, pretty_report, update_registry

MODEL_PATH = MODELS_DIR / "svm_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "svm_vectorizer.pkl"

# ``C`` values tuned against the validation split (macro-F1).
C_GRID = (0.5, 1.0, 2.0, 5.0)


def build_vectorizer() -> FeatureUnion:
    """Word (1-2gram) + character (3-5gram) TF-IDF, combined into one object."""
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)
    return FeatureUnion([("word", word), ("char", char)])


def train(data_path: str | Path | None = None, seed: int = 42) -> dict:
    t0 = time.time()
    df = load_dataframe(data_path) if data_path else load_dataframe()
    train_df, val_df, test_df = train_val_test_split(df, seed=seed)
    print("[SVM] Split sizes:\n" + describe_splits(train_df, val_df, test_df))

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    cw = class_weight_dict(train_df["label"])
    print(f"[SVM] Class weights: {cw}")

    # --- tune C on the validation split ------------------------------------ #
    best_c, best_f1, best_model = None, -1.0, None
    for c in C_GRID:
        clf = LinearSVC(C=c, class_weight=cw, max_iter=20000)
        clf.fit(X_train, train_df["label"])
        val_metrics = compute_metrics(val_df["label"], clf.predict(X_val), CLASS_NAMES)
        print(f"[SVM]   C={c:<4}  val macro-F1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_c, best_f1, best_model = c, val_metrics["macro_f1"], clf
    print(f"[SVM] Best C={best_c} (val macro-F1={best_f1:.4f})")

    # --- final evaluation on the untouched test split ---------------------- #
    y_pred = best_model.predict(X_test)
    metrics = compute_metrics(test_df["label"], y_pred, CLASS_NAMES)
    print("[SVM] Test set report:\n" + pretty_report(test_df["label"], y_pred, CLASS_NAMES))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    update_registry(
        "SVM (TF-IDF + LinearSVC)",
        metrics,
        extra={"best_C": best_c, "train_seconds": round(time.time() - t0, 1),
               "artifact": MODEL_PATH.name},
    )
    print(f"[SVM] Saved model -> {MODEL_PATH}")
    print(f"[SVM] Done in {time.time() - t0:.1f}s | test macro-F1={metrics['macro_f1']:.4f} "
          f"| accuracy={metrics['accuracy']:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train SafeScan SVM hate-speech classifier")
    ap.add_argument("--data", default=None, help="Path to dataset CSV (defaults to Data/…)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args.data, seed=args.seed)


if __name__ == "__main__":
    main()
