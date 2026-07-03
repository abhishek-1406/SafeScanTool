"""Train the Bi-LSTM (GloVe embeddings) text classifier.

Improvements vs. the original notebook:
  * proper 80/10/10 split — the held-out ``val`` split drives early stopping and
    the ``test`` split is only ever touched once, for the final score,
  * ``EarlyStopping(restore_best_weights=True)`` + ``ModelCheckpoint`` directly
    address the overfitting the report flagged (40 epochs, no stopping),
  * GloVe is *optional*: if ``glove.6B.100d.txt`` is available (via ``--glove``
    or ``$GLOVE_PATH``) it is loaded and frozen; otherwise a trainable embedding
    layer is used so the script runs with zero external downloads.

Run:  python -m src.train_bilstm            # trainable embeddings
      python -m src.train_bilstm --glove /path/to/glove.6B.100d.txt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np

from .data_utils import (
    CLASS_NAMES,
    class_weight_dict,
    describe_splits,
    load_dataframe,
    train_val_test_split,
)
from .eval_utils import MODELS_DIR, compute_metrics, pretty_report, update_registry

MODEL_PATH = MODELS_DIR / "bilstm_model.keras"
TOKENIZER_PATH = MODELS_DIR / "bilstm_tokenizer.pkl"

MAX_VOCAB = 20000
MAX_LEN = 60
EMBED_DIM = 100


def _load_glove(path: Path, word_index: dict) -> np.ndarray:
    """Build an embedding matrix from a GloVe text file for the known vocab."""
    print(f"[BiLSTM] Loading GloVe vectors from {path} ...")
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            embeddings[parts[0]] = np.asarray(parts[1:], dtype="float32")
    num_words = min(MAX_VOCAB, len(word_index) + 1)
    matrix = np.zeros((num_words, EMBED_DIM), dtype="float32")
    hits = 0
    for word, i in word_index.items():
        if i < num_words and (vec := embeddings.get(word)) is not None:
            matrix[i] = vec
            hits += 1
    print(f"[BiLSTM] GloVe coverage: {hits}/{num_words} tokens")
    return matrix


def build_model(num_words: int, embedding_matrix: np.ndarray | None):
    """BiLSTM(64) -> BiLSTM(32) -> Dense(64) -> softmax(3)."""
    from tensorflow.keras.layers import (
        Bidirectional, Dense, Dropout, Embedding, LSTM,
    )
    from tensorflow.keras.initializers import Constant
    from tensorflow.keras.models import Sequential

    # NB: Keras 3 (TF 2.16+) dropped Embedding's ``input_length`` arg and the
    # ``weights=[...]`` constructor kwarg — use an initializer instead so this
    # runs on both Keras 2 and Keras 3 (e.g. Colab).
    if embedding_matrix is not None:
        embed = Embedding(num_words, EMBED_DIM,
                          embeddings_initializer=Constant(embedding_matrix),
                          trainable=False)
    else:
        embed = Embedding(num_words, EMBED_DIM, trainable=True)

    model = Sequential([
        embed,
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train(data_path=None, glove_path: str | None = None, epochs: int = 40,
          batch_size: int = 64, seed: int = 42) -> dict:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical

    tf.random.set_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    df = load_dataframe(data_path) if data_path else load_dataframe()
    train_df, val_df, test_df = train_val_test_split(df, seed=seed)
    print("[BiLSTM] Split sizes:\n" + describe_splits(train_df, val_df, test_df))

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text"])  # fit ONLY on train to avoid leakage

    def encode(split):
        seqs = tokenizer.texts_to_sequences(split["text"])
        return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, X_val, X_test = encode(train_df), encode(val_df), encode(test_df)
    y_train = to_categorical(train_df["label"], num_classes=len(CLASS_NAMES))
    y_val = to_categorical(val_df["label"], num_classes=len(CLASS_NAMES))

    glove_path = glove_path or os.environ.get("GLOVE_PATH")
    embedding_matrix = None
    num_words = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    if glove_path and Path(glove_path).exists():
        embedding_matrix = _load_glove(Path(glove_path), tokenizer.word_index)
        num_words = embedding_matrix.shape[0]
    else:
        print("[BiLSTM] No GloVe file found — using a trainable embedding layer.")

    model = build_model(num_words, embedding_matrix)
    model.summary(print_fn=lambda s: print("[BiLSTM] " + s))

    cw = class_weight_dict(train_df["label"])
    print(f"[BiLSTM] Class weights: {cw}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=cw, callbacks=callbacks, verbose=2,
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    metrics = compute_metrics(test_df["label"], y_pred, CLASS_NAMES)
    print("[BiLSTM] Test set report:\n" + pretty_report(test_df["label"], y_pred, CLASS_NAMES))

    model.save(MODEL_PATH)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    update_registry(
        "Bi-LSTM (GloVe)",
        metrics,
        extra={"used_glove": embedding_matrix is not None,
               "train_seconds": round(time.time() - t0, 1),
               "artifact": MODEL_PATH.name},
    )
    print(f"[BiLSTM] Saved model -> {MODEL_PATH}")
    print(f"[BiLSTM] Done in {time.time() - t0:.1f}s | test macro-F1={metrics['macro_f1']:.4f} "
          f"| accuracy={metrics['accuracy']:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train SafeScan Bi-LSTM hate-speech classifier")
    ap.add_argument("--data", default=None)
    ap.add_argument("--glove", default=None, help="Path to glove.6B.100d.txt (optional)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args.data, glove_path=args.glove, epochs=args.epochs,
          batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
