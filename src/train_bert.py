"""Fine-tune BERT (bert-base-uncased) for 3-class hate-speech classification.

There was no BERT notebook in the original repo, so this trainer is written from
scratch to match the model the README benchmarks. It uses the HuggingFace
``Trainer`` with:
  * the same 80/10/10 split as the other models,
  * a **class-weighted cross-entropy loss** (subclassed Trainer) for imbalance,
  * ``EarlyStoppingCallback`` on validation macro-F1,
  * ``load_best_model_at_end`` so the saved checkpoint is the best epoch.

This is GPU-recommended. On CPU it will run but slowly; use ``--max-samples`` for
a quick smoke test.

Run:  python -m src.train_bert
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .data_utils import (
    CLASS_NAMES,
    ID2LABEL,
    LABEL2ID,
    class_weight_dict,
    describe_splits,
    load_dataframe,
    train_val_test_split,
)
from .eval_utils import MODELS_DIR, compute_metrics, pretty_report, update_registry

MODEL_DIR = MODELS_DIR / "bert"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128


def train(data_path=None, epochs: int = 4, batch_size: int = 16,
          lr: float = 2e-5, max_samples: int | None = None, seed: int = 42) -> dict:
    import torch
    from torch import nn
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    t0 = time.time()
    df = load_dataframe(data_path) if data_path else load_dataframe()
    if max_samples:  # quick smoke test on a stratified subset
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max_samples // len(CLASS_NAMES)), random_state=seed)
        )
    train_df, val_df, test_df = train_val_test_split(df, seed=seed)
    print("[BERT] Split sizes:\n" + describe_splits(train_df, val_df, test_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_ds(split):
        ds = Dataset.from_dict({"text": split["text"].tolist(), "labels": split["label"].tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN), batched=True)

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASS_NAMES),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    # --- class-weighted loss ------------------------------------------------ #
    cw = class_weight_dict(train_df["label"])
    weight_tensor = torch.tensor([cw[i] for i in range(len(CLASS_NAMES))], dtype=torch.float)
    print(f"[BERT] Class weights: {cw}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor.to(outputs.logits.device))
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        m = compute_metrics(labels, preds, CLASS_NAMES)
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=1,
        seed=seed,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    metrics = compute_metrics(test_df["label"], preds, CLASS_NAMES)
    print("[BERT] Test set report:\n" + pretty_report(test_df["label"], preds, CLASS_NAMES))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    update_registry(
        "BERT (bert-base-uncased)",
        metrics,
        extra={"train_seconds": round(time.time() - t0, 1), "artifact": "bert/"},
    )
    print(f"[BERT] Saved model -> {MODEL_DIR}")
    print(f"[BERT] Done in {time.time() - t0:.1f}s | test macro-F1={metrics['macro_f1']:.4f} "
          f"| accuracy={metrics['accuracy']:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Fine-tune SafeScan BERT hate-speech classifier")
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap total samples (stratified) for a quick smoke test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args.data, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, max_samples=args.max_samples, seed=args.seed)


if __name__ == "__main__":
    main()
