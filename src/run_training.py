"""Train all three text models sequentially and print a comparison table.

Each model is trained in order (SVM -> Bi-LSTM -> BERT). Heavy models whose
dependencies (TensorFlow / PyTorch+Transformers) aren't installed are skipped
with a clear message rather than crashing the whole run — so you can train just
the SVM on a laptop, or everything on a GPU box.

Run:  python -m src.run_training                 # train everything available
      python -m src.run_training --only svm      # just one
      python -m src.run_training --skip bert      # everything except BERT
      python -m src.run_training --bert-smoke     # tiny BERT run to validate wiring
"""
from __future__ import annotations

import argparse
import time
import traceback

from .eval_utils import REGISTRY_PATH

MODELS = ("svm", "bilstm", "bert")


def _run_svm(args):
    from .train_svm import train
    return train(args.data, seed=args.seed)


def _run_bilstm(args):
    from .train_bilstm import train
    return train(args.data, glove_path=args.glove, epochs=args.epochs, seed=args.seed)


def _run_bert(args):
    from .train_bert import train
    kw = dict(data_path=args.data, seed=args.seed)
    if args.bert_smoke:
        kw.update(max_samples=1500, epochs=1)
    return train(**kw)


RUNNERS = {"svm": _run_svm, "bilstm": _run_bilstm, "bert": _run_bert}


def main():
    ap = argparse.ArgumentParser(description="Train all SafeScan text models sequentially")
    ap.add_argument("--data", default=None)
    ap.add_argument("--only", choices=MODELS, help="Train only this model")
    ap.add_argument("--skip", nargs="*", choices=MODELS, default=[], help="Models to skip")
    ap.add_argument("--glove", default=None, help="GloVe path for the Bi-LSTM")
    ap.add_argument("--epochs", type=int, default=40, help="Max epochs for the Bi-LSTM")
    ap.add_argument("--bert-smoke", action="store_true", help="Tiny BERT run (wiring check)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    todo = [args.only] if args.only else [m for m in MODELS if m not in args.skip]

    results, skipped = {}, {}
    t0 = time.time()
    for name in todo:
        print("\n" + "=" * 70 + f"\n  TRAINING: {name.upper()}\n" + "=" * 70)
        try:
            results[name] = RUNNERS[name](args)
        except ImportError as e:
            skipped[name] = f"missing dependency ({e.name})"
            print(f"[{name}] SKIPPED — {skipped[name]}. Install extras to enable it.")
        except Exception as e:  # noqa: BLE001 — keep going so other models still train
            skipped[name] = f"error: {e}"
            print(f"[{name}] FAILED — {e}\n{traceback.format_exc()}")

    # --- summary ----------------------------------------------------------- #
    print("\n" + "=" * 70 + "\n  SUMMARY\n" + "=" * 70)
    header = f"{'model':<28}{'accuracy':>10}{'macro_f1':>10}{'weighted_f1':>13}"
    print(header + "\n" + "-" * len(header))
    for name in todo:
        if name in results:
            m = results[name]
            print(f"{name:<28}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['weighted_f1']:>13.4f}")
        else:
            print(f"{name:<28}{'— skipped: ' + skipped.get(name, ''):>33}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    if results:
        print(f"Metrics written to: {REGISTRY_PATH}")

    # Generate the fairness audit too (it uses the SVM), so a single run_training
    # call leaves every web-app tab populated. Best-effort — needs a trained SVM.
    try:
        from .fairness import run_audit
        res = run_audit()
        print(f"Fairness audit: overall flip-rate {res['overall_flip_rate']:.2%}")
    except FileNotFoundError:
        print("Fairness audit skipped (no SVM model). Run after training the SVM.")
    except Exception as e:  # noqa: BLE001
        print(f"Fairness audit skipped: {e}")


if __name__ == "__main__":
    main()
