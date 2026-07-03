"""Zero-shot multimodal meme classifier: pytesseract OCR + CLIP prompt ensembles.

Rewrite of the original ``clip&ocr.ipynb`` with three concrete improvements the
brief asked for:

1. **pytesseract OCR** replaces the tangle of EasyOCR / Google-Vision / Clarifai
   calls (no external API keys, fully local, deterministic).
2. **Engineered zero-shot prompt ensembles** — instead of a single
   "hateful/non-hateful" pair, each class is described by *many* templates whose
   text embeddings are averaged. Prompt ensembling is the standard trick for
   lifting CLIP zero-shot F1, and it needs no trained checkpoint.
3. **Confidence scores are always returned and can be persisted** — batch mode
   writes a CSV/JSON with per-image label + probability.

Heavy imports (torch, transformers, PIL, pytesseract) are lazy so this module
can be imported and inspected without them installed.

Run (batch):  python -m src.clip_meme_pipeline --images path/to/img_dir --out models/meme_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# --------------------------------------------------------------------------- #
# Prompt ensembles. Each class is described by several natural-language
# templates; their CLIP text embeddings are averaged into one class prototype.
# Richer, varied descriptions are what push zero-shot F1 up over a single prompt.
# --------------------------------------------------------------------------- #
HATEFUL_PROMPTS: List[str] = [
    "a hateful meme",
    "a meme that attacks or dehumanises a group of people",
    "a racist meme",
    "a meme expressing hatred based on religion, race, or gender",
    "an offensive meme promoting discrimination or violence",
    "a meme mocking people because of their identity",
    "hate speech in the form of a meme",
]
BENIGN_PROMPTS: List[str] = [
    "a harmless meme",
    "a wholesome and funny meme",
    "a friendly joke meme",
    "a neutral internet meme",
    "a meme that does not target or insult anyone",
    "a lighthearted picture with a caption",
    "a normal, non-offensive meme",
]
# Caption-aware templates: ``{}`` is filled with the OCR-extracted meme text so
# the prompt reflects what the meme actually says.
CAPTION_HATEFUL_TEMPLATE = 'a meme whose caption "{}" is hateful or offensive'
CAPTION_BENIGN_TEMPLATE = 'a meme whose caption "{}" is harmless and inoffensive'

LABELS = ["non-hateful", "hateful"]  # index 0 / 1


@dataclass
class MemePrediction:
    image: str
    label: str
    confidence: float          # probability of the predicted class
    p_hateful: float
    ocr_text: str
    used_caption: bool         # whether OCR text contributed to the decision

    def as_dict(self) -> Dict:
        return asdict(self)


def extract_text(image_path: str | Path, lang: str = "eng") -> str:
    """Extract meme text with pytesseract, with light preprocessing for accuracy."""
    import cv2  # lazy
    import pytesseract

    img = cv2.imread(str(image_path))
    if img is None:  # pytesseract can still try via PIL as a fallback
        from PIL import Image
        return pytesseract.image_to_string(Image.open(image_path), lang=lang).strip()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Upscale small images and binarise — meme captions are usually high-contrast.
    if max(gray.shape) < 800:
        scale = 800 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, lang=lang)
    return " ".join(text.split()).strip()


class MemeClassifier:
    """CLIP zero-shot meme classifier with OCR-augmented prompt ensembling."""

    def __init__(self, device: Optional[str] = None, caption_weight: float = 0.4,
                 temperature: float = 100.0):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        # How much the caption-aware prompts count vs. the generic visual prompts.
        self.caption_weight = caption_weight
        self.temperature = temperature  # CLIP's logit scale (~100) sharpens softmax
        self._generic_protos = self._encode_class_prototypes(HATEFUL_PROMPTS, BENIGN_PROMPTS)

    # -- embedding helpers -------------------------------------------------- #
    def _encode_texts(self, prompts: List[str]):
        torch = self.torch
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _encode_class_prototypes(self, hateful: List[str], benign: List[str]):
        """Average prompt embeddings into one prototype per class → (2, d)."""
        torch = self.torch
        h = self._encode_texts(hateful).mean(dim=0, keepdim=True)
        b = self._encode_texts(benign).mean(dim=0, keepdim=True)
        protos = torch.cat([b, h], dim=0)  # row 0 = non-hateful, row 1 = hateful
        return protos / protos.norm(dim=-1, keepdim=True)

    def _encode_image(self, image_path: str | Path):
        from PIL import Image
        torch = self.torch
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _softmax_scores(self, image_emb, protos):
        """Temperature-scaled softmax over the 2 class prototypes."""
        logits = self.temperature * (image_emb @ protos.T)  # (1, 2)
        return logits.softmax(dim=-1)[0]

    # -- public API --------------------------------------------------------- #
    def predict(self, image_path: str | Path, ocr_text: Optional[str] = None) -> MemePrediction:
        torch = self.torch
        if ocr_text is None:
            ocr_text = extract_text(image_path)

        image_emb = self._encode_image(image_path)
        generic = self._softmax_scores(image_emb, self._generic_protos)

        used_caption = bool(ocr_text and len(ocr_text) > 3)
        if used_caption:
            caption_protos = self._encode_class_prototypes(
                [CAPTION_HATEFUL_TEMPLATE.format(ocr_text)],
                [CAPTION_BENIGN_TEMPLATE.format(ocr_text)],
            )
            caption = self._softmax_scores(image_emb, caption_protos)
            probs = (1 - self.caption_weight) * generic + self.caption_weight * caption
        else:
            probs = generic

        probs = probs / probs.sum()
        idx = int(torch.argmax(probs).item())
        return MemePrediction(
            image=str(image_path),
            label=LABELS[idx],
            confidence=round(float(probs[idx].item()), 4),
            p_hateful=round(float(probs[1].item()), 4),
            ocr_text=ocr_text,
            used_caption=used_caption,
        )


def run_batch(image_dir: str | Path, out_path: str | Path,
              exts=(".png", ".jpg", ".jpeg", ".webp")) -> List[MemePrediction]:
    """Classify every image in a directory and persist predictions + confidence."""
    image_dir, out_path = Path(image_dir), Path(out_path)
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images ({', '.join(exts)}) under {image_dir}")

    clf = MemeClassifier()
    preds: List[MemePrediction] = []
    for i, path in enumerate(images, 1):
        try:
            pred = clf.predict(path)
        except Exception as e:  # noqa: BLE001 — one bad image shouldn't stop the batch
            print(f"[CLIP] {path.name}: ERROR {e}")
            continue
        preds.append(pred)
        print(f"[CLIP] ({i}/{len(images)}) {path.name}: {pred.label} "
              f"(conf={pred.confidence}) ocr='{pred.ocr_text[:40]}'")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        out_path.write_text(json.dumps([p.as_dict() for p in preds], indent=2))
    else:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(preds[0].as_dict().keys()))
            writer.writeheader()
            writer.writerows(p.as_dict() for p in preds)
    print(f"[CLIP] Wrote {len(preds)} predictions -> {out_path}")
    return preds


# Cached singleton so the Flask app loads CLIP only once.
@lru_cache(maxsize=1)
def get_classifier() -> "MemeClassifier":
    return MemeClassifier()


def main():
    ap = argparse.ArgumentParser(description="Zero-shot CLIP meme classifier (pytesseract OCR)")
    ap.add_argument("--images", required=True, help="Directory of meme images")
    ap.add_argument("--out", default="models/meme_predictions.csv", help="CSV or JSON output path")
    args = ap.parse_args()
    run_batch(args.images, args.out)


if __name__ == "__main__":
    main()
