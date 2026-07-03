"""Explainability: word-level attributions for text, Grad-CAM for memes.

Text (SVM): the classifier is *linear* over TF-IDF features, so a token's
contribution to a class is exactly ``coef[class, token] * tfidf(token)`` — this
is the SHAP value of a linear model under an all-zeros baseline. It is exact,
instant, and needs no sampling (unlike KernelExplainer in the old notebook).

Image (CLIP): Grad-CAM over the vision transformer's last hidden state, w.r.t.
the similarity between the image embedding and the "hateful" text prototype.
"""
from __future__ import annotations

import base64
import io
from typing import Dict, List

import numpy as np

from .data_utils import ID2LABEL, basic_clean


# --------------------------------------------------------------------------- #
# TEXT — linear SHAP-equivalent word attributions
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def explain_text(model, vectorizer, text: str, top_k: int = 15) -> Dict:
    """Return prediction, class probabilities, and signed per-word attributions.

    ``vectorizer`` is the FeatureUnion (word + char). We attribute only over the
    *word* sub-vectorizer so the highlights map back to readable tokens.
    """
    cleaned = basic_clean(text)
    X = vectorizer.transform([cleaned])
    decision = np.asarray(model.decision_function(X)).ravel()
    # LinearSVC is one-vs-rest; decision has one score per class.
    probs = _softmax(decision)
    pred_id = int(np.argmax(decision))

    # Locate the word sub-vectorizer inside the FeatureUnion and its column slice.
    word_vec = dict(vectorizer.transformer_list)["word"]
    word_names = word_vec.get_feature_names_out()
    n_word = len(word_names)  # word features occupy the first n_word columns

    coef = np.asarray(model.coef_)              # (n_classes, n_features)
    x_row = X.toarray().ravel()                 # dense feature values
    contrib = coef[pred_id, :n_word] * x_row[:n_word]  # exact linear attribution

    nz = np.nonzero(contrib)[0]
    pairs = sorted(((word_names[i], float(contrib[i])) for i in nz),
                   key=lambda t: abs(t[1]), reverse=True)[:top_k]

    return {
        "label": ID2LABEL[pred_id],
        "label_id": pred_id,
        "cleaned_text": cleaned,
        "probabilities": {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "confidence": round(float(probs[pred_id]), 4),
        # positive weight => pushes toward the predicted class; negative => away.
        "words": [{"word": w, "weight": round(v, 4)} for w, v in pairs],
    }


# --------------------------------------------------------------------------- #
# IMAGE — Grad-CAM over the CLIP vision transformer
# --------------------------------------------------------------------------- #
def gradcam_clip(classifier, image_path: str, alpha: float = 0.5) -> Dict:
    """Grad-CAM heatmap for a meme, overlaid on the original image.

    Returns the prediction plus a base64-encoded PNG of the overlay so the Flask
    route can hand it straight to the browser.
    """
    import cv2
    import torch
    from PIL import Image

    model = classifier.model
    processor = classifier.processor
    device = classifier.device

    # Classify first — this is reliable. Grad-CAM is layered on best-effort, so a
    # hiccup in the heatmap never costs us the prediction (it just omits the overlay).
    pred = classifier.predict(image_path)
    result = {
        "label": pred.label,
        "confidence": pred.confidence,
        "p_hateful": pred.p_hateful,
        "ocr_text": pred.ocr_text,
        "overlay_png_base64": "",
    }

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Hook the last vision encoder layer to capture activations + gradients.
        target_layer = model.vision_model.encoder.layers[-1]
        activations, gradients = {}, {}

        def fwd_hook(_m, _i, out):
            activations["value"] = out[0] if isinstance(out, tuple) else out

        def bwd_hook(_m, _gi, go):
            gradients["value"] = go[0]

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        try:
            model.zero_grad()
            img_emb = model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            # Similarity to the "hateful" prototype is the quantity we explain.
            hateful_proto = classifier._generic_protos[1:2]  # (1, d)
            score = (img_emb @ hateful_proto.T).squeeze()
            score.backward()

            acts = activations["value"].detach()   # (1, seq, dim)
            grads = gradients["value"].detach()     # (1, seq, dim)
            # Drop the CLS token (index 0); the rest are the patch tokens.
            acts, grads = acts[:, 1:, :], grads[:, 1:, :]
            weights = grads.mean(dim=1, keepdim=True)  # channel importance
            cam = torch.relu((weights * acts).sum(dim=-1)).squeeze(0)  # (n_patches,)
        finally:
            h1.remove()
            h2.remove()

        side = int(round(cam.shape[0] ** 0.5))
        cam = cam[: side * side].reshape(side, side).cpu().numpy()
        cam = (cam - cam.min()) / (float(np.ptp(cam)) + 1e-8)  # np.ptp: numpy 2.0 dropped ndarray.ptp()

        # Upscale to the original image size and build a JET overlay.
        orig = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        heat = cv2.resize((cam * 255).astype(np.uint8), (orig.shape[1], orig.shape[0]))
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heat, alpha, orig, 1 - alpha, 0)
        ok, buf = cv2.imencode(".png", overlay)
        if ok:
            result["overlay_png_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:  # noqa: BLE001 — never let Grad-CAM sink the classification
        result["gradcam_error"] = str(e)

    return result
