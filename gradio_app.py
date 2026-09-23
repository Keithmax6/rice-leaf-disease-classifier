"""Rice leaf checker — Gradio UI for local run or Hugging Face Spaces."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
WEIGHTS = next(
    (p for p in (ROOT / "rice_leaf_disease.pt", ROOT.parent / "rice_leaf_disease.pt") if p.exists()),
    ROOT / "rice_leaf_disease.pt",
)

DISPLAY = {
    "Healthy": "Healthy",
    "Bacterialblight": "Bacterial leaf blight",
    "Blast": "Rice blast",
    "Brownspot": "Brown spot",
    "Tungro": "Rice tungro",
}

DEVICE = torch.device("cpu")
CKPT = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
CLASS_NAMES = list(CKPT["class_names"])
IMG_SIZE = int(CKPT.get("img_size", 128))

model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
model.load_state_dict(CKPT["model_state"])
model.to(DEVICE).eval()

TF = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def check_leaf(img: Image.Image | None):
    if img is None:
        return "Upload a rice-leaf photo first.", {}
    x = TF(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    scores = {DISPLAY.get(n, n): float(p) for n, p in zip(CLASS_NAMES, probs)}
    top = max(scores, key=scores.get)
    conf = scores[top]
    if top == "Healthy":
        verdict = f"Okay — this leaf looks healthy ({conf:.0%})"
    else:
        verdict = f"Not okay — likely {top} ({conf:.0%})"
    return verdict, scores


demo = gr.Interface(
    fn=check_leaf,
    inputs=gr.Image(type="pil", label="Rice leaf photo"),
    outputs=[gr.Textbox(label="Result"), gr.Label(label="Class scores")],
    title="Rice leaf check",
    description="Upload a close-up of one rice leaf. Okay = healthy. Not okay = a likely disease.",
)

if __name__ == "__main__":
    demo.launch()
