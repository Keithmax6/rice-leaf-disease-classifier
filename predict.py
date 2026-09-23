#!/usr/bin/env python3
"""Predict rice-leaf disease from one or more images."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    img_size = int(ckpt.get("img_size", 160))
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, class_names, tf


def predict_image(model, tf, class_names, path: Path, device):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    ranked = sorted(zip(class_names, probs.tolist()), key=lambda t: -t[1])
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", default="rice_leaf_disease.pt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, tf = load_model(Path(args.ckpt), device)
    for p in args.images:
        ranked = predict_image(model, tf, names, Path(p), device)
        top, conf = ranked[0]
        print(f"{p}")
        print(f"  prediction: {top}  ({conf:.1%})")
        for name, pr in ranked:
            print(f"    {name:20s} {pr:.3f}")


if __name__ == "__main__":
    main()
