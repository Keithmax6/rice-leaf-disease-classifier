#!/usr/bin/env python3
"""Single-page rice leaf checker. Run: python site/app.py then open http://127.0.0.1:7860"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
CANDIDATE_WEIGHTS = [
    ROOT / "rice_leaf_disease.pt",
    ROOT.parent / "rice_leaf_disease.pt",
]

DISPLAY = {
    "Healthy": "Healthy",
    "Bacterialblight": "Bacterial leaf blight",
    "Blast": "Rice blast",
    "Brownspot": "Brown spot",
    "Tungro": "Rice tungro",
}

MODEL = None
CLASS_NAMES = []
TRANSFORM = None
DEVICE = None
IMG_SIZE = 128


def load_model():
    global MODEL, CLASS_NAMES, TRANSFORM, DEVICE, IMG_SIZE
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    ckpt_path = next((p for p in CANDIDATE_WEIGHTS if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError("rice_leaf_disease.pt not found. Put it next to site/app.py or in the repo root.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    CLASS_NAMES = list(ckpt["class_names"])
    IMG_SIZE = int(ckpt.get("img_size", 128))
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    MODEL = model
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    print(f"loaded {ckpt_path} on {DEVICE} classes={CLASS_NAMES}")


def predict_pil(img: Image.Image) -> dict:
    import torch

    x = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1)[0].tolist()
    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda t: -t[1])
    top_name, top_p = ranked[0]
    ok = top_name == "Healthy"
    return {
        "ok": ok,
        "label": top_name,
        "display_name": DISPLAY.get(top_name, top_name),
        "confidence": top_p,
        "scores": [{"label": DISPLAY.get(n, n), "class": n, "prob": p} for n, p in ranked],
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            return
        self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send(404, b"Not found", "text/plain")
            return
        try:
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                raise ValueError("Send an image as multipart form field 'image'")
            import cgi

            env = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": ctype,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            }
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=env)
            item = form["image"] if "image" in form else None
            if item is None or not getattr(item, "file", None):
                raise ValueError("Missing image file")
            img = Image.open(item.file)
            result = predict_pil(img)
            self._send(200, json.dumps(result).encode(), "application/json")
        except Exception as exc:
            payload = json.dumps({"error": str(exc)}).encode()
            self._send(400, payload, "application/json")

    def log_message(self, fmt: str, *args) -> None:
        print(self.address_string(), "-", fmt % args)


def main() -> None:
    load_model()
    host, port = "0.0.0.0", 7860
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Open http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
