#!/usr/bin/env python3
"""Train a rice-leaf disease classifier (transfer learning).

Classes (default public dataset):
  Bacterialblight, Blast, Brownspot, Healthy, Tungro

Usage:
  python train_rice_disease.py --data-dir DATA --out-dir . --epochs 4 --freeze-backbone
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms


CLASS_HELP = {
    "Bacterialblight": "Bacterial leaf blight (Xanthomonas oryzae) — yellowing, wilting leaf edges.",
    "Blast": "Rice blast (Magnaporthe oryzae) — diamond/spindle lesions with gray centers.",
    "Brownspot": "Brown spot (Bipolaris oryzae) — many oval brown spots on the leaf.",
    "Healthy": "Healthy rice leaf — no disease symptoms.",
    "Tungro": "Rice tungro virus — yellow-orange discoloration, stunting.",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_model(num_classes: int) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(1)
            total_loss += loss.item() * x.size(0)
            correct += (pred == y).sum().item()
            n += x.size(0)
            all_y.extend(y.cpu().tolist())
            all_p.extend(pred.cpu().tolist())
    return total_loss / max(n, 1), correct / max(n, 1), all_y, all_p


def per_class_acc(y, p, class_names):
    out = {}
    for i, name in enumerate(class_names):
        idx = [k for k, t in enumerate(y) if t == i]
        if not idx:
            out[name] = None
            continue
        ok = sum(1 for k in idx if p[k] == i)
        out[name] = round(ok / len(idx), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder with one subfolder per class")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-backbone", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.img_size + 32, args.img_size + 32)),
            transforms.RandomCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    probe = datasets.ImageFolder(args.data_dir, transform=train_tf)
    class_names = list(probe.classes)
    n_val = max(1, int(len(probe) * args.val_frac))
    n_train = len(probe) - n_val
    train_split, val_split = random_split(
        probe, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_ds = Subset(datasets.ImageFolder(args.data_dir, transform=train_tf), train_split.indices)
    val_ds = Subset(datasets.ImageFolder(args.data_dir, transform=val_tf), val_split.indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(len(class_names)).to(device)
    if args.freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"frozen backbone; trainable params={sum(p.numel() for p in params)}")
        optimizer = torch.optim.AdamW(params, lr=args.lr * 3, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"device={device}  images={len(probe)}  train={n_train}  val={n_val}")
    print(f"classes={class_names}")

    history = []
    best_acc = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_ok, run_n = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            run_ok += (logits.argmax(1) == y).sum().item()
            run_n += x.size(0)
            if step == 1 or step % 40 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        scheduler.step()
        train_loss, train_acc = run_loss / run_n, run_ok / run_n
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "per_class_val_acc": per_class_acc(y_true, y_pred, class_names),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "arch": "mobilenet_v3_small",
                    "val_acc": best_acc,
                },
                out_dir / "rice_leaf_disease.pt",
            )

    labels = {
        "class_names": class_names,
        "class_help": {c: CLASS_HELP.get(c, "") for c in class_names},
        "img_size": args.img_size,
        "arch": "mobilenet_v3_small",
        "best_val_acc": best_acc,
        "train_seconds": round(time.time() - t0, 1),
        "n_train": n_train,
        "n_val": n_val,
        "history": history,
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"saved {out_dir / 'rice_leaf_disease.pt'}  best_val_acc={best_acc:.3f}")


if __name__ == "__main__":
    main()
