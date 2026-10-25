import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
from transforms import get_transforms
from dataset import CXRDataset, CLASSES
from model import build_model


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x.to(device))
            ps.append(torch.sigmoid(logits.float()).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def auroc(y, p):
    per_class = {}
    for i, c in enumerate(CLASSES):
        if 0 < y[:, i].sum() < len(y):   # need both positives and negatives
            per_class[c] = float(roc_auc_score(y[:, i], p[:, i]))
    mean = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/splits.csv")
    ap.add_argument("--img_dir", default="data/processed/images")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="use only N images per split (smoke test)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = {s: CXRDataset(a.csv, a.img_dir, s, get_transforms(train=(s == "train")), a.limit)
          for s in ["train", "val", "test"]}
    dl = {s: DataLoader(ds[s], batch_size=a.bs, shuffle=(s == "train"),
                        num_workers=a.workers, pin_memory=device.type == "cuda")
          for s in ds}

    model = build_model().to(device)
    loss_fn = nn.BCEWithLogitsLoss()   # multi-label: each disease is its own yes/no
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history, best = [], -1.0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in tqdm(dl["train"], desc=f"epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)

        yv, pv = predict(model, dl["val"], device)
        val_auc, _ = auroc(yv, pv)
        sched.step(val_auc)
        row = {"epoch": epoch, "train_loss": round(total / len(ds["train"]), 4),
               "val_macro_auroc": round(val_auc, 4)}
        history.append(row)
        print(row)
        if val_auc > best:
            best = val_auc
            torch.save(model.state_dict(), os.path.join(a.out, "best.pth"))

    # test set is touched ONCE, with the best checkpoint
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pth"), map_location=device))
    yt, pt = predict(model, dl["test"], device)
    test_auc, per_class = auroc(yt, pt)
    print("TEST macro AUROC:", round(test_auc, 4))
    for c, v in per_class.items():
        print(f"  {c:20s} {v:.3f}")

    json.dump({"args": vars(a), "history": history, "best_val_macro_auroc": best,
               "test_macro_auroc": test_auc, "test_per_class_auroc": per_class,
               "n_images": {s: len(ds[s]) for s in ds}},
              open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    np.savez(os.path.join(a.out, "preds.npz"), yv=yv, pv=pv, yt=yt, pt=pt)


if __name__ == "__main__":
    main()