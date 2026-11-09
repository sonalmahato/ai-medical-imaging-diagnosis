import argparse
import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from seg_transforms import seg_transforms

ROI = (160, 160, 160)   # sliding-window size used at inference
POST_PRED = Compose([AsDiscrete(argmax=True, to_onehot=2)])
POST_LABEL = Compose([AsDiscrete(to_onehot=2)])


def build_model():
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)


def find_cases(root):
    imgs = sorted(p for p in glob.glob(os.path.join(root, "**", "imagesTr", "*.nii.gz"), recursive=True)
                  if not os.path.basename(p).startswith("."))
    labs = sorted(p for p in glob.glob(os.path.join(root, "**", "labelsTr", "*.nii.gz"), recursive=True)
                  if not os.path.basename(p).startswith("."))
    assert imgs and len(imgs) == len(labs), "image/label files not found or not matching"
    return [{"image": i, "label": l} for i, l in zip(imgs, labs)]


def predict(model, x, device):
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        return sliding_window_inference(x, ROI, 4, model)


def evaluate(model, loader, device):
    model.eval()
    metric = DiceMetric(include_background=False, reduction="mean")
    scores = []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = predict(model, x, device).float()
        pred = [POST_PRED(i) for i in decollate_batch(logits)]
        lab = [POST_LABEL(i) for i in decollate_batch(y)]
        metric(y_pred=pred, y=lab)
        scores.append(float(metric.aggregate().item()))
        metric.reset()
    return scores


def save_overlay(model, ds, device, path):
    model.eval()
    item = ds[0]
    x = item["image"][None].to(device)
    pred = predict(model, x, device).float().argmax(1)[0].cpu().numpy()
    img = np.asarray(item["image"][0].cpu())
    gt = np.asarray(item["label"][0].cpu())
    nz = np.where(gt.sum(axis=(0, 1)) > 0)[0]
    zs = [nz[len(nz) // 4], nz[len(nz) // 2], nz[3 * len(nz) // 4]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, z in zip(axes, zs):
        ax.imshow(np.rot90(img[:, :, z]), cmap="gray")
        ax.contour(np.rot90(gt[:, :, z]), levels=[0.5], colors="lime", linewidths=1.2)
        if pred[:, :, z].any():
            ax.contour(np.rot90(pred[:, :, z]), levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title("axial slice %d" % z)
        ax.axis("off")
    fig.suptitle("Spleen, test case: green = ground truth, red = model")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw/Task09_Spleen")
    ap.add_argument("--out", default="outputs/spleen")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--n_val", type=int, default=6)
    ap.add_argument("--n_test", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    set_determinism(seed=a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cases = find_cases(a.data_root)
    order = np.random.RandomState(a.seed).permutation(len(cases))
    test = [cases[i] for i in order[: a.n_test]]
    val = [cases[i] for i in order[a.n_test: a.n_test + a.n_val]]
    train = [cases[i] for i in order[a.n_test + a.n_val:]]
    print("cases: train %d, val %d, test %d" % (len(train), len(val), len(test)))

    train_ds = CacheDataset(train, seg_transforms(True), cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(val, seg_transforms(False), cache_rate=1.0, num_workers=2)
    test_ds = CacheDataset(test, seg_transforms(False), cache_rate=1.0, num_workers=2)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=1)
    test_dl = DataLoader(test_ds, batch_size=1)

    model = build_model().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_path = os.path.join(a.out, "best.pth")

    best, history, t0 = -1.0, [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_dl:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            n += 1
        row = {"epoch": epoch, "train_loss": round(total / n, 4)}
        if epoch % a.val_every == 0 or epoch == a.epochs:
            row["val_dice"] = round(float(np.mean(evaluate(model, val_dl, device))), 4)
            if row["val_dice"] > best:
                best = row["val_dice"]
                torch.save(model.state_dict(), best_path)
        row["seconds"] = round(time.time() - t0)
        history.append(row)
        print(row, flush=True)

    # test cases are used once, with the best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    scores = evaluate(model, test_dl, device)
    print("TEST mean Dice: %.4f +/- %.4f" % (np.mean(scores), np.std(scores)))
    for c, s in zip(test, scores):
        print("  %-24s %.4f" % (os.path.basename(c["image"]), s))
    save_overlay(model, test_ds, device, os.path.join(a.out, "overlay.png"))

    ep = [r["epoch"] for r in history]
    v = [(r["epoch"], r["val_dice"]) for r in history if "val_dice" in r]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    axs[0].plot(ep, [r["train_loss"] for r in history])
    axs[0].set_title("Training loss (Dice + cross-entropy)")
    axs[1].plot([e for e, _ in v], [d for _, d in v], "o-")
    axs[1].set_title("Validation Dice")
    for ax in axs:
        ax.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "learning_curve.png"), dpi=110)
    plt.close()

    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"args": vars(a), "history": history, "best_val_dice": best,
                   "test_mean_dice": float(np.mean(scores)), "test_std_dice": float(np.std(scores)),
                   "test_cases": [{"case": os.path.basename(c["image"]), "dice": round(s, 4)}
                                  for c, s in zip(test, scores)],
                   "n_cases": {"train": len(train), "val": len(val), "test": len(test)}}, f, indent=2)


if __name__ == "__main__":
    main()
