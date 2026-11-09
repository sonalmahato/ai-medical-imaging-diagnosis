import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..\\classification", "..\\data", "."):
    sys.path.insert(0, os.path.join(HERE, sub))
from model import build_model
from transforms import get_transforms
from dataset import CLASSES
from gradcam import raw_gradcam, to_heatmap

TARGETS = ["Cardiomegaly", "Effusion", "Pneumothorax", "Mass", "Atelectasis", "Edema"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=r"C:\Users\sonal\results_full\best.pth")
    ap.add_argument("--full_splits", default=r"C:\Users\sonal\results_full\splits_full.csv")
    ap.add_argument("--sample_splits", default="data/processed/splits.csv")
    ap.add_argument("--img_dir", default="data/processed/images")
    ap.add_argument("--calib", default="results/calibration.json")
    ap.add_argument("--out", default="docs/images/gradcam_examples.png")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if not os.path.isdir(a.img_dir):
        sys.exit("folder not found: " + a.img_dir)
    T = json.load(open(a.calib))["temperature"]
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()

    # only images that were in the FULL test split, so the model never trained on them
    full = pd.read_csv(a.full_splits, usecols=["Image Index", "split"]).rename(columns={"split": "full_split"})
    df = pd.read_csv(a.sample_splits).merge(full, on="Image Index")
    df = df[df["full_split"] == "test"]
    df = df[df["Image Index"].apply(lambda n: os.path.exists(os.path.join(a.img_dir, n)))]

    picks = []
    for c in TARGETS:
        pool = df[df[c] == 1]
        if len(pool):
            picks.append((c, pool.sample(1, random_state=a.seed).iloc[0]["Image Index"]))

    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.6))
    for j, (c, name) in enumerate(picks):
        img = Image.open(os.path.join(a.img_dir, name)).convert("L")
        x = get_transforms(False)(img)[None]
        ci = CLASSES.index(c)
        cam, logits = raw_gradcam(model, x, ci)
        heat = to_heatmap(cam, 224)
        p = float(torch.sigmoid(logits[ci] / T))
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(c, fontsize=11)
        axes[1, j].imshow(img, cmap="gray")
        axes[1, j].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, j].set_title("model confidence %.0f%%" % (100 * p), fontsize=10)
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    fig.suptitle("Grad-CAM on random test-set images (each labelled positive for the finding shown). "
                 "For education only, not clinical use.", fontsize=10)
    plt.tight_layout(h_pad=2.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
