import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]
EPS = 1e-6
MIN_POS = 20   # a class needs this many positives AND negatives in every subgroup


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def macro_auroc(y, p, cols):
    scores = []
    for i in cols:
        yi = y[:, i]
        if 0 < yi.sum() < len(yi):
            scores.append(roc_auc_score(yi, p[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


def bce(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece_quantile(y, p, bins=10):
    """Class-wise expected calibration error with equal-count bins, averaged over classes."""
    vals = []
    for i in range(y.shape[1]):
        order = np.argsort(p[:, i])
        yi, pi = y[order, i], p[order, i]
        total = 0.0
        for c in np.array_split(np.arange(len(yi)), bins):
            total += len(c) * abs(yi[c].mean() - pi[c].mean())
        vals.append(total / len(yi))
    return float(np.mean(vals))


def fit_temperature(z, y):
    res = minimize_scalar(lambda t: bce(y, sigmoid(z / t)), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def reliability_points(y, p, bins=10):
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return [p[c].mean() for c in chunks], [y[c].mean() for c in chunks]


def patient_bootstrap_ci(y, p, cols, pids, n_boot, seed=0):
    """95% interval from resampling whole patients (images of one patient stay together)."""
    rng = np.random.RandomState(seed)
    by_patient = pd.Series(np.arange(len(pids))).groupby(pids).indices
    keys = list(by_patient.keys())
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(keys), len(keys))
        rows = np.concatenate([by_patient[keys[k]] for k in pick])
        vals.append(macro_auroc(y[rows], p[rows], cols))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=r"C:\Users\sonal\results_full\preds.npz")
    ap.add_argument("--splits", default=r"C:\Users\sonal\results_full\splits_full.csv")
    ap.add_argument("--metrics", default=r"C:\Users\sonal\results_full\metrics.json")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("docs/images", exist_ok=True)

    d = np.load(a.preds)
    yv, pv, yt, pt = d["yv"], d["pv"], d["yt"], d["pt"]
    df = pd.read_csv(a.splits)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    assert len(val) == len(yv) and len(test) == len(yt), "row counts do not match"
    assert np.array_equal(yt, test[CLASSES].values.astype("float32")), "test rows not aligned"
    assert np.array_equal(yv, val[CLASSES].values.astype("float32")), "val rows not aligned"
    allc = list(range(len(CLASSES)))
    print("alignment OK; test macro AUROC =", round(macro_auroc(yt, pt, allc), 4))

    # ---- calibration: temperature fitted on validation, reported on test ----
    T = fit_temperature(logit(pv), yv)
    pt_cal = sigmoid(logit(pt) / T)
    cal = {
        "temperature": T,
        "test_nll": {"before": bce(yt, pt), "after": bce(yt, pt_cal)},
        "test_brier": {"before": brier(yt, pt), "after": brier(yt, pt_cal)},
        "test_ece_classwise_10bins": {"before": ece_quantile(yt, pt), "after": ece_quantile(yt, pt_cal)},
        "test_macro_auroc": {"before": macro_auroc(yt, pt, allc), "after": macro_auroc(yt, pt_cal, allc)},
    }
    with open("results/calibration.json", "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))

    show = ["Effusion", "Cardiomegaly", "Atelectasis", "Infiltration"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, name in zip(axes, show):
        i = CLASSES.index(name)
        xb, yb = reliability_points(yt[:, i], pt[:, i])
        xa, ya = reliability_points(yt[:, i], pt_cal[:, i])
        m = max(max(xb), max(yb), max(xa), max(ya)) * 1.05
        ax.plot([0, m], [0, m], "k--", label="perfect")
        ax.plot(xb, yb, "o-", label="before")
        ax.plot(xa, ya, "s-", label="after (T=%.2f)" % T)
        ax.set_title(name)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig("docs/images/reliability.png", dpi=110)
    plt.close()

    # ---- per-class bar chart and learning curve for the presentation ----
    pc = {c: float(roc_auc_score(yt[:, i], pt[:, i])) for i, c in enumerate(CLASSES)}
    names = sorted(pc, key=pc.get)
    plt.figure(figsize=(6.5, 5))
    plt.barh(names, [pc[n] for n in names])
    plt.axvline(0.5, color="k", linestyle="--", linewidth=1)
    plt.xlim(0.4, 1.0)
    plt.xlabel("test AUROC")
    plt.title("Per-class AUROC, official NIH test set")
    plt.tight_layout()
    plt.savefig("docs/images/per_class_auroc.png", dpi=110)
    plt.close()

    if os.path.exists(a.metrics):
        with open(a.metrics) as f:
            h = json.load(f)["history"]
        ep = [r["epoch"] for r in h]
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        axs[0].plot(ep, [r["train_loss"] for r in h], "o-")
        axs[0].set_title("Training loss")
        axs[1].plot(ep, [r["val_macro_auroc"] for r in h], "o-")
        axs[1].set_title("Validation macro AUROC")
        for ax in axs:
            ax.set_xlabel("epoch")
        plt.tight_layout()
        plt.savefig("docs/images/learning_curve.png", dpi=110)
        plt.close()

    # ---- results by subgroup (same classes in every group, so groups are comparable) ----
    age_group = pd.cut(test["Age"], [-1, 39, 59, 120], labels=["<40", "40-59", "60+"])
    groups = {
        "overall": np.ones(len(test), dtype=bool),
        "sex=M": (test["Patient Gender"] == "M").values,
        "sex=F": (test["Patient Gender"] == "F").values,
        "view=PA": (test["View Position"] == "PA").values,
        "view=AP": (test["View Position"] == "AP").values,
        "age<40": (age_group == "<40").values,
        "age40-59": (age_group == "40-59").values,
        "age60+": (age_group == "60+").values,
    }
    common = set(allc)
    for mask in groups.values():
        y = yt[mask]
        ok = {i for i in allc if y[:, i].sum() >= MIN_POS and (len(y) - y[:, i].sum()) >= MIN_POS}
        common &= ok
    cols = sorted(common)
    used = ";".join(CLASSES[i] for i in cols)
    print("classes used in subgroup comparison:", used)

    pids = test["Patient ID"].values
    rows = []
    for name, mask in groups.items():
        lo, hi = patient_bootstrap_ci(yt[mask], pt[mask], cols, pids[mask], a.boot)
        rows.append({"group": name, "n_images": int(mask.sum()),
                     "n_patients": int(len(np.unique(pids[mask]))),
                     "macro_auroc": round(macro_auroc(yt[mask], pt[mask], cols), 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                     "classes_used": used})
        print({k: rows[-1][k] for k in ("group", "n_images", "macro_auroc", "ci95_low", "ci95_high")})
    pd.DataFrame(rows).to_csv("results/subgroups.csv", index=False)


if __name__ == "__main__":
    main()
