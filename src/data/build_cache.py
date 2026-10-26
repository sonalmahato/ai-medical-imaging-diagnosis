import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]
SIZE = 224


def load(path):
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def find(root, pattern):
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    assert hits, f"{pattern} not found under {root}"
    print("using", hits[0])
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="random subset for a quick test")
    a = ap.parse_args()

    df = pd.read_csv(find(a.raw, "Data_Entry_2017*.csv"))
    trainval = set(open(find(a.raw, "train_val_list.txt")).read().split())
    test = set(open(find(a.raw, "test_list.txt")).read().split())
    if a.limit:
        df = df.sample(a.limit, random_state=a.seed).reset_index(drop=True)

    labels = df["Finding Labels"].str.split("|")
    for c in CLASSES:
        df[c] = labels.apply(lambda L: int(c in L))
    df["Age"] = df["Patient Age"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df.loc[df["Age"] > 120, "Age"] = None

    # NIH's official split is patient-wise; carve validation out of train_val by patient
    is_test = df["Image Index"].isin(test)
    is_tv = df["Image Index"].isin(trainval)
    assert (is_test ^ is_tv).all(), "some images are in neither or both official lists"
    tv_patients = df.loc[is_tv, "Patient ID"].unique()
    np.random.RandomState(a.seed).shuffle(tv_patients)
    val_p = set(tv_patients[: int(len(tv_patients) * a.val_frac)])
    df["split"] = "train"
    df.loc[is_test, "split"] = "test"
    df.loc[is_tv & df["Patient ID"].isin(val_p), "split"] = "val"
    assert (df.groupby("Patient ID")["split"].nunique() == 1).all(), "patient leakage"
    assert set(df["split"]) == {"train", "val", "test"}, "a split is empty"

    paths = {os.path.basename(p): p for p in glob.glob(f"{a.raw}/**/*.png", recursive=True)}
    missing = [n for n in df["Image Index"] if n not in paths]
    assert not missing, f"{len(missing)} images not found, e.g. {missing[:3]}"
    df["idx"] = np.arange(len(df))

    os.makedirs(a.out, exist_ok=True)
    n = len(df)
    mm = np.lib.format.open_memmap(os.path.join(a.out, "images_224.npy"), mode="w+",
                                   dtype=np.uint8, shape=(n, SIZE, SIZE))
    todo = [paths[x] for x in df["Image Index"]]
    with Pool(a.workers) as pool:
        for i, arr in enumerate(tqdm(pool.imap(load, todo, chunksize=32), total=n)):
            mm[i] = arr
    mm.flush()
    df.to_csv(os.path.join(a.out, "splits_full.csv"), index=False)
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()