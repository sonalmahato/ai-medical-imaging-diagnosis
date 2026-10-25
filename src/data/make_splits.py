import glob
import os
import numpy as np
import pandas as pd
import yaml

P = yaml.safe_load(open("params.yaml"))
CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]

RAW = os.environ.get("RAW_DIR", "data/raw/nih_sample")
csv = glob.glob(f"{RAW}/**/sample_labels.csv", recursive=True)[0]
df = pd.read_csv(csv)

# multi-hot labels (No Finding = all zeros)
labels = df["Finding Labels"].str.split("|")
for c in CLASSES:
    df[c] = labels.apply(lambda L: int(c in L))

# clean age (text like "060Y"); impossible ages become NaN
df["Age"] = df["Patient Age"].astype(str).str.extract(r"(\d+)")[0].astype(float)
df.loc[df["Age"] > 120, "Age"] = None

# patient-wise split: shuffle patients, then cut the list into three parts
s = P["split"]
patients = df["Patient ID"].unique()
np.random.RandomState(P["seed"]).shuffle(patients)
n_train = int(len(patients) * s["train"])
n_val = int(len(patients) * s["val"])
train_p = set(patients[:n_train])
val_p = set(patients[n_train:n_train + n_val])
df["split"] = df["Patient ID"].map(
    lambda p: "train" if p in train_p else ("val" if p in val_p else "test"))

assert set(df["split"]) == {"train", "val", "test"}, "a split is empty"
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/splits.csv", index=False)
print(df["split"].value_counts())