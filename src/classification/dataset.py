import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]


class CXRDataset(Dataset):
    """img_dir is a folder of PNGs (sample run) or a .npy cache file (full run)."""

    def __init__(self, csv_path, img_dir, split, transform, limit=None):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if limit:
            df = df.head(limit)
        self.names = df["Image Index"].tolist()
        self.labels = df[CLASSES].values.astype("float32")
        self.use_cache = img_dir.endswith(".npy")
        self.idx = df["idx"].values if self.use_cache else None
        self.img_dir = img_dir
        self.transform = transform
        self.arr = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self.use_cache:
            if self.arr is None:   # opened lazily, once per data-loader worker
                self.arr = np.load(self.img_dir, mmap_mode="r")
            img = Image.fromarray(np.array(self.arr[self.idx[i]]))
        else:
            img = Image.open(os.path.join(self.img_dir, self.names[i])).convert("L")
        return self.transform(img), torch.from_numpy(self.labels[i])
