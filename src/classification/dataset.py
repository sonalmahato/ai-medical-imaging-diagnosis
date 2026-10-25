import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]


class CXRDataset(Dataset):
    def __init__(self, csv_path, img_dir, split, transform, limit=None):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if limit:
            df = df.head(limit)
        self.names = df["Image Index"].tolist()
        self.labels = df[CLASSES].values.astype("float32")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.names[i])).convert("L")
        return self.transform(img), torch.from_numpy(self.labels[i])