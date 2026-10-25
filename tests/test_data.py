import yaml
import pandas as pd
from PIL import Image


def test_no_patient_in_two_splits():
    df = pd.read_csv("data/processed/splits.csv")
    assert (df.groupby("Patient ID")["split"].nunique() == 1).all()


def test_processed_images_have_right_size():
    size = yaml.safe_load(open("params.yaml"))["img_size"]
    df = pd.read_csv("data/processed/splits.csv")
    for name in df["Image Index"].head(20):
        assert Image.open(f"data/processed/images/{name}").size == (size, size)


def test_all_three_splits_exist():
    df = pd.read_csv("data/processed/splits.csv")
    share = df["split"].value_counts(normalize=True)
    assert set(share.index) == {"train", "val", "test"}
    assert share["train"] > 0.6