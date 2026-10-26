import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join("src", "classification"))
sys.path.insert(0, os.path.join("src", "data"))
from dataset import CXRDataset, CLASSES
from transforms import get_transforms


def test_cache_dataset_reads_the_right_row(tmp_path):
    arr = np.zeros((3, 224, 224), dtype=np.uint8)
    arr[1] = 255                                   # only row 1 is white
    np.save(tmp_path / "c.npy", arr)
    df = pd.DataFrame({"Image Index": ["a.png", "b.png", "c.png"],
                       "split": ["train", "train", "val"], "idx": [0, 1, 2]})
    for c in CLASSES:
        df[c] = 0
    df.loc[1, "Effusion"] = 1
    df.to_csv(tmp_path / "s.csv", index=False)

    ds = CXRDataset(str(tmp_path / "s.csv"), str(tmp_path / "c.npy"), "train", get_transforms(False))
    x, y = ds[1]
    assert x.shape == (3, 224, 224)
    assert float(x.min()) > 2                      # the white image, after normalization
    assert y[CLASSES.index("Effusion")] == 1