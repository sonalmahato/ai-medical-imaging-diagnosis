import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join("src", "data"))
from seg_transforms import seg_transforms


def _make_case(tmp_path):
    rng = np.random.RandomState(0)
    img = rng.normal(40, 20, size=(64, 64, 30)).astype("float32")   # HU-like values
    lab = np.zeros((64, 64, 30), dtype="uint8")
    lab[20:40, 20:40, 8:20] = 1
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    nib.save(nib.Nifti1Image(img, affine), str(tmp_path / "img.nii.gz"))
    nib.save(nib.Nifti1Image(lab, affine), str(tmp_path / "lab.nii.gz"))
    return {"image": str(tmp_path / "img.nii.gz"), "label": str(tmp_path / "lab.nii.gz")}


def test_train_transform_pads_small_scans_and_returns_four_patches(tmp_path):
    out = seg_transforms(train=True)(_make_case(tmp_path))
    assert len(out) == 4
    assert tuple(out[0]["image"].shape) == (1, 96, 96, 96)
    assert tuple(out[0]["label"].shape) == (1, 96, 96, 96)


def test_val_transform_scales_intensity_and_keeps_binary_labels(tmp_path):
    out = seg_transforms(train=False)(_make_case(tmp_path))
    assert float(out["image"].min()) >= 0.0 and float(out["image"].max()) <= 1.0
    assert set(np.unique(np.asarray(out["label"]))) <= {0, 1}
