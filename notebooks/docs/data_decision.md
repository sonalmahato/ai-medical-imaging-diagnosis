# Data Decision

For education only, not clinical use.**

## Summary

| Task | Dataset | Format | Size | License |
| Classification | NIH ChestX-ray14 | PNG + CSV labels | 112,120 images, 30,805 patients (full); 5,606 images (Kaggle sample) | CC0 (public domain) |
| Segmentation | Medical Segmentation Decathlon, Task09 Spleen | NIfTI (3D CT) | 41 labelled CT volumes | CC-BY-SA 4.0 (verify on medicaldecathlon.com) |

## Why these datasets

**NIH ChestX-ray14**
- Open access, no approval wait (MIMIC-CXR needs credentialed access).
- 14 thoracic disease labels, multi-label, with an official patient-wise train/test split.
- About 1,000 images include bounding boxes, which lets me check whether Grad-CAM
  highlights the right region.
- A 5,606-image Kaggle sample allows fast pipeline development before training on the full set.
- Rejected: CheXpert (large download, registration and redistribution terms),
  RSNA Pneumonia (binary label only).

**MSD Task09 Spleen**
- Small enough (41 volumes) to train a 3D U-Net on a free GPU.
- Supported directly by MONAI (`DecathlonDataset`), including download.
- Gives a clean, well-defined single-organ segmentation task with Dice as the metric.

## Labels and known issues

- NIH labels were mined from radiology reports with NLP, not read by radiologists.
  They are noisy, so scores measure agreement with these labels, not clinical truth.
- Results must be reported per class and macro-averaged (AUROC).

## EDA findings on the 5,606-image sample

- **Class imbalance:** No Finding = 3,044 images (about 54%). Infiltration 967,
  Effusion 644, Atelectasis 508, Nodule 313, Mass 284, Pneumothorax 271,
  Consolidation 226, Pleural Thickening 176, Cardiomegaly 141, Emphysema 127,
  Edema 118, Fibrosis 84, Pneumonia 62, Hernia 13.
- **Rare classes:** Hernia and Pneumonia are too small in the sample for reliable
  results. Real training uses the full dataset.
- **Age:** median 49, interquartile range 35 to 59. Few children and few patients
  over 80. The data contains invalid ages (maximum 411), which are filtered out.
- **Sex:** roughly 3,100 male and 2,500 female, a moderate skew.
- **View position:** both AP and PA views are present and look different
  (portable AP films often show tubes, wires and text markers).
- **Shortcut risk:** markers and devices in images may be learned instead of
  lung findings. To be checked with Grad-CAM.

## Handling decisions

- Use the official patient-wise split on the full dataset, so images from one
  patient never appear in both train and test.
- Raw data is not stored in Git. It is kept out via `.gitignore` and will be
  versioned with DVC.
- NIH PNG and MSD NIfTI files will be converted to DICOM (pydicom, SimpleITK)
  so they can be uploaded to Orthanc and tested end to end.
- Anonymize any DICOM metadata before use.

## Sources

- NIH ChestX-ray14: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database
  and Benchmarks on Weakly-Supervised Classification and Localization of Common
  Thorax Diseases" (CVPR 2017).
- Medical Segmentation Decathlon: Antonelli et al., "The Medical Segmentation
  Decathlon", Nature Communications, 2022.
