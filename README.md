# AI-Assisted Medical Imaging and Diagnosis System

>  **FOR EDUCATION ONLY. NOT FOR CLINICAL USE.**
> This project is a learning exercise. It must not be used to diagnose, treat,
> or make any medical decision.

## Overview
A platform where chest X-ray and CT scans are uploaded, analysed by deep
learning models, shown with Grad-CAM explanation heatmaps, and reviewed by a
doctor who approves or rejects each result. Every decision is audit-logged.

## Planned components
- Classification: DenseNet121 transfer learning on NIH ChestX-ray14
- Segmentation: MONAI 3D U-Net on Medical Segmentation Decathlon (Spleen)
- Explainability: Grad-CAM heatmaps
- Orthanc DICOM server with an inference worker
- React viewer with Approve/Reject and an audit log
- Docker Compose deployment, DVC data versioning

## Status
- [x] Step 1: data choice, DICOM basics, EDA
- [ ] Step 2: preprocessing and DVC
- [ ] Step 3: model training
- [ ] Step 4: Grad-CAM
- [ ] Step 5: Orthanc and inference service
- [ ] Step 6: React viewer and audit log
- [ ] Step 7: Docker, model card, documentation

## Data
Datasets are not stored in this repository. See `docs/data_decision.md`.

## Setup instructions
Coming soon.
