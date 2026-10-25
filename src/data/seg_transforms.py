from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, RandAffined,
    RandShiftIntensityd, EnsureTyped)

def seg_transforms(train: bool):
    k = ["image", "label"]
    t = [
        LoadImaged(keys=k),
        EnsureChannelFirstd(keys=k),
        Orientationd(keys=k, axcodes="RAS"),                        # same orientation for every scan
        Spacingd(keys=k, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),  # same voxel size
        ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),  # spleen HU window
        CropForegroundd(keys=k, source_key="image"),
    ]
    if train:
        t += [
            RandCropByPosNegLabeld(keys=k, label_key="label", spatial_size=(96, 96, 96),
                                   pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
            RandAffined(keys=k, prob=0.3, rotate_range=(0.1, 0.1, 0.1),
                        scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        ]
    return Compose(t + [EnsureTyped(keys=k)])