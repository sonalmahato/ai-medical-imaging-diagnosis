from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld, RandAffined,
    RandShiftIntensityd, EnsureTyped)

PIXDIM = (1.5, 1.5, 2.0)
HU_MIN, HU_MAX = -57, 164     # soft-tissue window used for the spleen
PATCH = (96, 96, 96)


def seg_transforms(train: bool):
    k = ["image", "label"]
    t = [
        LoadImaged(keys=k),
        EnsureChannelFirstd(keys=k),
        Orientationd(keys=k, axcodes="RAS"),                                      # same orientation for every scan
        Spacingd(keys=k, pixdim=PIXDIM, mode=("bilinear", "nearest")),            # same voxel size
        ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=k, source_key="image"),
    ]
    if train:
        t += [
            SpatialPadd(keys=k, spatial_size=PATCH),
            RandCropByPosNegLabeld(keys=k, label_key="label", spatial_size=PATCH,
                                   pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
            RandAffined(keys=k, prob=0.3, rotate_range=(0.1, 0.1, 0.1),
                        scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        ]
    return Compose(t + [EnsureTyped(keys=k)])
