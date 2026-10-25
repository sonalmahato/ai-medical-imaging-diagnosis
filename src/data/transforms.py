from torchvision import transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics

def get_transforms(train: bool):
    base = [T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize(MEAN, STD)]
    if not train:
        return T.Compose(base)
    aug = [
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    return T.Compose(aug + base)