import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transforms import get_transforms

img = Image.open("data/processed/images/00000013_005.png")
aug = get_transforms(train=True)
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for a in ax.ravel():
    a.imshow(aug(img)[0], cmap="gray")
    a.axis("off")
os.makedirs("docs/images", exist_ok=True)
plt.savefig("docs/images/augmentations.png", dpi=100, bbox_inches="tight")