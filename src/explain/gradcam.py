import numpy as np
import torch
import torch.nn.functional as F


def raw_gradcam(model, x, class_idx):
    """Grad-CAM for one image x of shape (1, 3, H, W) and one output class.

    Returns the low-resolution map (7 x 7 for a 224 x 224 input) and the 14 raw logits."""
    model.eval()
    with torch.enable_grad():
        feats = model.features(x)
        act = F.relu(feats)                       # not in-place, so its gradient can be read
        pooled = F.adaptive_avg_pool2d(act, (1, 1)).flatten(1)
        logits = model.classifier(pooled)
        grad = torch.autograd.grad(logits[0, class_idx], act)[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)                # one weight per feature map
    cam = F.relu((weights * act.detach()).sum(dim=1))[0]         # keep only positive evidence
    return cam.detach().cpu(), logits[0].detach().cpu()


def to_heatmap(cam, size=224):
    """Resize a low-resolution map to the image size and scale it to 0..1."""
    cam = F.interpolate(cam[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).numpy()


def heatmap_rgba(heat, alpha_scale=0.6):
    """Heat map (0..1) -> RGBA uint8 image, see-through where the map is weak (used by the viewer later)."""
    import matplotlib
    rgba = np.array(matplotlib.colormaps["jet"](heat))
    rgba[..., 3] = heat * alpha_scale
    return (rgba * 255).astype(np.uint8)
