import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join("src", "classification"))
sys.path.insert(0, os.path.join("src", "explain"))
from model import build_model
from gradcam import heatmap_rgba, raw_gradcam, to_heatmap


def test_gradcam_equals_classifier_weighted_features():
    # DenseNet ends in global-average-pool then a linear layer, so the Grad-CAM weights must
    # equal the classifier weights divided by the number of map cells. This checks the gradient code.
    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    cls = 3
    cam, logits = raw_gradcam(model, x, cls)
    with torch.no_grad():
        act = F.relu(model.features(x))
        w = model.classifier.weight[cls].view(1, -1, 1, 1) / (act.shape[2] * act.shape[3])
        expected = F.relu((w * act).sum(dim=1))[0]
    assert float(expected.max()) > 0
    assert torch.allclose(cam, expected, rtol=1e-3, atol=1e-6)


def test_heatmap_is_resized_normalised_and_colourised():
    heat = to_heatmap(torch.rand(7, 7), 224)
    assert heat.shape == (224, 224)
    assert heat.min() >= 0.0 and heat.max() <= 1.0 + 1e-6
    rgba = heatmap_rgba(heat)
    assert rgba.shape == (224, 224, 4) and rgba.dtype == np.uint8
