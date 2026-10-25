import os
import sys

import torch

sys.path.insert(0, os.path.join("src", "classification"))
from model import build_model


def test_model_output_shape():
    m = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 14)