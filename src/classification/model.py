import torch.nn as nn
from torchvision import models


def build_model(num_classes=14, pretrained=True):
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.densenet121(weights=weights)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)  # new 14-output head
    return m