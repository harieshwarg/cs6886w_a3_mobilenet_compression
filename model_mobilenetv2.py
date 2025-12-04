
import torch.nn as nn
import torchvision

def mobilenet_v2_cifar10(
    num_classes: int = 10,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    stride1_stem: bool = True,
):
    """
    MobileNet-V2 adapted for CIFAR-10:

    - Uses stride=1 in the initial stem so 32×32 resolution is preserved longer
    - Replaces classifier with Dropout + Linear(num_classes)
    - width_mult controls model scaling
    """

    m = torchvision.models.mobilenet_v2(weights=None, width_mult=width_mult)

    if stride1_stem:
        m.features[0][0].stride = (1, 1)

    in_feats = m.classifier[-1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_feats, num_classes),
    )
    return m
