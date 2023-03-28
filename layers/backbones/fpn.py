import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet as vrn
from torchvision.models import mobilenet as vmn

from .resnet import ResNet
from .mobilenet import MobileNet
from .utils import register

class FPN(nn.Module):
    """
    Feature pyramid network
    """
    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features

        if isinstance(features, ResNet):
            is_light = features.bottleneck == vrn.BasicBlock
            channels = [128, 256, 512] if is_light else [512, 1024, 2048]
        elif isinstance(features, MobileNet):
            channels = [32, 96, 320]