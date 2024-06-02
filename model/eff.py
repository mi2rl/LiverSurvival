import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch_3d import EfficientNet3D

class Eff(nn.Module):
    def __init__(self, out_size=2, in_channels=1, backbone="efficientnet-b7"):
        super(Eff, self).__init__()
        self.network = EfficientNet3D.from_name(backbone, override_params={'num_classes':out_size}, in_channels=in_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.network(x)
        x = self.sigmoid(x)
        return x