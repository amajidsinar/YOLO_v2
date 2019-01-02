import torch
import torch.nn as nn
import torch.nn.functional as F

class Darknet_v1(nn.Module):
    

Darknet_v1 = nn.Sequential(
    # 1st block
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 2nd block
    nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 3rd block
    nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 4th block
    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 5th block
    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    # 6th block
    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=1024, out_channels=4096, kernel_size=1),
    nn.Conv2d(in_channels=4096, out_channels=90, kernel_size=1),
)