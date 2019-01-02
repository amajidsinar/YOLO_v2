import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1, is_maxpool=True):
        super(BasicBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.is_maxpool = is_maxpool
    def forward(self, input):
        x = self.conv3x3(input)
        x = self.relu(x)
        x = self.bn(x)
        if self.is_maxpool:
            x = self.maxpool(x)
        return x

class DarkBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(DarkBlock, self).__init__()
        self.conv3x3= nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        x = self.conv3x3(input)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv1x1(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class DetectionBlock(nn.Module):
    def __init__(self,num_detections, num_classes):
        super(DetectionBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels=1024, out_channels=(num_detections*(5+num_classes)), kernel_size=1)

    def forward(self, input):
        x = self.conv3x3(input)
        x = self.conv3x3(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        return x

class YOLO_v2(nn.Module):
    def __init__(self, num_detections, num_classes):
        super(YOLO_v2, self).__init__()
        self.block_1 = BasicBlock(in_channels=3, out_channels=32, stride=2)
        self.block_2 = BasicBlock(in_channels=32, out_channels=64, stride=1)
        self.block_3 = self._make_layer(in_channels=64, out_channels=128, num_layers=3)
        self.block_4 = self._make_layer(in_channels=128, out_channels=256, num_layers=3)
        self.block_5 = self._make_layer(in_channels=256, out_channels=512, num_layers=5)
        self.block_6 = self._make_layer(in_channels=512, out_channels=1024, num_layers=5, is_maxpool=False)
        self.block_7 = DetectionBlock(num_detections, num_classes)

    def forward(self, input):
        x = self.block_1(input)
        print(x.shape)
        x = self.block_2(x)
        print(x.shape)
        x = self.block_3(x)
        print(x.shape)
        x = self.block_4(x)
        print(x.shape)
        x = self.block_5(x)
        print(x.shape)
        x = self.block_6(x)
        print(x.shape)
        x = self.block_7(x)
        print(x.shape)
        return x


    def _make_layer(self,in_channels, out_channels, num_layers, is_maxpool=True):
        block = []
        block.append(BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=1, is_maxpool=False))
        for i in range(1, num_layers-1):
            block.append(DarkBlock(in_channels=out_channels, out_channels=in_channels))
        if is_maxpool:
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*block)

def yolo_v2(num_detections, num_classes):
    framework = YOLO_v2(num_detections, num_classes)
    return framework 