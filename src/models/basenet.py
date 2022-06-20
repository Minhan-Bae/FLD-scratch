import torch
import torch.nn as nn


class BfSNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=54, pretrained=False):
        super().__init__()
        self.resnet = self._build_resnet("resnet18", in_channels, pretrained)

        self.resnet4 = ResNet4(1024, 2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048 + 2, out_channels)

        self.log_weights_points = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def _build_resnet(self, model_name, in_channels, pretrained):
        # load model
        resnet = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=pretrained)

        # change in_channels
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # disable unused parameters
        for param in resnet.fc.parameters():
            param.requires_grad = False

        return resnet

    def _forward_without_fc(self, x, resnet):
        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)

        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        x = resnet.layer4(x)
        return x

    def forward(self):
        front = self._forward_without_fc(front, self.front_resnet)

        x = self.resnet4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block1 = ResNetBlock(in_channels, out_channels, downsample=True)
        self.block2 = ResNetBlock(out_channels, out_channels, downsample=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        stride = (2, 2) if downsample else (1, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out