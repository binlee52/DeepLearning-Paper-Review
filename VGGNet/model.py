import torch
import torch.nn as nn
from torchsummary import summary

class VGGNet(nn.Module):

    def __init__(self, cfg, num_classes=1000, batch_norm=True):
        self.cfg = cfg
        self.batch_norm = batch_norm
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]
            else:
                f, out_channels = v.split("-")
                kernel_size = int(f[-1])
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers)


cfgs = {
    "A": ["conv3-64", 'M', "conv3-128", 'M', "conv3-256", "conv3-256", 'M', "conv3-512", "conv3-512", 'M',  "conv3-512",
          "conv3-512", 'M'],

    "B": ["conv3-64", "conv3-64", 'M', "conv3-128", "conv3-128", 'M', "conv3-256", "conv3-256", 'M',
          "conv3-512", "conv3-512", 'M', "conv3-512", "conv3-512", 'M'],

    "C": ["conv3-64", "conv3-64", "M", "conv3-128", "conv3-128", "M", "conv3-256", "conv3-256", "conv1-256", "M",
          "conv3-512", "conv3-512", "conv1-512", "M", "conv3-512", "conv3-512", "conv1-512", "M"],

    "D": ["conv3-64", "conv3-64", "M", "conv3-128", "conv3-128", "M", "conv3-256", "conv3-256", "conv3-256", "M",
          "conv3-512", "conv3-512", "conv3-512", "M", "conv3-512", "conv3-512", "conv3-512", "M"],

    "E": ["conv3-64", "conv3-64", "M", "conv3-128", "conv3-128", "M", "conv3-256", "conv3-256", "conv3-256",
          "conv3-256", "M", "conv3-512", "conv3-512", "conv3-512", "conv3-512", "M", "conv3-512", "conv3-512",
          "conv3-512", "conv3-512", "M"]
}

model = VGGNet(cfg=cfgs["E"])
summary(model, (3, 224, 224), device="cpu")