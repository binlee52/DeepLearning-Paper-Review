import torch
import torch.nn as nn

# ignore user warnings
import warnings

warnings.filterwarnings("ignore")


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, down=None):
        super(BasicBlock, self).__init__()
        self.down = down
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down is not None:
            identity = self.down(identity)

        out += identity
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, down=None):
        super(BottleNeck, self).__init__()
        self.down = down
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = conv1x1(out_planes, out_planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.down is not None:
            identity = self.down(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layers(block, 64, layers[0], stride=1)
        self.conv3 = self._make_layers(block, 128, layers[1], stride=2)
        self.conv4 = self._make_layers(block, 256, layers[2], stride=2)
        self.conv5 = self._make_layers(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, out_planes, blocks, stride=1):
        layers = []
        down = None
        if (stride != 1) or self.in_planes != out_planes * block.expansion:
            down = nn.Sequential(
                conv1x1(self.in_planes, out_planes * block.expansion, stride=stride),
                nn.BatchNorm2d(out_planes * block.expansion),)
        layers.append(block(self.in_planes, out_planes, stride, down))
        self.in_planes = out_planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)


def resnet18(num_classes=1000):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


def resnet34(num_classes=1000):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    return model


def resnet50(num_classes=1000):
    model = ResNet(BottleNeck, [3, 4, 6, 3], num_classes)
    return model


def resnet101(num_classes=1000):
    model = ResNet(BottleNeck, [3, 4, 23, 3], num_classes)
    return model


def resnet152(num_classes=1000):
    model = ResNet(BottleNeck, [3, 8, 36, 3], num_classes)
    return model


if __name__ == "__main__":
    model = resnet50(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)

    from torchsummary import summary

    summary(model, input_size=(3, 224, 224), device="cpu")
