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
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.down = stride != 1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()
        mid_planes = out_planes//4
        self.down = stride != 1 or (in_planes != out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_planes, mid_planes, stride)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = conv3x3(mid_planes, mid_planes)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = conv1x1(mid_planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_planes)
        )

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
        
        if self.down:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            BottleNeck(64, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256),
        )
        self.conv3 = nn.Sequential(
            BottleNeck(256, 512, stride=2),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
        )
        self.conv4 = nn.Sequential(
            BottleNeck(512, 1024, stride=2),
            BottleNeck(1024, 1024),
            BottleNeck(1024, 1024),
            BottleNeck(1024, 1024),
            BottleNeck(1024, 1024),
            BottleNeck(1024, 1024),
        )
        self.conv5 = nn.Sequential(
            BottleNeck(1024, 2048, stride=2),
            BottleNeck(2048, 2048),
            BottleNeck(2048, 2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

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

if __name__ == "__main__":
    model = ResNet(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out.shape)
    # model = BasicBlock(in_planes=3, out_planes=64, stride=2)
    # out = model(x)
    # print(out.shape)
    from torchsummary import summary
    summary(model, input_size=(3, 224, 224), device="cpu")
