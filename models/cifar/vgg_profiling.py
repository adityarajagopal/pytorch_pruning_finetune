'''VGGtime for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import time


__all__ = ['vgg16_bn_profiling']


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGGProfiling(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGGProfiling, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        timeList = []
        t0 = time.time()
        x = self.conv1(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn1(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv2(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn2(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.maxpool(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        
        x = self.conv3(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn3(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv4(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn4(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.maxpool(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        
        x = self.conv5(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn5(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv6(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn6(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv7(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn7(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.maxpool(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv8(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn8(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv9(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn9(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv10(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn10(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.maxpool(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv11(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn11(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv12(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn12(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.conv13(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.bn13(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        x = self.relu(x)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.maxpool(x)
        timeList.append(time.time() - t0)
        t0 = time.time()
        
        x = x.view(x.size(0), -1)
        timeList.append(time.time() - t0)
        t0 = time.time()

        x = self.classifier(x)
        timeList.append(time.time() - t0)
        return x, timeList

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_bn_profiling(**kwargs):
    """VGGtime 16-layer model (configuration "D") with batch normalization"""
    model = VGGProfiling(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model
