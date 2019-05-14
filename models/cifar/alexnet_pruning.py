import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from pruning.layers import MaskedLinear, MaskedConv2d 


__all__ = ['alexnet_pruning', 'AlexNetPruning']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNetPruning(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNetPruning, self).__init__()
        self.conv1 = MaskedConv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = MaskedConv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = MaskedConv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = MaskedConv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = MaskedConv2d(256, 256, kernel_size=3, padding=1)
        self.linear = MaskedLinear(256, num_classes)
        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3,
            nn.ReLU(inplace=True),
            self.conv4,
            nn.ReLU(inplace=True),
            self.conv5,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = self.linear 

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_masks(self, masks):
        # self.conv1.set_mask(masks[0])
        # self.conv2.set_mask(masks[1])
        # self.conv3.set_mask(masks[2])
        # self.conv4.set_mask(masks[3])
        # self.conv5.set_mask(masks[4])
        self.linear.set_mask(masks[5])

def alexnet_pruning(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNetPruning(**kwargs)
    return model
