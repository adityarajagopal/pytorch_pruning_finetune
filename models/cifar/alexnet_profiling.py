'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import sys
import time

__all__ = ['alexnet_profiling']

class AlexNetProfiling(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetProfiling, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        times = [] 
        
        t0 = time.time()
        x = self.conv1(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.relu(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.maxpool2d(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.conv2(x) 
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.relu(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.maxpool2d(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.conv3(x) 
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.relu(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.conv4(x) 
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.relu(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.conv5(x) 
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.relu(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.maxpool2d(x)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = x.view(x.size(0), -1)
        times.append(time.time() - t0)
        
        t0 = time.time()
        x = self.classifier(x)
        times.append(time.time() - t0)
        
        return x, times


def alexnet_profiling(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNetProfiling(**kwargs)
    return model
