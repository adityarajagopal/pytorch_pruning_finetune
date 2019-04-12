'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['googlenet']

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        
        # pre_layers
        self.conv0 = nn.Conv2d(3, 192, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(192) 
        self.relu0 = nn.ReLU(True)
        
        # inception a3
        in_planes = 192
        n1x1 = 64
        n3x3red = 96
        n3x3 = 128
        n5x5red = 16
        n5x5 = 32
        pool_planes = 32
        
        self.a3_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.a3_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.a3_b1_relu0 = nn.ReLU(True)

        self.a3_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.a3_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.a3_b2_relu0 = nn.ReLU(True)
        self.a3_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.a3_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.a3_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.a3_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.a3_b3_relu0 = nn.ReLU(True)
        self.a3_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.a3_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.a3_b3_relu1 = nn.ReLU(True)
        self.a3_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.a3_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.a3_b3_relu2 = nn.ReLU(True)
            
        self.a3_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.a3_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.a3_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.a3_b4_relu0 = nn.ReLU(True)

        # inception b3
        in_planes = 256
        n1x1 = 128
        n3x3red = 128
        n3x3 = 192
        n5x5red = 32
        n5x5 = 96
        pool_planes = 64
        
        self.b3_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.b3_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.b3_b1_relu0 = nn.ReLU(True)

        self.b3_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.b3_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.b3_b2_relu0 = nn.ReLU(True)
        self.b3_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.b3_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.b3_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.b3_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.b3_b3_relu0 = nn.ReLU(True)
        self.b3_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.b3_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.b3_b3_relu1 = nn.ReLU(True)
        self.b3_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.b3_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.b3_b3_relu2 = nn.ReLU(True)
            
        self.b3_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.b3_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.b3_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.b3_b4_relu0 = nn.ReLU(True)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # inception a4
        in_planes = 480
        n1x1 = 192
        n3x3red = 96
        n3x3 = 208
        n5x5red = 16
        n5x5 = 48
        pool_planes = 64
        
        self.a4_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.a4_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.a4_b1_relu0 = nn.ReLU(True)

        self.a4_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.a4_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.a4_b2_relu0 = nn.ReLU(True)
        self.a4_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.a4_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.a4_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.a4_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.a4_b3_relu0 = nn.ReLU(True)
        self.a4_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.a4_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.a4_b3_relu1 = nn.ReLU(True)
        self.a4_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.a4_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.a4_b3_relu2 = nn.ReLU(True)
            
        self.a4_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.a4_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.a4_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.a4_b4_relu0 = nn.ReLU(True)
        
        # inception b4
        in_planes = 512
        n1x1 = 160
        n3x3red = 112
        n3x3 = 224
        n5x5red = 24
        n5x5 = 64
        pool_planes = 64
        
        self.b4_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.b4_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.b4_b1_relu0 = nn.ReLU(True)

        self.b4_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.b4_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.b4_b2_relu0 = nn.ReLU(True)
        self.b4_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.b4_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.b4_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.b4_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.b4_b3_relu0 = nn.ReLU(True)
        self.b4_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.b4_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.b4_b3_relu1 = nn.ReLU(True)
        self.b4_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.b4_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.b4_b3_relu2 = nn.ReLU(True)
            
        self.b4_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.b4_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.b4_b4_relu0 = nn.ReLU(True)

        # inception c4
        in_planes = 512
        n1x1 = 128
        n3x3red = 128
        n3x3 = 256
        n5x5red = 24
        n5x5 = 64
        pool_planes = 64
        
        self.c4_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.c4_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.c4_b1_relu0 = nn.ReLU(True)

        self.c4_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.c4_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.c4_b2_relu0 = nn.ReLU(True)
        self.c4_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.c4_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.c4_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.c4_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.c4_b3_relu0 = nn.ReLU(True)
        self.c4_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.c4_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.c4_b3_relu1 = nn.ReLU(True)
        self.c4_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.c4_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.c4_b3_relu2 = nn.ReLU(True)
            
        self.c4_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.c4_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.c4_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.c4_b4_relu0 = nn.ReLU(True)

        # inception d4
        in_planes = 512
        n1x1 = 112
        n3x3red = 144
        n3x3 = 288
        n5x5red = 32
        n5x5 = 64
        pool_planes = 64
        
        self.d4_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.d4_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.d4_b1_relu0 = nn.ReLU(True)

        self.d4_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.d4_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.d4_b2_relu0 = nn.ReLU(True)
        self.d4_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.d4_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.d4_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.d4_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.d4_b3_relu0 = nn.ReLU(True)
        self.d4_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.d4_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.d4_b3_relu1 = nn.ReLU(True)
        self.d4_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.d4_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.d4_b3_relu2 = nn.ReLU(True)
            
        self.d4_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.d4_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.d4_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.d4_b4_relu0 = nn.ReLU(True)
        
        # inception e4
        in_planes = 528
        n1x1 = 256
        n3x3red = 160
        n3x3 = 320
        n5x5red = 32
        n5x5 = 128
        pool_planes = 128
        
        self.e4_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.e4_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.e4_b1_relu0 = nn.ReLU(True)

        self.e4_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.e4_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.e4_b2_relu0 = nn.ReLU(True)
        self.e4_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.e4_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.e4_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.e4_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.e4_b3_relu0 = nn.ReLU(True)
        self.e4_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.e4_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.e4_b3_relu1 = nn.ReLU(True)
        self.e4_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.e4_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.e4_b3_relu2 = nn.ReLU(True)
            
        self.e4_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.e4_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.e4_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.e4_b4_relu0 = nn.ReLU(True)
        
        # inception a5
        in_planes = 832 
        n1x1 = 256
        n3x3red = 160
        n3x3 = 320
        n5x5red = 32
        n5x5 = 128
        pool_planes = 128
        
        self.a5_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.a5_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.a5_b1_relu0 = nn.ReLU(True)

        self.a5_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.a5_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.a5_b2_relu0 = nn.ReLU(True)
        self.a5_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.a5_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.a5_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.a5_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.a5_b3_relu0 = nn.ReLU(True)
        self.a5_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.a5_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.a5_b3_relu1 = nn.ReLU(True)
        self.a5_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.a5_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.a5_b3_relu2 = nn.ReLU(True)
            
        self.a5_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.a5_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.a5_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.a5_b4_relu0 = nn.ReLU(True)
        
        # inception b5
        in_planes = 832 
        n1x1 = 384
        n3x3red = 192
        n3x3 = 384
        n5x5red = 48
        n5x5 = 128
        pool_planes = 128
        
        self.b5_b1_conv0 = nn.Conv2d(in_planes, n1x1, kernel_size=1)  
        self.b5_b1_bn0 = nn.BatchNorm2d(n1x1)
        self.b5_b1_relu0 = nn.ReLU(True)

        self.b5_b2_conv0 = nn.Conv2d(in_planes, n3x3red, kernel_size=1)
        self.b5_b2_bn0 = nn.BatchNorm2d(n3x3red) 
        self.b5_b2_relu0 = nn.ReLU(True)
        self.b5_b2_conv1 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1) 
        self.b5_b2_bn1 = nn.BatchNorm2d(n3x3) 
            
        self.b5_b3_conv0 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
        self.b5_b3_bn0 = nn.BatchNorm2d(n5x5red)
        self.b5_b3_relu0 = nn.ReLU(True)
        self.b5_b3_conv1 = nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.b5_b3_bn1 = nn.BatchNorm2d(n5x5)
        self.b5_b3_relu1 = nn.ReLU(True)
        self.b5_b3_conv2 = nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
        self.b5_b3_bn2 = nn.BatchNorm2d(n5x5)
        self.b5_b3_relu2 = nn.ReLU(True)
            
        self.b5_b4_maxpool0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.b5_b4_conv0 = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
        self.b5_b4_bn0 = nn.BatchNorm2d(pool_planes)
        self.b5_b4_relu0 = nn.ReLU(True)
        
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        # pre_layers
        out = self.conv0(x)
        out = self.bn0(out) 
        out = self.relu0(out)

        # a3
        out_1 = self.a3_b1_conv0(out)
        out_1 = self.a3_b1_bn0(out_1)
        out_1 = self.a3_b1_relu0(out_1)
        
        out_2 = self.a3_b2_conv0(out)
        out_2 = self.a3_b2_bn0(out_2)
        out_2 = self.a3_b2_relu0(out_2)
        out_2 = self.a3_b2_conv1(out_2)
        out_2 = self.a3_b2_bn1(out_2)

        out_3 = self.a3_b3_conv0(out)
        out_3 = self.a3_b3_bn0(out_3)
        out_3 = self.a3_b3_relu0(out_3)
        out_3 = self.a3_b3_conv1(out_3)
        out_3 = self.a3_b3_bn1(out_3)
        out_3 = self.a3_b3_relu1(out_3)
        out_3 = self.a3_b3_conv2(out_3)
        out_3 = self.a3_b3_bn2(out_3)
        out_3 = self.a3_b3_relu2(out_3)
            
        out_4 = self.a3_b4_maxpool0(out)
        out_4 = self.a3_b4_conv0(out_4)
        out_4 = self.a3_b4_bn0(out_4)
        out_4 = self.a3_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)

        # b3
        out_1 = self.b3_b1_conv0(out)
        out_1 = self.b3_b1_bn0(out_1)
        out_1 = self.b3_b1_relu0(out_1)
        
        out_2 = self.b3_b2_conv0(out)
        out_2 = self.b3_b2_bn0(out_2)
        out_2 = self.b3_b2_relu0(out_2)
        out_2 = self.b3_b2_conv1(out_2)
        out_2 = self.b3_b2_bn1(out_2)

        out_3 = self.b3_b3_conv0(out)
        out_3 = self.b3_b3_bn0(out_3)
        out_3 = self.b3_b3_relu0(out_3)
        out_3 = self.b3_b3_conv1(out_3)
        out_3 = self.b3_b3_bn1(out_3)
        out_3 = self.b3_b3_relu1(out_3)
        out_3 = self.b3_b3_conv2(out_3)
        out_3 = self.b3_b3_bn2(out_3)
        out_3 = self.b3_b3_relu2(out_3)
            
        out_4 = self.b3_b4_maxpool0(out)
        out_4 = self.b3_b4_conv0(out_4)
        out_4 = self.b3_b4_bn0(out_4)
        out_4 = self.b3_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)

        out = self.maxpool(out)
        
        # a4
        out_1 = self.a4_b1_conv0(out)
        out_1 = self.a4_b1_bn0(out_1)
        out_1 = self.a4_b1_relu0(out_1)
        
        out_2 = self.a4_b2_conv0(out)
        out_2 = self.a4_b2_bn0(out_2)
        out_2 = self.a4_b2_relu0(out_2)
        out_2 = self.a4_b2_conv1(out_2)
        out_2 = self.a4_b2_bn1(out_2)

        out_3 = self.a4_b3_conv0(out)
        out_3 = self.a4_b3_bn0(out_3)
        out_3 = self.a4_b3_relu0(out_3)
        out_3 = self.a4_b3_conv1(out_3)
        out_3 = self.a4_b3_bn1(out_3)
        out_3 = self.a4_b3_relu1(out_3)
        out_3 = self.a4_b3_conv2(out_3)
        out_3 = self.a4_b3_bn2(out_3)
        out_3 = self.a4_b3_relu2(out_3)
            
        out_4 = self.a4_b4_maxpool0(out)
        out_4 = self.a4_b4_conv0(out_4)
        out_4 = self.a4_b4_bn0(out_4)
        out_4 = self.a4_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        
        # b4
        out_1 = self.b4_b1_conv0(out)
        out_1 = self.b4_b1_bn0(out_1)
        out_1 = self.b4_b1_relu0(out_1)
        
        out_2 = self.b4_b2_conv0(out)
        out_2 = self.b4_b2_bn0(out_2)
        out_2 = self.b4_b2_relu0(out_2)
        out_2 = self.b4_b2_conv1(out_2)
        out_2 = self.b4_b2_bn1(out_2)

        out_3 = self.b4_b3_conv0(out)
        out_3 = self.b4_b3_bn0(out_3)
        out_3 = self.b4_b3_relu0(out_3)
        out_3 = self.b4_b3_conv1(out_3)
        out_3 = self.b4_b3_bn1(out_3)
        out_3 = self.b4_b3_relu1(out_3)
        out_3 = self.b4_b3_conv2(out_3)
        out_3 = self.b4_b3_bn2(out_3)
        out_3 = self.b4_b3_relu2(out_3)
            
        out_4 = self.b4_b4_maxpool0(out)
        out_4 = self.b4_b4_conv0(out_4)
        out_4 = self.b4_b4_bn0(out_4)
        out_4 = self.b4_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        
        # c4
        out_1 = self.c4_b1_conv0(out)
        out_1 = self.c4_b1_bn0(out_1)
        out_1 = self.c4_b1_relu0(out_1)
        
        out_2 = self.c4_b2_conv0(out)
        out_2 = self.c4_b2_bn0(out_2)
        out_2 = self.c4_b2_relu0(out_2)
        out_2 = self.c4_b2_conv1(out_2)
        out_2 = self.c4_b2_bn1(out_2)

        out_3 = self.c4_b3_conv0(out)
        out_3 = self.c4_b3_bn0(out_3)
        out_3 = self.c4_b3_relu0(out_3)
        out_3 = self.c4_b3_conv1(out_3)
        out_3 = self.c4_b3_bn1(out_3)
        out_3 = self.c4_b3_relu1(out_3)
        out_3 = self.c4_b3_conv2(out_3)
        out_3 = self.c4_b3_bn2(out_3)
        out_3 = self.c4_b3_relu2(out_3)
            
        out_4 = self.c4_b4_maxpool0(out)
        out_4 = self.c4_b4_conv0(out_4)
        out_4 = self.c4_b4_bn0(out_4)
        out_4 = self.c4_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        
        # d4
        out_1 = self.d4_b1_conv0(out)
        out_1 = self.d4_b1_bn0(out_1)
        out_1 = self.d4_b1_relu0(out_1)
        
        out_2 = self.d4_b2_conv0(out)
        out_2 = self.d4_b2_bn0(out_2)
        out_2 = self.d4_b2_relu0(out_2)
        out_2 = self.d4_b2_conv1(out_2)
        out_2 = self.d4_b2_bn1(out_2)

        out_3 = self.d4_b3_conv0(out)
        out_3 = self.d4_b3_bn0(out_3)
        out_3 = self.d4_b3_relu0(out_3)
        out_3 = self.d4_b3_conv1(out_3)
        out_3 = self.d4_b3_bn1(out_3)
        out_3 = self.d4_b3_relu1(out_3)
        out_3 = self.d4_b3_conv2(out_3)
        out_3 = self.d4_b3_bn2(out_3)
        out_3 = self.d4_b3_relu2(out_3)
            
        out_4 = self.d4_b4_maxpool0(out)
        out_4 = self.d4_b4_conv0(out_4)
        out_4 = self.d4_b4_bn0(out_4)
        out_4 = self.d4_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        
        # e4
        out_1 = self.e4_b1_conv0(out)
        out_1 = self.e4_b1_bn0(out_1)
        out_1 = self.e4_b1_relu0(out_1)
        
        out_2 = self.e4_b2_conv0(out)
        out_2 = self.e4_b2_bn0(out_2)
        out_2 = self.e4_b2_relu0(out_2)
        out_2 = self.e4_b2_conv1(out_2)
        out_2 = self.e4_b2_bn1(out_2)

        out_3 = self.e4_b3_conv0(out)
        out_3 = self.e4_b3_bn0(out_3)
        out_3 = self.e4_b3_relu0(out_3)
        out_3 = self.e4_b3_conv1(out_3)
        out_3 = self.e4_b3_bn1(out_3)
        out_3 = self.e4_b3_relu1(out_3)
        out_3 = self.e4_b3_conv2(out_3)
        out_3 = self.e4_b3_bn2(out_3)
        out_3 = self.e4_b3_relu2(out_3)
            
        out_4 = self.e4_b4_maxpool0(out)
        out_4 = self.e4_b4_conv0(out_4)
        out_4 = self.e4_b4_bn0(out_4)
        out_4 = self.e4_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        
        out = self.maxpool(out)
        
        # a5
        out_1 = self.a5_b1_conv0(out)
        out_1 = self.a5_b1_bn0(out_1)
        out_1 = self.a5_b1_relu0(out_1)
        
        out_2 = self.a5_b2_conv0(out)
        out_2 = self.a5_b2_bn0(out_2)
        out_2 = self.a5_b2_relu0(out_2)
        out_2 = self.a5_b2_conv1(out_2)
        out_2 = self.a5_b2_bn1(out_2)

        out_3 = self.a5_b3_conv0(out)
        out_3 = self.a5_b3_bn0(out_3)
        out_3 = self.a5_b3_relu0(out_3)
        out_3 = self.a5_b3_conv1(out_3)
        out_3 = self.a5_b3_bn1(out_3)
        out_3 = self.a5_b3_relu1(out_3)
        out_3 = self.a5_b3_conv2(out_3)
        out_3 = self.a5_b3_bn2(out_3)
        out_3 = self.a5_b3_relu2(out_3)
            
        out_4 = self.a5_b4_maxpool0(out)
        out_4 = self.a5_b4_conv0(out_4)
        out_4 = self.a5_b4_bn0(out_4)
        out_4 = self.a5_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        
        # b5
        out_1 = self.b5_b1_conv0(out)
        out_1 = self.b5_b1_bn0(out_1)
        out_1 = self.b5_b1_relu0(out_1)
        
        out_2 = self.b5_b2_conv0(out)
        out_2 = self.b5_b2_bn0(out_2)
        out_2 = self.b5_b2_relu0(out_2)
        out_2 = self.b5_b2_conv1(out_2)
        out_2 = self.b5_b2_bn1(out_2)

        out_3 = self.b5_b3_conv0(out)
        out_3 = self.b5_b3_bn0(out_3)
        out_3 = self.b5_b3_relu0(out_3)
        out_3 = self.b5_b3_conv1(out_3)
        out_3 = self.b5_b3_bn1(out_3)
        out_3 = self.b5_b3_relu1(out_3)
        out_3 = self.b5_b3_conv2(out_3)
        out_3 = self.b5_b3_bn2(out_3)
        out_3 = self.b5_b3_relu2(out_3)
            
        out_4 = self.b5_b4_maxpool0(out)
        out_4 = self.b5_b4_conv0(out_4)
        out_4 = self.b5_b4_bn0(out_4)
        out_4 = self.b5_b4_relu0(out_4)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

def googlenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return GoogLeNet(**kwargs)
