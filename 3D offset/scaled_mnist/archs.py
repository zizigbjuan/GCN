# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from deform_conv_v2 import *


class ScaledMNISTNet(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool3d((3,3,3))
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        features = []
        inplanes = 1
        outplanes = 1
        self.DeformConv3d = DeformConv3d(inplanes, outplanes, 3, padding=1, bias=False, modulation=args.modulation)
        self.BatchNorm3d = nn.BatchNorm3d(outplanes)
        self.relu
           
        inplanes = 1
        outplanes = 32
        for i in range(2):
            features.append(nn.Conv3d(inplanes, outplanes, 3, padding=1, bias=False)) 
            features.append(nn.BatchNorm3d(outplanes))
            features.append(self.relu)
            if i == 1:
                features.append(self.pool)
            inplanes = outplanes
            outplanes *= 2
        self.features = nn.Sequential(*features)
        self.fc = nn.Linear(64, 10)

    def forward(self, input):
        
        for i in range(2):
            x,p = self.DeformConv3d(input)
            xxx = x
            x = self.BatchNorm3d(x)
            out = self.relu(x)
            input = out            
        out = input
        x = self.features(input)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        output = self.fc(x)

        return output,xxx,p
