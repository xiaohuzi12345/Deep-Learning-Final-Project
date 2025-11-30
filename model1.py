import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gc

class myCNN(nn.Module):
    def __init__(self, input_channels):
        super(myCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 1)  # output 1 logit

    def forward(self, x):
        # [n, dim, dim, ch] → [n, ch, dim, dim]
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # output shape [n, 1]