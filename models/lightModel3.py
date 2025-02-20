import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


"""LIGHT CNN MODEL"""
class DeePromoterModel(nn.Module):
    def __init__(self):
        super(DeePromoterModel, self).__init__()

        self.conv1 = nn.Conv1d(4, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm1d(512)

        # Calcola la dimensione dell'input per il layer FC
        self._to_linear = None
        self._get_conv_output_size((4, 600))

        # FC
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 3)

    def _get_conv_output_size(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self._forward_conv(input)
        self._to_linear = int(np.prod(output.size()[1:]))
        print(self._to_linear)

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers con ReLU e dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x