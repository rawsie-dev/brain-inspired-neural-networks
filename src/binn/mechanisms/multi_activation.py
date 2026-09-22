import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        tanh_x, sigmoid_x = torch.tanh(x), torch.sigmoid(x)
        gaussian = 1.0 / (x.square() + 1.0)
        snake = x + torch.sin(2.0 * math.pi * x).square() / (2.0 * math.pi)

        return (
            0.25 * F.relu(x) + 
            x * tanh_x + 
            tanh_x + 
            sigmoid_x + 
            torch.sin(x) + 
            gaussian + 
            snake + 
            sigmoid_x * torch.cos(x) + 
            torch.erf(x) + 
            x * sigmoid_x
        ) / 10.0
