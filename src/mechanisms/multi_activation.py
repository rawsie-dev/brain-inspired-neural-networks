import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(x):
    return 1.0 / (x.square() + 1.0)


def snake(x):
    return x + torch.sin(2.0 * math.pi * x).square() / (2.0 * math.pi)


class MultiActivation(nn.Module):
    """
    Multi-Activation Neuron.

    Each feature is transformed by the same fixed combination
    of ten activation functions. No additional trainable
    parameters are introduced.

    The ReLU term has weight 0.25; all other terms have
    weight 1.0. The resulting sum is divided by 10.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        tanh_x = torch.tanh(x)
        sigmoid_x = torch.sigmoid(x)

        return (
            0.25 * F.relu(x)
            + x * tanh_x
            + tanh_x
            + sigmoid_x
            + torch.sin(x)
            + gaussian(x)
            + snake(x)
            + sigmoid_x * torch.cos(x)
            + torch.erf(x)
            + x * sigmoid_x
        ) / 10.0
