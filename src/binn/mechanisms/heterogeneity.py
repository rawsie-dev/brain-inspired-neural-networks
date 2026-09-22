import torch
from torch import nn
from torch.nn import functional as F


class ChunkedHeterogeneity(nn.Module):
    """
    Chunked heterogeneity means the four activation types are assigned 
    continuously to features.

    For example, if there are 32 features, the first 8 features are assigned
    relu, the next 8 are gelu, the next 8 are silu, and the last 8 are mish
    """
    def __init__(self, features):
        super().__init__()
        if features % 4:
            raise ValueError("features must be divisible by 4 for equal activation groups")
        
        assignment = torch.arange(4).repeat_interleave(features // 4)
        self.register_buffer("activation_assignment", assignment)

    def forward(self, x):
        if x.shape[-1] != self.activation_assignment.numel():
            raise ValueError("input feature dimension differs from activation assignment")
        
        assignment = self.activation_assignment.view(*([1] * (x.ndim - 1)), -1)

        values = (F.relu(x), F.gelu(x), F.silu(x), F.mish(x))
        output = values[0]
        
        for index, value in enumerate(values[1:], 1):
            output = torch.where(assignment == index, value, output)

        return output


class RandomHeterogeneity(nn.Module):
    """
    Random heterogeneity means the four activation types are distributed
    evenly randomly across features.

    For example, if there are 8 features, a possible assignment is 
    [silu, relu, gelu, mish, mish, silu, relu, gelu]
    """
    def __init__(self, features, seed=None):
        super().__init__()
        if features % 4:
            raise ValueError("features must be divisible by 4 for equal activation groups")
        
        assignment = torch.arange(4).repeat_interleave(features // 4)
        generator = torch.Generator()

        if seed is not None:
            generator.manual_seed(seed)

        assignment = assignment[torch.randperm(features, generator=generator)]
        self.register_buffer("activation_assignment", assignment)

    def forward(self, x):
        if x.shape[-1] != self.activation_assignment.numel():
            raise ValueError("input feature dimension differs from activation assignment")
        
        assignment = self.activation_assignment.view(*([1] * (x.ndim - 1)), -1)
        
        values = (F.relu(x), F.gelu(x), F.silu(x), F.mish(x))
        output = values[0]

        for index, value in enumerate(values[1:], 1):
            output = torch.where(assignment == index, value, output)

        return output
    