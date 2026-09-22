import torch
from torch import nn


class KWTA(nn.Module):
    def __init__(self, keep_ratio=0.5):
        super().__init__()

        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        
        self.keep_ratio = float(keep_ratio)

    def forward(self, x):
        k = max(1, int(x.shape[-1] * self.keep_ratio))
        indices = x.topk(k, dim=-1, sorted=False).indices
        mask = torch.zeros_like(x, dtype=torch.bool).scatter_(-1, indices, True)
        
        return torch.where(mask, x, torch.zeros_like(x))


class GELUKWTA(nn.Sequential):
    def __init__(self, keep_ratio=0.5):
        super().__init__(nn.GELU(), KWTA(keep_ratio))
