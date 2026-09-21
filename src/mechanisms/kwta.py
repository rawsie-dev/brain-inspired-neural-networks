import torch
import torch.nn as nn

class KWTA(nn.Module):
    """Apply k-winners-take-all sparsification along the final dimension.

    For each vector in x, this module retains the largest
    max(1, int(C * keep_ratio)) values, where C = x.shape[-1], and
    replaces every other value with zero. The operation supports inputs with
    any number of leading dimensions and preserves their shape and dtype.

    Args:
        keep_ratio: Fraction of features in the final dimension to retain.
            Must be in (0, 1].
    """
    def __init__(self, keep_ratio=0.5):
        super().__init__()

        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1].")

        self.keep_ratio = keep_ratio

        self.debug = False
        self.active_sum = 0
        self.total_sum = 0

    def forward(self, x):
        C = x.shape[-1]
        k = max(1, int(C * self.keep_ratio))

        _, indices = torch.topk(
            x,
            k=k,
            dim=-1,
            largest=True,
            sorted=False,
        )

        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, indices, True)

        y = torch.where(
            mask,
            x,
            torch.zeros_like(x),
        )

        if self.debug:
            self.active_sum += mask.sum().item()
            self.total_sum += mask.numel()

        return y

    def reset_stats(self):
        self.active_sum = 0
        self.total_sum = 0

    def active_ratio(self):
        if self.total_sum == 0:
            return 0.0

        return self.active_sum / self.total_sum


class GELUKWTA(nn.Module):
    def __init__(self, keep_ratio=0.5):
        super().__init__()

        self.gelu = nn.GELU()
        self.kwta = KWTA(keep_ratio)

    def forward(self, x):
        x = self.gelu(x)
        x = self.kwta(x)
        return x
