import torch
import torch.nn as nn
import torch.nn.functional as F

class ChunkedHeterogeneity(nn.Module):
    """
    Neuronal heterogeneity with fixed contiguous activation groups.

    The final feature dimension is divided into four equal contiguous
    groups, assigned to ReLU, GELU, SiLU, and Mish respectively.

    For ConvNeXtV2 MLP blocks:
        x: [B, H, W, C]
                     ^
                     feature dimension
    """

    def __init__(self):
        super().__init__()

        # Initialized lazily because the feature dimension is
        # determined by the input tensor.
        self.register_buffer("activation_mask", None)

    def _initialize(self, features, device):
        if features % 4 != 0:
            raise ValueError(
                f"Feature dimension ({features}) must be divisible by 4 for chunked heterogeneity."
            )

        quarter = features // 4

        activation_mask = torch.empty(
            features,
            dtype=torch.long,
            device=device,
        )

        # 0 = ReLU
        # 1 = GELU
        # 2 = SiLU
        # 3 = Mish
        activation_mask[:quarter] = 0
        activation_mask[quarter:2 * quarter] = 1
        activation_mask[2 * quarter:3 * quarter] = 2
        activation_mask[3 * quarter:] = 3

        self.activation_mask = activation_mask

    def forward(self, x):
        features = x.shape[-1]

        if self.activation_mask is None:
            self._initialize(features, x.device)

        mask = self.activation_mask.view(
            *([1] * (x.ndim - 1)),
            -1,
        )

        relu = F.relu(x)
        gelu = F.gelu(x)
        silu = F.silu(x)
        mish = F.mish(x)

        return torch.where(
            mask == 0,
            relu,
            torch.where(
                mask == 1,
                gelu,
                torch.where(
                    mask == 2,
                    silu,
                    mish,
                ),
            ),
        )


class RandomHeterogeneity(nn.Module):
    """
    Neuronal heterogeneity with a fixed random activation assignment.

    The final feature dimension is divided into four approximately
    equal groups and assigned to ReLU, GELU, SiLU, and Mish.
    The assignments are randomly shuffled once and remain fixed
    throughout training.


    For ConvNeXtV2 MLP blocks:
        x: [B, H, W, C]
                     ^
                     feature dimension
    """

    def __init__(self):
        super().__init__()

        # Initialized lazily because the feature dimension is
        # determined by the input tensor.
        self.register_buffer("activation_mask", None)

    def _initialize(self, features, device):
        if features % 4 != 0:
            raise ValueError(
                f"Feature dimension ({features}) must be divisible by 4 for chunked heterogeneity."
            )

        quarter = features // 4

        activation_mask = torch.empty(
            features,
            dtype=torch.long,
            device=device,
        )

        # 0 = ReLU
        # 1 = GELU
        # 2 = SiLU
        # 3 = Mish
        perm = torch.randperm(features, device=device)

        quarter = features // 4

        activation_mask[perm[:quarter]] = 0
        activation_mask[perm[quarter:2 * quarter]] = 1
        activation_mask[perm[2 * quarter:3 * quarter]] = 2
        activation_mask[perm[3 * quarter:]] = 3

        self.activation_mask = activation_mask

    def forward(self, x):
        features = x.shape[-1]

        if self.activation_mask is None:
            self._initialize(features, x.device)

        mask = self.activation_mask.view(
            *([1] * (x.ndim - 1)),
            -1,
        )

        relu = F.relu(x)
        gelu = F.gelu(x)
        silu = F.silu(x)
        mish = F.mish(x)

        return torch.where(
            mask == 0,
            relu,
            torch.where(
                mask == 1,
                gelu,
                torch.where(
                    mask == 2,
                    silu,
                    mish,
                ),
            ),
        )