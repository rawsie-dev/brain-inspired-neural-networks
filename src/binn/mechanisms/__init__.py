from torch import nn
from .heterogeneity import ChunkedHeterogeneity, RandomHeterogeneity
from .kwta import GELUKWTA, KWTA
from .multi_activation import MultiActivation

def build_activation(name, *, features=None, seed=None, keep_ratio=None, **_):
    """Create an activation module; ``features`` is required by heterogeneity."""
    name = name.lower().replace("-", "_")

    if name in {"baseline", "gelu"}:
        return nn.GELU()
    
    if name in {"relu", "silu", "mish"}:
        return {"relu": nn.ReLU, "silu": nn.SiLU, "mish": nn.Mish}[name]()
    
    if name in {"multi_act", "multi_activation"}:
        return MultiActivation()
    
    if name in {"het_random", "heterogeneity_random"}:
        return RandomHeterogeneity(features, seed=seed)
    
    if name in {"het_chunked", "heterogeneity_chunked"}:
        return ChunkedHeterogeneity(features)
    
    if name in {"kwta", "gelu_kwta"}:
        return GELUKWTA(0.5 if keep_ratio is None else keep_ratio)
    
    raise ValueError(f"Unknown activation mechanism: {name}")

__all__ = ["build_activation", "MultiActivation", "RandomHeterogeneity", "ChunkedHeterogeneity", "KWTA", "GELUKWTA"]
