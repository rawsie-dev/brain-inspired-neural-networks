import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from .adult import DataBundle


def _path(path):
    if path or os.getenv("DRYBEAN_PATH"): 
        return Path(path or os.getenv("DRYBEAN_PATH"))

    import kagglehub
    root = Path(kagglehub.dataset_download("muratkokludataset/dry-bean-dataset"))

    return next(iter(root.rglob("*.xlsx")))

def get_drybean(seed, batch_size=256, data_path=None, num_workers=0):
    path = _path(data_path)

    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")].dropna()

    encoder = LabelEncoder()
    y = encoder.fit_transform(frame.pop("Class").astype(str))

    labels, names = pd.Series(y), tuple(encoder.classes_)
    Xtr, Xtmp, ytr, ytmp = train_test_split(frame, labels, test_size=.2, stratify=labels, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=.5, stratify=ytmp, random_state=seed)

    scaler=StandardScaler().fit(Xtr)

    def ds(x, y): 
        return TensorDataset(torch.tensor(scaler.transform(x), dtype=torch.float32), torch.tensor(y.to_numpy(), dtype=torch.long))
    
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers)

    return DataBundle(
        DataLoader(ds(Xtr, ytr), shuffle=True, generator=g, **kw), 
        DataLoader(ds(Xva, yva), shuffle=False, **kw), 
        DataLoader(ds(Xte, yte), shuffle=False, **kw), 
        frame.shape[1], 
        len(names), 
        "multiclass", 
        names
    )
