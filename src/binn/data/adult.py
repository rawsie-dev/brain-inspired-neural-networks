import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


NUM_COLS = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country"]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    num_outputs: int
    task_type: str
    class_names: tuple


def _load(path=None):
    path = path or os.getenv("ADULT_CSV")

    if not path:
        import kagglehub
        path = os.path.join(kagglehub.dataset_download("wenruliu/adult-income-dataset"), "adult.csv")

    frame = pd.read_csv(path).dropna()
    labels = frame.pop("income").str.strip().str.replace(".", "", regex=False).eq(">50K").astype(np.int64)

    return frame, labels

def get_adult(seed, batch_size=256, data_path=None, num_workers=0):
    X, y = _load(data_path)

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=.5, stratify=ytmp, random_state=seed)

    scaler = StandardScaler().fit(Xtr[NUM_COLS])

    try: 
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Xtr[CAT_COLS])
    except TypeError: 
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(Xtr[CAT_COLS])

    def tensorize(frame, labels):
        features = np.concatenate((scaler.transform(frame[NUM_COLS]), encoder.transform(frame[CAT_COLS])), axis=1)
        return TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels.to_numpy(), dtype=torch.long))
    
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers)

    return DataBundle(
        DataLoader(tensorize(Xtr,ytr), shuffle=True, generator=generator, **kwargs), 
        DataLoader(tensorize(Xva,yva), shuffle=False, **kwargs), 
        DataLoader(tensorize(Xte,yte), shuffle=False, **kwargs), 
        len(NUM_COLS) + sum(len(x) for x in encoder.categories_), 
        1, 
        "binary", 
        ("<=50K", ">50K")
    )
