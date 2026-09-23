import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def criterion(task): 
    return torch.nn.BCEWithLogitsLoss() if task == "binary" else torch.nn.CrossEntropyLoss()

def targets(y,task): 
    return y.float().view(-1, 1) if task == "binary" else y.long().view(-1)

@torch.no_grad()
def evaluate(model, loader, device, task, loss_fn=None):
    loss_fn = loss_fn or criterion(task)
    model.eval()
    ys = []
    ps = []
    probs = []
    loss = 0
    count = 0

    for x, y in loader:
        x = x.to(device)
        target = targets(y, task).to(device)
        logits = model(x)
        loss += loss_fn(logits, target).item() * len(x)
        count += len(x)

        if task == "binary": 
            p = torch.sigmoid(logits).view(-1)
            pred = (p >= 0.5).long()
        else: 
            p = torch.softmax(logits, 1)
            pred = p.argmax(1)
        ys.extend(y.view(-1).tolist())
        ps.extend(pred.cpu().tolist())
        probs.extend(p.cpu().tolist())

    try: 
        auc = float(roc_auc_score(ys,probs, multi_class="ovr", average="macro") if task=="multiclass" else roc_auc_score(ys, probs))
    except ValueError: 
        auc = float("nan")

    if task == "binary":
        f1 = float(f1_score(ys, ps, zero_division=0))
        out = {
            "loss": loss / count,
            "accuracy": float(accuracy_score(ys, ps)),
            "f1_positive": f1,
            "auc": auc,
        }
    else:
        out = {
            "loss": loss / count,
            "accuracy": float(accuracy_score(ys, ps)),
            "f1_macro": float(f1_score(ys, ps, average="macro", zero_division=0)),
            "auc": auc,
        }

    return out
