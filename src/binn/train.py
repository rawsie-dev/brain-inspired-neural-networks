import copy
from time import perf_counter
from tqdm.auto import tqdm
from .evaluate import criterion, evaluate, targets


def train(model, train_loader, val_loader, optimizer, device, task_type, max_epochs=100, patience=5, verbose=True, progress_every=0, progress_bar=False):
    model.to(device)
    loss_fn = criterion(task_type)
    best = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    bad = 0
    history = {"train_loss": [], "val_loss": [], "epoch_time_sec": []}

    for epoch in range(1, max_epochs + 1):
        started = perf_counter()
        model.train()
        total = n = 0
        batches = tqdm(train_loader, desc=f"Train {epoch:03d}", leave=False) if progress_bar else train_loader

        for batch_index, (x, y) in enumerate(batches, start=1):
            x = x.to(device)
            y = targets(y, task_type).to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            total += loss.item()*len(x)
            n += len(x)

            if progress_bar: 
                batches.set_postfix(loss=f"{loss.item():.4f}")

            if progress_every and batch_index % progress_every == 0:
                print(f"epoch {epoch:03d}: batch {batch_index}/{len(train_loader)}", flush=True)

        val = evaluate(model, val_loader, device, task_type, loss_fn)
        history["train_loss"].append(total/n)
        history["val_loss"].append(val["loss"])
        history["epoch_time_sec"].append(perf_counter() - started)

        if verbose: 
            print(f"epoch {epoch:03d}: train loss={total/n:.4f}, val loss={val['loss']:.4f}, val acc={val['accuracy']:.4f}")

        if val["loss"] < best_loss: 
            best_loss = val["loss"]
            best = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad = 0
        else: 
            bad += 1

        if bad >= patience: 
            break

    model.load_state_dict(best)

    history.update(
        best_epoch=best_epoch, 
        best_val_loss=best_loss, 
        epochs=len(history["val_loss"]), 
        total_time_sec=sum(history["epoch_time_sec"])
    )

    return model, history
