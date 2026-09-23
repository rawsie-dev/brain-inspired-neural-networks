import argparse, json, sys
from pathlib import Path
import yaml, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from binn.mechanisms import build_activation
from binn.data import DATASETS
from binn.evaluate import evaluate
from binn.models import MLP, build_convnextv2
from binn.train import train
from binn.utils.seed import seed_everything


def load(path):
    with open(path) as f: return yaml.safe_load(f) or {}

def default_device():
    if torch.cuda.is_available(): 
        return "cuda"
    
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): 
        return "mps"
    
    return "cpu"

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--task",required=True)
    p.add_argument("--mechanism",required=True)
    p.add_argument("--seeds",nargs="*",type=int)
    p.add_argument("--data-path")
    p.add_argument("--output-dir",default="results")
    p.add_argument("--device")

    args = p.parse_args()

    task_name = {"tiny_imagenet": "tinyimagenet"}.get(args.task, args.task)

    task = load(ROOT / "configs/tasks" / f"{args.task}.yaml")
    mechanism = load(ROOT / "configs/mechanisms" / f"{args.mechanism}.yaml")
    seeds = args.seeds or task.get("seeds", [42])

    device = torch.device(args.device or default_device())

    print(f"Task={args.task}, mechanism={args.mechanism}, device={device}", flush=True)
    if task["model"] == "convnextv2" and device.type == "cpu":
        print("Warning: ConvNeXtV2-Base on CPU is very slow. Use CUDA or Apple MPS if available.", flush=True)

    for seed in seeds:
        print(f"Seed {seed}: preparing data (this may download the dataset on first use)...", flush=True)
        seed_everything(seed)

        if task_name in {"deepdetect","tinyimagenet"}:
            data = DATASETS[task_name](
                seed, 
                batch_size=task.get("batch_size", 256),
                data_path=args.data_path,
                image_size=task.get("image_size", 224),
                num_workers=task.get("num_workers", 0)
            )
        else: 
            data = DATASETS[task_name](
                seed,
                batch_size=task.get("batch_size", 256),
                data_path=args.data_path,
                num_workers=task.get("num_workers", 0)
            )

        print(f"Seed {seed}: data ready; train batches={len(data.train_loader)}, validation batches={len(data.val_loader)}.", flush=True)
        
        activation_index = 0
        def factory(width):
            nonlocal activation_index
            layer_seed = seed + activation_index
            activation_index += 1
            return build_activation(mechanism["name"], 
                                    features=width,
                                    seed=layer_seed,
                                    keep_ratio=mechanism.get("keep_ratio")
                                )
        print(f"Seed {seed}: building model...", flush=True)

        if task["model"] == "mlp":
            model = MLP(data.input_dim, data.num_outputs, factory, dropout=task.get("dropout", .2))
        else:
            model = build_convnextv2(data.num_outputs, factory, task.get("variant", "convnextv2_base"))

        print(f"Seed {seed}: training...", flush=True)

        optimizer_cls = (
            torch.optim.AdamW
            if task["model"] == "convnextv2"
            else torch.optim.Adam
        )

        optimizer = optimizer_cls(
            model.parameters(),
            lr=task.get("learning_rate",.001),
            weight_decay=task.get("weight_decay",0)
        ); 
        model, history = train(
            model,
            data.train_loader,
            data.val_loader,
            optimizer,
            device,
            data.task_type,
            max_epochs=task.get("max_epochs", 100),
            patience=task.get("patience", 5),
            progress_every=0,
            progress_bar=True
        ); 
        metrics = evaluate(
            model,
            data.test_loader,
            device,
            data.task_type
        )

        record = {
            "task": args.task,
            "mechanism": args.mechanism,
            "seed": seed,
            "metrics": metrics,
            "history": history,
            "device": str(device)
        }

        dest = ROOT / args.output_dir / args.task / args.mechanism
        dest.mkdir(parents=True, exist_ok=True)
        (dest/f"seed_{seed}.json").write_text(json.dumps(record,indent=2,allow_nan=True))
        print(json.dumps(metrics,indent=2))

if __name__ == "__main__": 
    main()
