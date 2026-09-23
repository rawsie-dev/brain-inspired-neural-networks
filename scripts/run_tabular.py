import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("adult", "drybean")
MECHANISMS = (
    "baseline",
    "multi_act",
    "het_random",
    "het_chunked",
    "kwta_75",
    "kwta_50",
    "kwta_25",
)
METRICS = ("loss", "accuracy", "f1", "auc")

def command(task, mechanism, args):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run.py"),
        "--task", task,
        "--mechanism", mechanism,
        "--output-dir", str(args.output_dir),
    ]
    if args.device:
        cmd.extend(("--device", args.device))

    data_path = getattr(args, f"{task}_data_path", None)

    if data_path:
        cmd.extend(("--data-path", str(data_path)))

    return cmd

def configured_seed_counts(tasks):
    counts = {}

    for task in tasks:
        config = yaml.safe_load((ROOT / "configs" / "tasks" / f"{task}.yaml").read_text()) or {}
        counts[task] = len(config.get("seeds", [42]))

    return counts

def read_results(results_dir, tasks=TASKS):
    rows = []

    for path in results_dir.glob("*/*/seed_*.json"):
        record = json.loads(path.read_text())
        task = record["task"]

        if task not in tasks:
            continue

        metrics = record["metrics"]
        f1_key = "f1_positive" if "f1_positive" in metrics else "f1_macro"

        rows.append(
            {
                "task": task,
                "mechanism": record["mechanism"],
                "seed": record["seed"],
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "f1": metrics[f1_key],
                "auc": metrics["auc"],
            }
        )

    return pd.DataFrame(rows)

def write_summary(results_dir, output, tasks=TASKS, expected_seed_count=10, completed_pairs=None):
    frame = read_results(results_dir, tasks)

    if frame.empty:
        raise ValueError(f"no seed result files found under {results_dir}")

    expected = (
        {(task, mechanism) for task in tasks for mechanism in MECHANISMS}
        if completed_pairs is None
        else set(completed_pairs)
    )

    counts = frame.groupby(["task", "mechanism"]).size()
    missing = expected - set(counts.index)
    incomplete = {}

    for key in counts.index:
        task, _ = key

        expected_count = (
            expected_seed_count[task]
            if isinstance(expected_seed_count, dict)
            else expected_seed_count
        )

        if counts[key] != expected_count:
            incomplete[key] = int(counts[key])

    if missing or incomplete:
        details = []
        if missing:
            details.append(
                "missing: "
                + ", ".join(
                    f"{task}/{mechanism}"
                    for task, mechanism in sorted(missing)
                )
            )
            
        if incomplete:
            details.append(
                "wrong seed count: "
                + ", ".join(
                    f"{task}/{mechanism}={count}"
                    for (task, mechanism), count in sorted(incomplete.items())
                )
            )

        raise ValueError("; ".join(details))

    grouped = frame.groupby(["task", "mechanism"])[list(METRICS)]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n", grouped.size())

    output.parent.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(output, index=False, float_format="%.6f")

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tabular_results"),
        help="Directory below the repository root for new results.",
    )

    parser.add_argument("--adult-data-path", type=Path)
    parser.add_argument("--drybean-data-path", type=Path)
    parser.add_argument("--device", help="For example: cpu, mps, or cuda.")

    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )

    total = len(TASKS) * len(MECHANISMS)
    summary = args.output_dir / "summary.csv"
    completed_pairs = set()
    expected_seed_count = configured_seed_counts(TASKS)
    
    for number, (task, mechanism) in enumerate(((task, mechanism) for task in TASKS for mechanism in MECHANISMS), start=1):
        print(f"[{number}/{total}] {task} / {mechanism}", flush=True)
        subprocess.run(command(task, mechanism, args), cwd=ROOT, check=True)
        completed_pairs.add((task, mechanism))

        try:
            write_summary(
                args.output_dir,
                summary,
                completed_pairs=completed_pairs,
                expected_seed_count=expected_seed_count,
            )
        except ValueError as error:
            parser.error(str(error))

        print(f"Updated summary: {summary}", flush=True)
        
    print(f"Completed. Summary: {summary}")


if __name__ == "__main__":
    main()
