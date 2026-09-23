#!/usr/bin/env python3
"""Run all vision experiments and write a summary."""
import argparse
import subprocess
import sys
from pathlib import Path

from run_tabular import MECHANISMS, command, write_summary


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("deepdetect", "tinyimagenet")

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/vision_results"),
        help="Directory below the repository root for new results.",
    )

    parser.add_argument("--deepdetect-data-path", type=Path)
    parser.add_argument("--tinyimagenet-data-path", type=Path)
    parser.add_argument("--device", help="For example: cpu, mps, or cuda.")
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {args.output_dir}")

    total = len(TASKS) * len(MECHANISMS)
    for number, (task, mechanism) in enumerate(((task, mechanism) for task in TASKS for mechanism in MECHANISMS), start=1):
        print(f"[{number}/{total}] {task} / {mechanism}", flush=True)
        subprocess.run(command(task, mechanism, args), cwd=ROOT, check=True)

    summary = args.output_dir / "summary.csv"

    try:
        write_summary(args.output_dir, summary, tasks=TASKS, expected_seed_count=2)
    except ValueError as error:
        parser.error(str(error))

    print(f"Completed. Summary: {summary}")

if __name__ == "__main__":
    main()
    