#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()

p.add_argument("results", nargs="?", default="results")
p.add_argument("--output", default="results/summary.csv")
a = p.parse_args()

rows = []
for path in Path(a.results).glob("*/*/seed_*.json"):
    result = json.loads(path.read_text())
    rows.append({"task":result["task"], "mechanism":result["mechanism"], **result["metrics"]})

frame = pd.DataFrame(rows)
numeric = frame.select_dtypes("number").columns
out = frame.groupby(["task", "mechanism"])[numeric].agg(["mean", "std"])

out.to_csv(a.output)
print(out.to_string())
