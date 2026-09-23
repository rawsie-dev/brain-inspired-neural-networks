import argparse
import pandas as pd
p = argparse.ArgumentParser()
p.add_argument("summary", nargs="?", default="results/summary.csv")
a = p.parse_args()

frame = pd.read_csv(a.summary, index_col=[0, 1], header=[0, 1])

print(frame.to_latex(float_format="%.4f", na_rep="--"))
