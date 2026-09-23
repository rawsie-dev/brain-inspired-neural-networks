# Brain-Inspired Neural Networks

Reproducible experiments for multi-activation neurons, fixed heterogeneous
activation assignments, and GELU + k-winners-take-all (kWTA). The shared
runner supports Adult Income and Dry Bean tabular MLPs plus DeepDetect and
Tiny ImageNet ConvNeXtV2 models.

Install dependencies with `pip install -r requirements.txt`. Dataset paths may
be supplied with `--data-path` or the corresponding `ADULT_CSV`,
`DRYBEAN_PATH`, `DEEPDETECT_DIR`, or `TINY_IMAGENET_DIR` environment variable;
otherwise KaggleHub downloads the public dataset.

Run one experiment:

```bash
python scripts/run.py --task adult --mechanism kwta_25
```

Available tasks are `adult`, `drybean`, `deepdetect`, `tinyimagenet`. Selecting `adult` or `drybean` would automatically use the 8-layer MLP. Selecting `deepdetect` or `tinyimagenet` would automatically use a modified ConvNeXtV2 

Available mechanisms are `baseline`, `multi_act`, `het_random`, `het_chunked`,
`kwta_75`, `kwta_50`, and `kwta_25`. Results are written to
`results/{task}/{mechanism}/seed_{n}.json`. Use `python scripts/aggregate.py`
and `python scripts/make_latex_tables.py` to produce summary and LaTeX tables.

Run all tabular or vision experiments and write a validated `summary.csv`:

```bash
python scripts/run_tabular.py
python scripts/run_vision.py --device mps
```
