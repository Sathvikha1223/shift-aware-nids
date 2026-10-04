# Shift-Aware Selective NIDS

This project contains a modular repeated leave-one-domain-out (LODO) pipeline for a shift-aware selective network intrusion detection system. The original `nids_shift_aware.ipynb` remains unchanged as the reduced-scope proof-of-concept reference.

## Detected datasets

The project automatically detects CSV and Parquet datasets under the `data/` folder:

- NF-UNSW-NB15-v3.csv
- NF-ToN-IoT-v3.csv
- NF-BoT-IoT-v3.csv
- NF-CICIDS2018-v3.csv

These datasets are NetFlow-style CSVs with the common column family used by NF-v3 datasets, including `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS`, `IPV4_SRC_ADDR`, `L4_SRC_PORT`, `IPV4_DST_ADDR`, `L4_DST_PORT`, `Label`, and `Attack`.

## Verified assumptions

- Label column: `Label`
- Attack column: `Attack`
- Benign marker: `Benign`
- Identity/shortcut columns removed: IP addresses, ports, TTLs, flow timestamps, and flow IDs.
- Target labels are excluded from model training, calibration, and threshold selection; they are used only for final evaluation.
- Global normalization is fit on source training rows only. Per-domain normalization and CORAL may use unlabeled held-out features, as recorded in each result's protocol fields.
- Shortcut identifiers are excluded and each fold audits target membership and exact source/target feature overlap.

## Current implementation status

The modular pipeline includes:

- bounded-memory, seeded CSV-chunk and Parquet-batch sampling across the full file
- configurable `stratified` sampling (default 50% benign) or `uniform` sampling
- population and selected-sample class counts in run summaries
- LightGBM, MLP, CORAL, GroupDRO-style reweighting, DANN, and tree/MLP ensemble wrappers
- source-validation temperature scaling and cross-fitted leave-one-source-domain-out temperature scaling
- per-fold training/calibration/threshold/duplicate leakage audit
- CSV/JSON result export and a Streamlit results viewer

Stratified sampling intentionally uses labels to form an evaluation sample with both classes where available; it is not a natural-prevalence estimate. Use `--sampling-strategy uniform` when natural class prevalence is needed. Both strategies scan the full input in chunks/batches and retain only bounded samples.

CORAL is a **transductive, unsupervised domain-adaptation** result: it uses unlabeled held-out target features to estimate alignment statistics, never target labels. The result rows and `summary.json` mark it as `transductive_unlabeled_target_adaptation`; other configured models are marked `source_only`.

DANN includes a Gradient Reversal Layer with configurable `--dann-grl-lambda`. LDO calibration fits one calibration model per source domain, leaving that domain out of model training; the outer held-out target is excluded from all calibration folds.

## Run a small modular validation

```bash
python src/run_full.py --seeds 0 --models lightgbm mlp coral groupdro dann tree_mlp_ensemble --max-rows-per-domain 500 --normalizers global --sampling-strategy stratified --sampling-seed 42
```

The command scans each input once, then evaluates all discovered target domains. Results, per-domain sample counts, and protocol labels are written under `reports/full/`. Increase the row cap only after validating the small run.

The original reference proof-of-concept runner remains available:

```bash
python src/run_poc.py --max-rows-per-domain 20000 --seed 42
```

Its results are written to `reports/poc/`.

## Project structure

- `src/` - core pipeline modules
- `data/` - dataset CSV files
- `reports/` - generated CSV/JSON outputs
- `tests/` - pipeline regression tests
- `app.py` - minimal Streamlit dashboard
- `nids_shift_aware.ipynb` - original research notebook used as the starting point
