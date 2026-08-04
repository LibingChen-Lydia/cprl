# CPRL

Adaptive Conformal Prediction via Frequency-Domain Wasserstein Calibration and Latent State Modeling

---

## 1. Overview

This repository contains the experimental implementation for **CPRL**, a conformal prediction framework for non-stationary time series forecasting. The standard experimental protocol separates point forecasting from conformal calibration: first train a base forecaster and export its validation/test predictions as a deterministic forecast cache; then run CPRL and the conformal baselines on the same cached forecasts.

The main components are:

- Regime-aware adaptive conformal calibration with spectral drift sensing.
- Baselines including `aci`, `agaci`, `nex`, `cqr`, `dfpi`, `enbpi`, `cptc`, `hopcpt`, `spci`, `cpid`, and `bellman`.
- Cache-based conformal evaluation from fixed base-forecaster predictions, which keeps CP comparisons independent of forecasting retraining noise.
- Optional integration with forecasting architectures under `time_series_library/`.

The main CLI entry point is `run_exp.py`.

---

## 2. Environment Setup

### 2.1 Basic Dependencies

Python 3.9+ is recommended. Key dependencies include:

- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- Other dependencies (if a `requirements.txt` is provided in the repo).

Install using pip:

```bash
pip install -r requirements.txt
```

### 2.2 Optional Forecasting Backbones

To use advanced deep learning models (instead of the default linear model) as the base forecaster:

- Install and ensure `time_series_library` is importable.
- Use the `--base_model` argument to specify a model name registered in the library.

If the library fails to import, the code gracefully degrades to support only the `linear` base model and prints:

> `[Warning] MODEL_REGISTRY unavailable; only 'linear' base model is usable.`

---

## 3. Data Format

The codebase currently assumes a **univariate time series** stored in a CSV file:

- Use `--data_path` to specify the CSV file path (required).
- By default, the **last column** is treated as the target variable.
- Use `--target_col` to explicitly specify a column name.

Example CSV (can contain multiple columns):

```text
timestamp,value
2020-01-01 00:00:00, 1.23
2020-01-01 01:00:00, 1.27
...
```

Notes:

- The target series is automatically normalized.
- Use `--lags` to control the window size: the series is transformed into supervised samples `(X, y)`, where `X` is a lagged window of length `lags` and `y` is the next-step target.
- Data is split **chronologically** into train / calibration / test (no shuffling).

Benchmark datasets are not included in the repository. Keep local datasets outside version control and pass their paths through `--data_path`.

---

## 4. Quick Start

### 4.1 Minimal Example

Run with the default linear base model and ACP mode with online updates:

```bash
python run_exp.py \
  --data_path path/to/your_series.csv \
  --lags 96 \
  --train_ratio 0.6 \
  --calib_ratio 0.2 \
  --alpha 0.1 \
  --cp_mode acp \
  --run_mode online
```

Key arguments (commonly used):

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to CSV data file | (required) |
| `--target_col` | Target column name; uses last column if omitted | `None` |
| `--lags` | Lag window size (lookback) | `96` |
| `--train_ratio` | Proportion of data for training | `0.6` |
| `--calib_ratio` | Proportion of data for calibration | `0.2` |
| `--alpha` | Nominal significance level; target coverage = `1 - alpha` | `0.1` |
| `--cp_mode` | CP method: `acp`, `aci`, `agaci`, `nex`, `cqr`, `dfpi`, `enbpi`, `cptc`, `hopcpt`, `spci`, `cpid`, `bellman` | `acp` |
| `--run_mode` | `online` or `offline` conformal evaluation | `online` |
| `--results_dir` | Directory for numerical results | `./results` |

### 4.2 Standard Two-Stage Protocol

For paper-scale experiments, first train the base forecaster and export its predictions:

```bash
python scripts/run_tsl_forecast_cache.py \
  --model_id ETTh1_cache \
  --model DLinear \
  --data custom \
  --root_path time_series_library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --seed 2021
```

This creates a file of the form:

```text
forecast_cache_seed2021/<forecast-setting>/forecast_full.npz
```

Then run conformal calibration on the fixed forecast cache:

```bash
python run_exp.py \
  --data_path time_series_library/dataset/ETT-small/ETTh1.csv \
  --cache_path forecast_cache_seed2021/<forecast-setting>/forecast_full.npz \
  --base_model DLinear \
  --cp_mode acp \
  --run_mode online \
  --lags 96 \
  --x_lag 96 \
  --alpha 0.1 \
  --results_dir results/ETTh1_DLinear_acp
```

Use the same `--cache_path` when comparing different conformal methods. This ensures that differences in coverage and interval width come from the conformal procedure rather than from different point forecasts.

Forecast caches are generated artifacts and are not tracked by Git. If precomputed caches are needed for exact reproduction, host them externally and place them under `forecast_cache_seed*/` after download.

### 4.3 Forecast Cache Format

The conformal runner expects an `.npz` file with validation and test predictions. The required arrays are:

| Key | Description |
|-----|-------------|
| `val_y_true` or `val_y_true_full` | Ground-truth values on the validation/calibration split. |
| `val_y_pred` or `val_y_pred_full` | Base-forecaster predictions on the validation/calibration split. |
| `test_y_true` or `test_y_true_full` | Ground-truth values on the test split. |
| `test_y_pred` or `test_y_pred_full` | Base-forecaster predictions on the test split. |

Optional arrays `val_time_idx` and `test_time_idx` are used to sort predictions chronologically when present. Lag features for regime detection are rebuilt from past true values using `--x_lag`.

### 4.4 One-Step Sanity Runs

For quick local checks, `run_exp.py` can also train the built-in linear forecaster and run conformal calibration in one command:

```bash
python run_exp.py --data_path path/to/series.csv --base_model linear
```

This mode is useful for smoke tests, but benchmark comparisons should use the two-stage cache protocol above.

### 4.5 Choosing the Base Model

Use a registered model from `time_series_library` when generating forecast caches, for example:

```bash
python scripts/run_tsl_forecast_cache.py \
  --model_id ETTh1_cache \
  --model Autoformer \
  --root_path time_series_library/dataset/ETT-small/ \
  --data_path ETTh1.csv
```

If the specified model is not in `MODEL_REGISTRY`, an error is raised showing available options.

### 4.6 Device Selection

| Argument | Description |
|----------|-------------|
| `--use_gpu` | Enable CUDA when available. Use `0` for CPU-only execution and `1` for CUDA-enabled execution. |

Example: Use CUDA GPU 0:

```bash
python run_exp.py \
  --data_path path/to/series.csv \
  --use_gpu 1
```

---

## 5. Outputs and Visualizations

After execution, numerical outputs are written under `results_dir`, while diagnostic figures are written under `v_results/`.

### 5.1 Numerical Results

- `results/conformal_results.csv`
- `results/adaptive_conformal_results.csv`
- Corresponding Excel files may be generated for local inspection.

### 5.2 Dynamics Logs

- `results/dynamics/<setting>.csv`  
  Records rolling-window metrics such as coverage, interval width, and control signal evolution over time.

### 5.3 Visualizations (under `v_results/`)

| Path | Description |
|------|-------------|
| `v_results/prediction_intervals/*.png` | Time-series plot of true values, point predictions, and prediction intervals. |
| `v_results/alpha_curves/*.png` | Adaptive control signal (alpha or equivalent) over the test set. |
| `v_results/interval_widths/*.png` | Prediction interval width over time. |

File names typically include: dataset name, base model, CP mode, run mode, random seed, and timestamp.

---

## 6. Code Structure

Key modules relevant to the main experimental workflow:

| File | Description |
|------|-------------|
| `run_exp.py` | Command-line entry point: parses arguments, configures imports, instantiates `ExpConformal`, and invokes `run()`. |
| `exp/exp_basic.py` | `ExpBasic`: Base experiment class. Loads CSV, normalizes, constructs lagged features, chronological split, and builds `DataLoader`s. |
| `exp/exp_conformal.py` | `ExpConformal(ExpBasic)`: Full pipeline. Defines base-model or cache-based evaluation, builds CP predictor via `build_conformal_predictor`, computes metrics, and saves outputs. |
| `src/utils.py` | Data preprocessing, metric computation (coverage, width, CES, RCS, worst-window coverage), and print utilities. |
| `src/base_conformal/` | Implementations and builders for various conformal predictors. |
| `src/result_logger.py` | Logging to CSV and Excel. |
| `scripts/run_tsl_forecast_cache.py` | Trains a `time_series_library` forecaster and exports deterministic forecast caches. |

---

## 7. Reproducing Experiments

Recommended workflow:

1. Prepare a univariate CSV time series (ensure sufficient length for `lags + train + calib + test` samples).
2. Train the base forecaster with a fixed seed and export `forecast_full.npz`.
3. Run each conformal method with the same `--cache_path`.
4. Compare coverage, width, and stability metrics from the generated result files.

Generated results, cached forecasts, logs, and exploratory figures are intentionally excluded from version control. Commit only source code, scripts, lightweight configuration, and documentation.

---

## 8. Citation

If you use this code in your research, please cite:

> Adaptive Conformal Prediction via Frequency-Domain Wasserstein Calibration and Latent State Modeling

---

## 9. License

This repository is released under the license specified in `LICENSE`.
