#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_EXP = ROOT / "run_exp.py"
LOG_DIR = ROOT / "experiment_logs"
COMMAND_LOG_DIR = LOG_DIR / "commands"
REBUILD_BASELINES = ROOT / "scripts" / "rebuild_baseline_results.py"

BASE_SEEDS = [2021, 2022, 2023, 2024, 2025]
BASE_FORECASTERS = [
    "Autoformer",
    "Crossformer",
    "DLinear",
    "FEDformer",
    "iTransformer",
    "PatchTST",
    "Pyraformer",
    "TimesNet",
    "Transformer",
    "TSMixer",
]
BASELINE_MODES = ["acp", "aci", "agaci", "dfpi", "spci", "hopcpt", "nex", "cpid", "bellman"]
ABLATION_MODES = ["M0", "M1", "M2", "M3", "M4", "M5"]
DATASET_PATHS = {
    "exchange_rate": "dataset/exchange_rate/exchange_rate.csv",
    "ETTh1": "time_series_library/dataset/ETT-small/ETTh1.csv",
    "ETTh2": "time_series_library/dataset/ETT-small/ETTh2.csv",
    "ETTm1": "time_series_library/dataset/ETT-small/ETTm1.csv",
    "ETTm2": "time_series_library/dataset/ETT-small/ETTm2.csv",
    "weather": "time_series_library/dataset/weather/weather.csv",
}
NEW_DATASET_MANIFEST = ROOT / "time_series_library" / "dataset" / "new_benchmarks" / "manifest.csv"
RUN_HISTORY_CSV = LOG_DIR / "run_history_v2.csv"
RUN_HISTORY_MD = LOG_DIR / "run_history_v2.md"
RUN_HISTORY_HEADER = [
    "timestamp",
    "phase",
    "dataset",
    "base_seed",
    "cp_seed",
    "cache_path",
    "base_model",
    "cp_mode",
    "ablation_mode",
    "sensitivity_param",
    "sensitivity_value",
    "run_mode",
    "alpha",
    "results_dir",
    "status",
    "failure_reason",
    "command",
    "outputs",
]


@dataclass
class Job:
    phase: str
    dataset: str
    data_path: str
    base_seed: int
    cp_seed: int
    cache_path: Optional[Path]
    base_model: str
    cp_mode: str
    results_dir: Path
    x_lag: int
    lags: int
    alpha: float
    run_mode: str
    ablation_mode: str = ""
    sensitivity_param: str = ""
    sensitivity_value: str = ""


def ensure_log_files() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    COMMAND_LOG_DIR.mkdir(exist_ok=True)
    if not RUN_HISTORY_CSV.exists():
        with RUN_HISTORY_CSV.open("w", newline="") as f:
            csv.writer(f).writerow(RUN_HISTORY_HEADER)
    if not RUN_HISTORY_MD.exists():
        RUN_HISTORY_MD.write_text("", encoding="utf-8")


def append_log(row: Dict[str, str]) -> None:
    ensure_log_files()
    with RUN_HISTORY_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in RUN_HISTORY_HEADER])

    lines = [
        f"## {row['timestamp']}",
        f"- phase: {row['phase']}",
        f"- dataset: {row['dataset']}",
        f"- base_seed: {row.get('base_seed', '')}",
        f"- cp_seed: {row.get('cp_seed', '')}",
        f"- base_model: {row['base_model']}",
        f"- cp_mode: {row['cp_mode']}",
        f"- ablation_mode: {row.get('ablation_mode', '')}",
        f"- sensitivity_param: {row.get('sensitivity_param', '')}",
        f"- sensitivity_value: {row.get('sensitivity_value', '')}",
        f"- cache_path: {row['cache_path']}",
        f"- results_dir: {row['results_dir']}",
        f"- status: {row['status']}",
        f"- outputs: {row.get('outputs', '')}",
        f"- notes: {row.get('failure_reason', '') or 'completed normally'}",
        "",
    ]
    with RUN_HISTORY_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def dataset_to_path(dataset: str) -> str:
    if dataset in DATASET_PATHS:
        return DATASET_PATHS[dataset]
    if NEW_DATASET_MANIFEST.exists():
        try:
            manifest = pd.read_csv(NEW_DATASET_MANIFEST, dtype=str).fillna("")
            hit = manifest[manifest["dataset"] == dataset]
            if not hit.empty and hit.iloc[0].get("data_path"):
                return str(hit.iloc[0]["data_path"])
        except Exception:
            pass
    return dataset


def discover_cache(base_seed: int, dataset: str, model: str) -> Optional[Path]:
    root = ROOT / f"forecast_cache_seed{base_seed}"
    if not root.exists():
        return None

    dataset_l = dataset.lower()
    dataset_aliases = {
        "exchange_rate": ["exchange_rate", "exchange"],
    }
    dataset_keys = dataset_aliases.get(dataset_l, [dataset_l])
    model_l = model.lower()
    setting_re = re.compile(
        r"^long_term_forecast_(?P<dataset>.+?)_cache_(?P<model>[^_]+)_.+$",
        re.IGNORECASE,
    )
    candidates: List[Path] = []
    for path in root.rglob("forecast_full.npz"):
        parent_name = path.parent.name
        m = setting_re.match(parent_name)
        if not m:
            continue

        parsed_dataset = m.group("dataset").lower()
        parsed_model = m.group("model").lower()
        if parsed_dataset in dataset_keys and parsed_model == model_l:
            candidates.append(path)

    if not candidates:
        return None

    chosen = sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]
    return chosen


def build_setting(job: Job) -> str:
    dataset_name = os.path.basename(job.data_path)
    setting = (
        f"{dataset_name}_lags{job.lags}_model{job.base_model}"
        f"_cp{job.cp_mode}_mode{job.run_mode}_base{job.base_seed}_seed{job.cp_seed}"
    )
    if job.ablation_mode:
        setting = f"{setting}_ABL{job.ablation_mode}"
    if job.sensitivity_param:
        setting = f"{setting}_SENS_{job.sensitivity_param}_{job.sensitivity_value}"
    return setting


def result_csv_paths(job: Job) -> Dict[str, Path]:
    if job.phase == "ablation":
        base = job.results_dir / "ablation"
        return {
            "conformal": base / "ablation_conformal_results.csv",
            "adaptive": base / "ablation_adaptive_results.csv",
        }
    return {
        "conformal": job.results_dir / "conformal_results.csv",
        "adaptive": job.results_dir / "adaptive_conformal_results.csv",
    }


def command_for_job(job: Job) -> List[str]:
    csv_paths = result_csv_paths(job)
    cmd = [
        "python3",
        str(RUN_EXP),
        "--data_path", job.data_path,
        "--cache_path", str(job.cache_path),
        "--x_lag", str(job.x_lag),
        "--lags", str(job.lags),
        "--base_model", job.base_model,
        "--cp_mode", job.cp_mode,
        "--run_mode", job.run_mode,
        "--alpha", str(job.alpha),
        "--base_seed", str(job.base_seed),
        "--seed", str(job.cp_seed),
        "--results_dir", str(job.results_dir),
        "--conformal_csv_path", str(csv_paths["conformal"]),
        "--adaptive_csv_path", str(csv_paths["adaptive"]),
    ]
    if job.ablation_mode:
        cmd.extend(["--ablation_mode", job.ablation_mode])
    if job.sensitivity_param:
        cmd.extend(["--setting_suffix", f"SENS_{job.sensitivity_param}_{job.sensitivity_value}"])
    if job.sensitivity_param:
        cmd.extend([f"--{job.sensitivity_param}", str(job.sensitivity_value)])
    return cmd


def write_batch_summary(phase: str, rows: List[Dict[str, str]]) -> None:
    success = sum(1 for r in rows if r["status"] == "success")
    failed = sum(1 for r in rows if r["status"] == "failed")
    skipped = sum(1 for r in rows if r["status"] == "skipped")
    lines = [
        f"## Batch Summary {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- phase: {phase}",
        f"- successful jobs: {success}",
        f"- failed jobs: {failed}",
        f"- skipped jobs: {skipped}",
        "",
    ]
    with RUN_HISTORY_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def rebuild_baseline_outputs(results_dir: Path) -> None:
    rel_output = results_dir.relative_to(ROOT)
    subprocess.run(
        ["python3", str(REBUILD_BASELINES), "--output_dir", str(rel_output)],
        cwd=str(ROOT),
        check=True,
    )


def fetch_result_row(job: Job) -> Dict[str, object]:
    setting = build_setting(job)
    csv_paths = result_csv_paths(job)
    out: Dict[str, object] = {}

    conf_path = csv_paths["conformal"]
    if conf_path.exists():
        df = pd.read_csv(conf_path)
        hit = df[df["setting"] == setting]
        if not hit.empty:
            out.update(hit.iloc[-1].to_dict())

    adp_path = csv_paths["adaptive"]
    if adp_path.exists():
        df = pd.read_csv(adp_path)
        hit = df[df["setting"] == setting]
        if not hit.empty:
            for k, v in hit.iloc[-1].to_dict().items():
                out[f"adaptive_{k}"] = v

    regime_path = job.results_dir / "regime_metrics" / f"{setting}.csv"
    if regime_path.exists():
        out["regime_metrics_path"] = str(regime_path)

    return out


def has_success_history(job: Job) -> bool:
    if not RUN_HISTORY_CSV.exists():
        return False

    try:
        df = pd.read_csv(RUN_HISTORY_CSV, dtype=str).fillna("")
    except Exception:
        return False

    mask = (
        (df["status"] == "success")
        & (df["phase"] == job.phase)
        & (df["dataset"] == job.dataset)
        & (df["base_seed"] == str(job.base_seed))
        & (df["cp_seed"] == str(job.cp_seed))
        & (df["base_model"] == job.base_model)
        & (df["cp_mode"] == job.cp_mode)
        & (df["ablation_mode"] == job.ablation_mode)
        & (df["sensitivity_param"] == job.sensitivity_param)
        & (df["sensitivity_value"] == str(job.sensitivity_value))
        & (df["run_mode"] == job.run_mode)
        & (df["alpha"] == str(job.alpha))
        & (df["results_dir"] == str(job.results_dir))
    )
    return bool(mask.any())


def run_job(job: Job) -> Dict[str, str]:
    ensure_log_files()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_row = {
        "timestamp": ts,
        "phase": job.phase,
        "dataset": job.dataset,
        "base_seed": str(job.base_seed),
        "cp_seed": str(job.cp_seed),
        "cache_path": str(job.cache_path) if job.cache_path else "",
        "base_model": job.base_model,
        "cp_mode": job.cp_mode,
        "ablation_mode": job.ablation_mode,
        "sensitivity_param": job.sensitivity_param,
        "sensitivity_value": str(job.sensitivity_value),
        "run_mode": job.run_mode,
        "alpha": str(job.alpha),
        "results_dir": str(job.results_dir),
        "status": "",
        "failure_reason": "",
        "command": "",
        "outputs": "",
    }

    if job.cache_path is None or not job.cache_path.exists():
        base_row["status"] = "skipped"
        base_row["failure_reason"] = "cache not found"
        append_log(base_row)
        return base_row

    if has_success_history(job):
        base_row["status"] = "skipped"
        base_row["failure_reason"] = "already succeeded previously"
        append_log(base_row)
        return base_row

    cmd = command_for_job(job)
    base_row["command"] = shlex.join(cmd)
    log_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job.phase}_"
        f"{job.dataset}_{job.base_model}_{job.cp_mode}_base{job.base_seed}_cp{job.cp_seed}.log"
    )
    cmd_log_path = COMMAND_LOG_DIR / log_name

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp/cprl_mpl")

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    cmd_log_path.write_text(proc.stdout or "", encoding="utf-8")

    if proc.returncode != 0:
        base_row["status"] = "failed"
        base_row["failure_reason"] = f"command exited non-zero; see {cmd_log_path}"
        append_log(base_row)
        return base_row

    result_row = fetch_result_row(job)
    if not result_row:
        base_row["status"] = "failed"
        base_row["failure_reason"] = f"run completed but no result row found; see {cmd_log_path}"
        append_log(base_row)
        return base_row

    outputs = []
    for p in result_csv_paths(job).values():
        if p.exists():
            outputs.append(str(p))
    if "regime_metrics_path" in result_row:
        outputs.append(str(result_row["regime_metrics_path"]))

    base_row["status"] = "success"
    base_row["outputs"] = ", ".join(outputs)
    append_log(base_row)
    return base_row


def summarise_runs(rows: List[Dict[str, object]], group_cols: List[str], out_dir: Path) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    raw_path = out_dir / "seed_summary_raw.csv"
    agg_path = out_dir / "seed_summary_agg.csv"
    df.to_csv(raw_path, index=False)

    metric_cols = [
        "coverage",
        "coverage_gap",
        "avg_width",
        "ces",
        "rcs",
        "point_mse",
        "point_mae",
        "adaptive_worst_window_coverage",
        "adaptive_width_step_mean",
        "adaptive_width_std",
        "adaptive_control_alpha_step_mean",
        "adaptive_control_alpha_std",
    ]
    use_metrics = [c for c in metric_cols if c in df.columns]
    if not use_metrics:
        return

    agg = df.groupby(group_cols, dropna=False)[use_metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    agg.to_csv(agg_path, index=False)


def summarise_regime_metrics(success_jobs: List[Job], out_dir: Path) -> None:
    rows: List[pd.DataFrame] = []
    for job in success_jobs:
        regime_path = out_dir / "regime_metrics" / f"{build_setting(job)}.csv"
        if not regime_path.exists():
            continue
        df = pd.read_csv(regime_path)
        df["dataset"] = job.dataset
        df["base_seed"] = job.base_seed
        df["cp_seed"] = job.cp_seed
        df["base_model"] = job.base_model
        df["cp_mode"] = job.cp_mode
        df["ablation_mode"] = job.ablation_mode
        df["sensitivity_param"] = job.sensitivity_param
        df["sensitivity_value"] = job.sensitivity_value
        rows.append(df)

    if not rows:
        return

    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(out_dir / "regime_metrics_by_seed.csv", index=False)

    group_cols = [
        "dataset",
        "base_model",
        "cp_mode",
        "target_coverage",
        "ablation_mode",
        "sensitivity_param",
        "sensitivity_value",
        "regime_id",
    ]
    metrics = ["count", "coverage", "avg_width", "ces"]
    agg = raw.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    agg.to_csv(out_dir / "regime_metrics_agg.csv", index=False)


def make_jobs(args: argparse.Namespace) -> List[Job]:
    dataset = args.dataset
    data_path = dataset_to_path(args.dataset_path or dataset)
    results_dir = ROOT / args.results_dir
    jobs: List[Job] = []

    if args.phase == "baseline":
        cp_modes = args.cp_modes or BASELINE_MODES
        models = args.models or BASE_FORECASTERS
        for base_seed in [args.base_seed]:
            for model in models:
                cache_path = discover_cache(base_seed, dataset, model)
                for cp_mode in cp_modes:
                    jobs.append(Job(
                        phase="baseline",
                        dataset=dataset,
                        data_path=data_path,
                        base_seed=base_seed,
                        cp_seed=args.cp_seed,
                        cache_path=cache_path,
                        base_model=model,
                        cp_mode=cp_mode,
                        results_dir=results_dir,
                        x_lag=args.x_lag,
                        lags=args.lags,
                        alpha=args.alpha,
                        run_mode=args.run_mode,
                    ))
    elif args.phase == "ablation":
        models = args.models or BASE_FORECASTERS
        modes = args.ablation_modes or ABLATION_MODES
        for seed in BASE_SEEDS:
            for model in models:
                cache_path = discover_cache(seed, dataset, model)
                for mode in modes:
                    jobs.append(Job(
                        phase="ablation",
                        dataset=dataset,
                        data_path=data_path,
                        base_seed=seed,
                        cp_seed=seed,
                        cache_path=cache_path,
                        base_model=model,
                        cp_mode="acp",
                        results_dir=results_dir,
                        x_lag=args.x_lag,
                        lags=args.lags,
                        alpha=args.alpha,
                        run_mode=args.run_mode,
                        ablation_mode=mode,
                    ))
    else:
        models = args.models or ["iTransformer"]
        values = args.values
        for seed in BASE_SEEDS:
            for model in models:
                cache_path = discover_cache(seed, dataset, model)
                for value in values:
                    jobs.append(Job(
                        phase="sensitivity",
                        dataset=dataset,
                        data_path=data_path,
                        base_seed=seed,
                        cp_seed=seed,
                        cache_path=cache_path,
                        base_model=model,
                        cp_mode="acp",
                        results_dir=results_dir,
                        x_lag=args.x_lag,
                        lags=args.lags,
                        alpha=args.alpha,
                        run_mode=args.run_mode,
                        sensitivity_param=args.param,
                        sensitivity_value=str(value),
                    ))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cache-based CPRL experiments with separate base and CP seeds.")
    sub = parser.add_subparsers(dest="phase", required=True)

    def add_common(p: argparse.ArgumentParser, default_results: str) -> None:
        p.add_argument("--dataset", required=True, help="Dataset key, e.g. exchange_rate, ETTh1, ETTh2, ETTm1, ETTm2, weather")
        p.add_argument("--dataset_path", default="", help="Optional explicit data_path passed to run_exp.py")
        p.add_argument("--models", nargs="*", default=[], help="Base forecasters to run")
        p.add_argument("--alpha", type=float, default=0.1)
        p.add_argument("--run_mode", default="online", choices=["online", "offline"])
        p.add_argument("--x_lag", type=int, default=96)
        p.add_argument("--lags", type=int, default=96)
        p.add_argument("--results_dir", default=default_results)

    p1 = sub.add_parser("baseline")
    add_common(p1, "results/baseline_all_v2")
    p1.add_argument("--base_seed", type=int, default=2021, help="Seed used to select the cached base forecaster.")
    p1.add_argument("--cp_seed", type=int, default=2011, help="Seed used by the CP method.")
    p1.add_argument(
        "--cp_modes",
        nargs="*",
        default=[],
        help="Comparison methods; default: acp aci agaci dfpi hopcpt nex cpid bellman",
    )

    p2 = sub.add_parser("ablation")
    add_common(p2, "results_cache_ablation")
    p2.add_argument("--ablation_modes", nargs="*", default=[], help="Default: M0 M1 M2 M3 M4 M5")

    p3 = sub.add_parser("sensitivity")
    add_common(p3, "results_cache_sensitivity")
    p3.add_argument("--param", required=True, help="ACP hyperparameter name without leading dashes")
    p3.add_argument("--values", nargs="+", required=True, help="Values for the sensitivity parameter")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = make_jobs(args)
    run_rows: List[Dict[str, str]] = []
    success_jobs: List[Job] = []
    success_metric_rows: List[Dict[str, object]] = []

    for job in jobs:
        row = run_job(job)
        run_rows.append(row)
        if row["status"] != "success":
            continue
        success_jobs.append(job)
        metrics = fetch_result_row(job)
        metrics.update({
            "phase": job.phase,
            "dataset": job.dataset,
            "base_seed": job.base_seed,
            "cp_seed": job.cp_seed,
            "base_model": job.base_model,
            "cp_mode": job.cp_mode,
            "ablation_mode": job.ablation_mode,
            "sensitivity_param": job.sensitivity_param,
            "sensitivity_value": job.sensitivity_value,
        })
        success_metric_rows.append(metrics)

    results_dir = ROOT / args.results_dir
    group_cols = ["dataset", "base_model", "cp_mode", "target_coverage"]
    if args.phase == "ablation":
        group_cols.append("ablation_mode")
    if args.phase == "sensitivity":
        group_cols.extend(["sensitivity_param", "sensitivity_value"])

    summarise_runs(success_metric_rows, group_cols, results_dir)
    summarise_regime_metrics(success_jobs, results_dir)
    # Optional post-processing; the experiment runner itself does not require it.
    if args.phase == "baseline" and os.environ.get("RUN_BASELINE_REBUILD") == "1":
        rebuild_baseline_outputs(results_dir)
    write_batch_summary(args.phase, run_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
