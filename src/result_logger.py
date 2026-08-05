import csv
import os
import pandas as pd
from typing import Dict, Optional


class ResultLogger:

    CONFORMAL_HEADER = [
        "setting",
        "cp_mode",
        "target_coverage",
        "runtime_seconds",
        "coverage",
        "abs_coverage_gap",
        "under_coverage_gap",
        "over_coverage_gap",
        "coverage_bias",
        "coverage_bias_direction",
        "avg_width",
        "ces",
        "rcs",
        "point_mse",
        "point_mae",
        "comment",
    ]

    ADAPTIVE_HEADER = [
        "setting",
        "cp_mode",
        "target_coverage",
        "runtime_seconds",
        "worst_window_coverage",
        "width_step_mean",
        "width_std",
        "control_alpha_step_mean",
        "control_alpha_std",
        "comment",
    ]

    def __init__(
        self,
        conformal_csv_path: str = "results/conformal_results.csv",
        adaptive_csv_path: str = "results/adaptive_conformal_results.csv",
    ):
        self.conformal_csv_path = conformal_csv_path
        self.adaptive_csv_path = adaptive_csv_path

        os.makedirs(os.path.dirname(self.conformal_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.adaptive_csv_path), exist_ok=True)

        self._ensure_header(self.conformal_csv_path, self.CONFORMAL_HEADER)
        self._ensure_header(self.adaptive_csv_path, self.ADAPTIVE_HEADER)

    @staticmethod
    def _ensure_header(csv_path: str, header: list):
        if not os.path.exists(csv_path):
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            return

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            return

        if list(df.columns) == header:
            return

        for col in header:
            if col not in df.columns:
                df[col] = None
        df = df[header]
        df.to_csv(csv_path, index=False)

    @staticmethod
    def _append_row(csv_path: str, header: list, row_dict: Dict):
        row = [row_dict.get(col, None) for col in header]
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


    def log_conformal(
        self,
        setting: str,
        cp_mode: str,
        target_coverage: float,
        runtime_seconds: Optional[float],
        metrics: Dict,
        comment: str = "",
    ):

        row = {
            "setting": setting,
            "cp_mode": cp_mode,
            "target_coverage": target_coverage,
            "runtime_seconds": runtime_seconds,
            "coverage": metrics.get("coverage"),
            "abs_coverage_gap": metrics.get("abs_coverage_gap"),
            "under_coverage_gap": metrics.get("under_coverage_gap"),
            "over_coverage_gap": metrics.get("over_coverage_gap"),
            "coverage_bias": metrics.get("coverage_bias"),
            "coverage_bias_direction": metrics.get("coverage_bias_direction"),
            "avg_width": metrics.get("avg_width"),
            "ces": metrics.get("ces"),
            "rcs": metrics.get("rcs"),
            "point_mse": metrics.get("point_mse"),
            "point_mae": metrics.get("point_mae"),
            "comment": comment,
        }
        self._append_row(self.conformal_csv_path, self.CONFORMAL_HEADER, row)

    def log_adaptive(
        self,
        setting: str,
        cp_mode: str,
        target_coverage: float,
        runtime_seconds: Optional[float],
        adaptive_metrics: Dict,
        comment: str = "",
    ):

        row = {
            "setting": setting,
            "cp_mode": cp_mode,
            "target_coverage": target_coverage,
            "runtime_seconds": runtime_seconds,
            "worst_window_coverage": adaptive_metrics.get("worst_window_coverage"),
            "width_step_mean": adaptive_metrics.get("width_step_mean"),
            "width_std": adaptive_metrics.get("width_std"),
            "control_alpha_step_mean": adaptive_metrics.get("control_alpha_step_mean"),
            "control_alpha_std": adaptive_metrics.get("control_alpha_std"),
            "comment": comment,
        }
        self._append_row(self.adaptive_csv_path, self.ADAPTIVE_HEADER, row)

    def save(self):
        """Compatibility hook (no-op)."""
        pass

    def to_excel(self):

        out = {}

        df1 = pd.read_csv(self.conformal_csv_path)
        xlsx1 = self.conformal_csv_path.replace(".csv", ".xlsx")
        df1.to_excel(xlsx1, index=False)
        out["conformal"] = xlsx1

        df2 = pd.read_csv(self.adaptive_csv_path)
        xlsx2 = self.adaptive_csv_path.replace(".csv", ".xlsx")
        df2.to_excel(xlsx2, index=False)
        out["adaptive"] = xlsx2

        return out
