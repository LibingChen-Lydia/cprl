# src/base_conformal/aci_cp.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple, Union

import numpy as np

Number = Union[int, float]


@dataclass
class ACIConfig:
    """
    Time-series ACI (sequential) aligned with Zaffran et al. (2022):
    - Online update of alpha_t (ACI recursion).
    - Sequential calibration set (no random split) to account for temporal structure.
      See: "we change the random split to a sequential split." :contentReference[oaicite:4]{index=4}
    - Allows alpha_t <= 0 leading to infinite intervals (predict R) by convention. :contentReference[oaicite:5]{index=5}

    This implementation is a wrapper around a point-forecasting model:
    base model gives mu_hat; we conformalize absolute residuals |y - mu_hat|.
    """

    alpha0: float = 0.1     
    gamma: float = 0.01     

    T0: int = 500

    min_calib_size: int = 30

    warm_start: int = 50
    fallback_width: float = 3.0

    clip_alpha: bool = True
    eps: float = 1e-6

    seed: int = 0


def _corrected_conformal_quantile(scores: np.ndarray, alpha_t: float) -> float:
    """
    Finite-sample corrected split conformal quantile using order statistics:
      k = ceil((n + 1) * (1 - alpha_t))
      q = k-th smallest score (1-indexed), with k clipped to [1, n].

    Paper-aligned conventions for ACI:
    - If alpha_t <= 0: q = +inf  => interval is R (infinite).
    - If alpha_t >= 1: q = 0     => degenerate to point (rare in practice).
    """
    a = float(alpha_t)

    if a <= 0.0:
        return float("inf")
    if a >= 1.0:
        return 0.0

    s = np.asarray(scores, dtype=float).reshape(-1)
    n = int(s.size)
    if n <= 0:
        return float("inf")

    s_sorted = np.sort(s)
    k = int(np.ceil((n + 1) * (1.0 - a)))  # 1-indexed
    k = max(1, min(n, k))
    return float(s_sorted[k - 1])


class ACICP:
    """
    Time-series ACI (sequential split) conformal predictor.

    Expected usage in your framework:
      interval = cp.predict(base_prediction=mu_hat, model_uncertainty=..., x=...)
      cp.update(y_true=y, y_pred=mu_hat, prediction_interval=interval, x=...)

    Notes:
    - We do not refit the base model inside ACI; we only maintain a sequential
      calibration set of recent residual scores and adapt alpha_t online.
    - This matches the common "wrap-around" usage of ACI for forecasting.
    """

    def __init__(
        self,
        alpha: float,
        window_size: Optional[int] = None,   # kept for builder compatibility (unused)
        min_calib_size: int = 30,
        gamma: Optional[float] = None,
        lr: Optional[float] = None,          # backward-compatible alias for gamma
        T0: int = 500,
        warm_start: int = 50,
        fallback_width: float = 3.0,
        clip_alpha: bool = True,
        eps: float = 1e-6,
        seed: int = 0,
        **kwargs,
    ):
        if gamma is None:
            gamma = 0.01 if lr is None else lr

        self.config = ACIConfig(
            alpha0=float(alpha),
            gamma=float(gamma),
            T0=int(T0),
            min_calib_size=int(min_calib_size),
            warm_start=int(warm_start),
            fallback_width=float(fallback_width),
            clip_alpha=bool(clip_alpha),
            eps=float(eps),
            seed=int(seed),
        )

        self.initial_alpha = float(alpha)
        self.alpha_t = float(alpha)
        self.alpha = self.alpha_t

        # Sequential rolling window of residual scores (last T0 points).
        self._scores: Deque[float] = deque(maxlen=self.config.T0)

        self.rng = np.random.default_rng(self.config.seed)

    def initialize(self, initial_data=None):
        self.alpha_t = float(self.initial_alpha)
        self._scores.clear()

    def _maybe_clip_alpha(self, a: float) -> float:
        if not self.config.clip_alpha:
            return float(a)
        return float(np.clip(a, self.config.eps, 1.0 - self.config.eps))

    def _ready(self) -> bool:
        return len(self._scores) >= max(self.config.min_calib_size, self.config.warm_start)

    def predict(
        self,
        base_prediction: Optional[Number] = None,
        model_uncertainty: Number = 1.0,
        x=None,
        **kwargs,
    ) -> Tuple[float, float]:
        mu = float(base_prediction) if base_prediction is not None else 0.0
        unc = float(model_uncertainty)

        # warm start fallback
        if not self._ready():
            w = self.config.fallback_width * max(1e-12, unc)
            return mu - w, mu + w

        a = self._maybe_clip_alpha(self.alpha_t)
        scores = np.fromiter(self._scores, dtype=float)
        q_hat = _corrected_conformal_quantile(scores, a)

        # Infinite interval convention (alpha_t <= 0 => q_hat = +inf)
        if np.isinf(q_hat):
            return float("-inf"), float("inf")

        return mu - q_hat, mu + q_hat

    def update(
        self,
        y_true: Number,
        y_pred: Optional[Number] = None,
        prediction_interval: Optional[Tuple[Number, Number]] = None,
        interval: Optional[Tuple[Number, Number]] = None,
        x=None,
        **kwargs,
    ):
        if prediction_interval is None:
            prediction_interval = interval
        if prediction_interval is None:
            raise ValueError("ACICP.update requires prediction_interval (or interval).")
        if y_pred is None:
            raise ValueError("ACICP.update requires y_pred (point prediction mu_hat).")

        yt = float(y_true)
        mu = float(y_pred)
        lo, hi = float(prediction_interval[0]), float(prediction_interval[1])

        miss = 0.0 if (lo <= yt <= hi) else 1.0

        self.alpha_t = float(self.alpha_t + self.config.gamma * (self.initial_alpha - miss))
        self.alpha_t = self._maybe_clip_alpha(self.alpha_t)
        self.alpha = self.alpha_t

        self._scores.append(float(abs(yt - mu)))
