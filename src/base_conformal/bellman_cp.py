from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple, Union

import numpy as np

Number = Union[int, float]


@dataclass
class BellmanCIConfig:
    alpha: float = 0.1
    alpha_grid: Tuple[float, ...] = tuple(np.linspace(0.01, 0.2, 40))
    horizon: int = 1
    width_weight: float = 1.0
    miss_weight: float = 3.0
    smooth_weight: float = 0.1
    score_window: int = 100
    min_calib_size: int = 30
    warm_start: int = 50
    fallback_width: float = 3.0
    clip_alpha: bool = True
    alpha_min: float = 1e-6
    alpha_max: float = 1.0 - 1e-6
    eps: float = 1e-12
    seed: int = 0


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    a = float(alpha)
    if a <= 0.0:
        return float("inf")
    if a >= 1.0:
        return 0.0
    s = np.asarray(scores, dtype=float).reshape(-1)
    n = int(s.size)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - a)))
    k = max(1, min(n, k))
    return float(np.sort(s)[k - 1])


class BellmanCICP:
    """
    Bellman Conformal Inference baseline.

    The paper frames calibration as a control problem with a Bellman-style
    objective over interval length and coverage.  For this project, we keep the
    interface aligned with the rest of the conformal baselines and implement a
    discrete-action controller over candidate alpha values.

    State summary:
      - recent residual scores
      - recent miss indicators
      - previous alpha

    Action:
      - choose alpha from a fixed grid

    Cost proxy:
      - predicted interval width under candidate alpha
      - predicted miscoverage penalty using recent empirical miss rate
      - smoothness penalty against rapid alpha changes
    """

    def __init__(
        self,
        alpha: float = 0.1,
        alpha_grid: Optional[Tuple[float, ...]] = None,
        horizon: int = 1,
        width_weight: float = 1.0,
        miss_weight: float = 3.0,
        smooth_weight: float = 0.1,
        score_window: int = 100,
        min_calib_size: int = 30,
        warm_start: int = 50,
        fallback_width: float = 3.0,
        clip_alpha: bool = True,
        alpha_min: float = 1e-6,
        alpha_max: float = 1.0 - 1e-6,
        eps: float = 1e-12,
        seed: int = 0,
        **kwargs,
    ):
        self.cfg = BellmanCIConfig(
            alpha=float(alpha),
            alpha_grid=tuple(alpha_grid) if alpha_grid is not None else BellmanCIConfig().alpha_grid,
            horizon=int(horizon),
            width_weight=float(width_weight),
            miss_weight=float(miss_weight),
            smooth_weight=float(smooth_weight),
            score_window=int(score_window),
            min_calib_size=int(min_calib_size),
            warm_start=int(warm_start),
            fallback_width=float(fallback_width),
            clip_alpha=bool(clip_alpha),
            alpha_min=float(alpha_min),
            alpha_max=float(alpha_max),
            eps=float(eps),
            seed=int(seed),
        )

        self.initial_alpha = float(alpha)
        self.alpha_t = float(alpha)
        self.alpha = float(alpha)

        self._scores: Deque[float] = deque(maxlen=self.cfg.score_window)
        self._miss_hist: Deque[float] = deque(maxlen=self.cfg.score_window)
        self._last_interval: Optional[Tuple[float, float]] = None
        self._last_qhat: Optional[float] = None
        self.rng = np.random.default_rng(self.cfg.seed)

    def initialize(self, initial_data=None):
        self.alpha_t = float(self.initial_alpha)
        self.alpha = float(self.initial_alpha)
        self._scores.clear()
        self._miss_hist.clear()
        self._last_interval = None
        self._last_qhat = None

    def start_test(self):
        return

    def _maybe_clip_alpha(self, a: float) -> float:
        if not self.cfg.clip_alpha:
            return float(a)
        return float(np.clip(a, self.cfg.alpha_min, self.cfg.alpha_max))

    def _ready(self) -> bool:
        return len(self._scores) >= max(self.cfg.min_calib_size, self.cfg.warm_start)

    def _recent_miss_rate(self) -> float:
        if not self._miss_hist:
            return float(self.initial_alpha)
        return float(np.mean(self._miss_hist))

    def _predict_next_width(self, alpha: float) -> float:
        scores = np.asarray(list(self._scores), dtype=float)
        q = _conformal_quantile(scores, alpha)
        if not np.isfinite(q):
            return float("inf")
        return float(2.0 * q)

    def _bellman_objective(self, alpha: float) -> float:
        width = self._predict_next_width(alpha)
        if not np.isfinite(width):
            width = 1e6
        miss_rate = self._recent_miss_rate()
        target = float(self.initial_alpha)
        # one-step Bellman proxy: immediate cost + smooth continuation penalty
        return (
            self.cfg.width_weight * width
            + self.cfg.miss_weight * abs(miss_rate - target)
            + self.cfg.smooth_weight * abs(alpha - self.alpha_t)
        )

    def _choose_alpha(self) -> float:
        grid = np.asarray(self.cfg.alpha_grid, dtype=float).reshape(-1)
        if grid.size == 0:
            return self.alpha_t
        grid = np.unique(np.clip(grid, self.cfg.alpha_min, self.cfg.alpha_max))
        objs = np.array([self._bellman_objective(float(a)) for a in grid], dtype=float)
        idx = int(np.argmin(objs))
        return float(grid[idx])

    def predict(
        self,
        base_prediction: Optional[Number] = None,
        model_uncertainty: Number = 1.0,
        x=None,
        **kwargs,
    ) -> Tuple[float, float]:
        mu = float(base_prediction) if base_prediction is not None else 0.0
        unc = float(model_uncertainty)

        if not self._ready():
            w = self.cfg.fallback_width * max(1e-12, unc)
            self._last_interval = (mu - w, mu + w)
            return self._last_interval

        self.alpha_t = self._maybe_clip_alpha(self._choose_alpha())
        scores = np.asarray(list(self._scores), dtype=float)
        q_hat = _conformal_quantile(scores, self.alpha_t)
        q_hat = float(q_hat if np.isfinite(q_hat) else self.cfg.fallback_width * max(1e-12, unc))
        self._last_qhat = q_hat
        self._last_interval = (mu - q_hat, mu + q_hat)
        return self._last_interval

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
            raise ValueError("BellmanCICP.update requires prediction_interval (or interval).")
        if y_pred is None:
            raise ValueError("BellmanCICP.update requires y_pred (point prediction mu_hat).")

        yt = float(y_true)
        mu = float(y_pred)
        lo, hi = float(prediction_interval[0]), float(prediction_interval[1])
        miss = 0.0 if (lo <= yt <= hi) else 1.0

        self._scores.append(float(abs(yt - mu)))
        self._miss_hist.append(float(miss))

        # Keep alpha on the grid and smooth with a tiny correction toward target.
        target = float(self.initial_alpha)
        correction = 0.5 * (target - miss)
        self.alpha_t = self._maybe_clip_alpha(self.alpha_t + correction * 0.01)
        self.alpha = self.alpha_t
