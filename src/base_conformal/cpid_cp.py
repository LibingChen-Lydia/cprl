from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple, Union

import numpy as np

Number = Union[int, float]


@dataclass
class ConformalPIDConfig:
    alpha: float = 0.1
    kp: float = 0.05
    ki: float = 0.01
    kd: float = 0.0

    # score forecasting / smoothing
    score_ema: float = 0.2
    score_window: int = 50

    # finite-sample calibration history
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


class ConformalPIDCP:
    """
    Conformal PID Control for time series prediction.

    Interface matches the other baselines in this project:
      - predict(base_prediction=..., model_uncertainty=..., x=...)
      - update(y_true=..., y_pred=..., prediction_interval=..., x=...)

    Implementation notes:
      - The paper's update is fundamentally a feedback controller over the
        nominal miscoverage/quantile-tracking signal.
      - We keep the model-agnostic wrapper form used elsewhere in the codebase:
        the base forecaster is unchanged and the controller adapts alpha_t.
      - A light score-forecasting path is included via EMA + recent trend.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        kp: float = 0.05,
        ki: float = 0.01,
        kd: float = 0.0,
        score_ema: float = 0.2,
        score_window: int = 50,
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
        self.cfg = ConformalPIDConfig(
            alpha=float(alpha),
            kp=float(kp),
            ki=float(ki),
            kd=float(kd),
            score_ema=float(score_ema),
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
        self._err_hist: Deque[float] = deque(maxlen=3)
        self._score_ema_state: Optional[float] = None
        self._last_qhat: Optional[float] = None
        self._last_interval: Optional[Tuple[float, float]] = None

        self.rng = np.random.default_rng(self.cfg.seed)

    def initialize(self, initial_data=None):
        self.alpha_t = float(self.initial_alpha)
        self.alpha = float(self.initial_alpha)
        self._scores.clear()
        self._err_hist.clear()
        self._score_ema_state = None
        self._last_qhat = None
        self._last_interval = None

    def start_test(self):
        return

    def _maybe_clip_alpha(self, a: float) -> float:
        if not self.cfg.clip_alpha:
            return float(a)
        return float(np.clip(a, self.cfg.alpha_min, self.cfg.alpha_max))

    def _ready(self) -> bool:
        return len(self._scores) >= max(self.cfg.min_calib_size, self.cfg.warm_start)

    def _forecast_score(self) -> float:
        if not self._scores:
            return 0.0
        recent = np.asarray(list(self._scores), dtype=float)
        if self._score_ema_state is None:
            self._score_ema_state = float(recent[-1])
        ema = float(self._score_ema_state)
        beta = float(self.cfg.score_ema)
        for s in recent[-self.cfg.score_window:]:
            ema = beta * float(s) + (1.0 - beta) * ema

        if recent.size >= 2:
            trend = float(recent[-1] - recent[-2])
        else:
            trend = 0.0
        return float(max(0.0, ema + 0.5 * trend))

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

        a = self._maybe_clip_alpha(self.alpha_t)
        scores = np.asarray(list(self._scores), dtype=float)
        q_hat = _conformal_quantile(scores, a)

        # Score forecast is used only as a stabilizer, not as a hard replacement.
        q_forecast = self._forecast_score()
        q_hat = float(0.7 * q_hat + 0.3 * q_forecast)

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
            raise ValueError("ConformalPIDCP.update requires prediction_interval (or interval).")
        if y_pred is None:
            raise ValueError("ConformalPIDCP.update requires y_pred (point prediction mu_hat).")

        yt = float(y_true)
        mu = float(y_pred)
        lo, hi = float(prediction_interval[0]), float(prediction_interval[1])
        miss = 0.0 if (lo <= yt <= hi) else 1.0

        # PID control on the coverage error e_t = target_miscoverage - miss_t.
        target_miss = float(self.initial_alpha)
        err = target_miss - miss
        self._err_hist.append(err)
        err_i = float(np.sum(self._err_hist))
        err_d = float(self._err_hist[-1] - self._err_hist[-2]) if len(self._err_hist) >= 2 else 0.0

        # Update alpha in the direction of the observed coverage error.
        delta = (
            self.cfg.kp * err
            + self.cfg.ki * err_i
            + self.cfg.kd * err_d
        )
        self.alpha_t = self._maybe_clip_alpha(self.alpha_t + delta)
        self.alpha = self.alpha_t

        # Update score history after observing the label.
        self._scores.append(float(abs(yt - mu)))
        self._score_ema_state = (
            float(self._score_ema_state) * (1.0 - self.cfg.score_ema) + float(abs(yt - mu)) * self.cfg.score_ema
            if self._score_ema_state is not None
            else float(abs(yt - mu))
        )
