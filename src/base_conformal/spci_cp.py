from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SPCIConfig:
    alpha: float = 0.1
    past_window: int = 10
    min_calib_size: int = 30
    calib_window_size: int = 200
    n_estimators: int = 100
    max_depth: int = 5
    max_features: float = 1.0
    min_samples_leaf: int = 1
    random_state: int = 1103
    beta_grid: int = 101
    refit_every: int = 1
    fallback_width: float = 3.0


class SPCICP:
    """
    Sequential Predictive Conformal Inference (SPCI), Xu and Xie (ICML 2023).

    This class adapts the official SPCI-code implementation to this project's
    streaming CP interface. The core SPCI mechanics are preserved:
      - use signed forecast residuals y_t - yhat_t;
      - build rolling residual-history features;
      - fit a Quantile Random Forest over future residuals;
      - evaluate beta in [0, alpha] and select the narrowest predicted interval
        [Q_beta, Q_{1-alpha+beta}].

    Official source mirrored:
      https://github.com/hamrel-cxu/SPCI-code
    """

    def __init__(
        self,
        alpha: float,
        past_window: int = 10,
        min_calib_size: int = 30,
        calib_window_size: int = 200,
        n_estimators: int = 100,
        max_depth: int = 5,
        max_features: float = 1.0,
        min_samples_leaf: int = 1,
        random_state: int = 1103,
        beta_grid: int = 101,
        refit_every: int = 1,
        fallback_width: float = 3.0,
        **kwargs,
    ):
        self.cfg = SPCIConfig(
            alpha=float(alpha),
            past_window=int(past_window),
            min_calib_size=int(min_calib_size),
            calib_window_size=int(calib_window_size),
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            max_features=float(max_features),
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            beta_grid=int(beta_grid),
            refit_every=max(1, int(refit_every)),
            fallback_width=float(fallback_width),
        )

        if self.cfg.past_window < 1:
            raise ValueError("SPCI past_window must be >= 1.")
        if self.cfg.beta_grid < 2:
            raise ValueError("SPCI beta_grid must be >= 2.")

        self.alpha = float(alpha)
        self.initial_alpha = float(alpha)

        self._residuals: List[float] = []
        self._resid_pool = deque(maxlen=self.cfg.calib_window_size)
        self._model = None
        self._q_levels: Optional[np.ndarray] = None
        self._in_test = False
        self._t_test = 0

    def initialize(self, initial_data=None):
        self.alpha = float(self.initial_alpha)
        self._residuals = []
        self._resid_pool = deque(maxlen=self.cfg.calib_window_size)
        self._model = None
        self._q_levels = None
        self._in_test = False
        self._t_test = 0

    @staticmethod
    def _beta_levels(alpha: float, beta_grid: int) -> np.ndarray:
        betas = np.linspace(0.0, float(alpha), int(beta_grid))
        levels = np.concatenate([betas, 1.0 - float(alpha) + betas])
        return np.clip(levels, 0.0, 1.0)

    @staticmethod
    def _lagged_residual_dataset(residuals: Sequence[float], past_window: int) -> Tuple[np.ndarray, np.ndarray]:
        r = np.asarray(residuals, dtype=float).reshape(-1)
        n = int(r.size)
        if n <= past_window:
            return np.empty((0, past_window), dtype=float), np.empty((0,), dtype=float)

        X = np.empty((n - past_window, past_window), dtype=float)
        y = np.empty((n - past_window,), dtype=float)
        for i in range(past_window, n):
            X[i - past_window] = r[i - past_window:i]
            y[i - past_window] = r[i]
        return X, y

    @staticmethod
    def _format_quantile_predictions(pred, n_quantiles: int) -> np.ndarray:
        arr = np.asarray(pred, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.shape[0] == n_quantiles:
            return arr[:, 0]
        if arr.shape[-1] == n_quantiles:
            return arr.reshape(-1, n_quantiles)[0]
        if arr.size == n_quantiles:
            return arr.reshape(-1)

        raise ValueError(
            f"Unexpected sklearn_quantile prediction shape {arr.shape}; "
            f"expected {n_quantiles} quantiles."
        )

    def _make_qrf(self, q_levels: np.ndarray):
        try:
            from sklearn_quantile import RandomForestQuantileRegressor
        except ImportError as exc:
            raise ImportError(
                "SPCI requires the official implementation's Quantile Random Forest dependency. "
                "Install it with `pip install sklearn-quantile` before running cp_mode=spci."
            ) from exc

        return RandomForestQuantileRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            max_features=self.cfg.max_features,
            min_samples_leaf=self.cfg.min_samples_leaf,
            q=list(q_levels),
            random_state=self.cfg.random_state,
        )

    def _fit_qrf(self):
        X, y = self._lagged_residual_dataset(self._residuals, self.cfg.past_window)
        if y.size < max(1, self.cfg.min_calib_size - self.cfg.past_window):
            self._model = None
            return

        self._q_levels = self._beta_levels(self.alpha, self.cfg.beta_grid)
        self._model = self._make_qrf(self._q_levels)
        self._model.fit(X, y)

    def start_test(self):
        if len(self._residuals) < self.cfg.min_calib_size:
            raise ValueError(
                f"Not enough calibration residuals for SPCI: "
                f"{len(self._residuals)} < {self.cfg.min_calib_size}."
            )

        self._resid_pool = deque(self._residuals[-self.cfg.calib_window_size:], maxlen=self.cfg.calib_window_size)
        self._fit_qrf()
        self._in_test = True
        self._t_test = 0

    def _empirical_beta_interval(self) -> Tuple[float, float]:
        pool = np.asarray(list(self._resid_pool), dtype=float)
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            return -self.cfg.fallback_width, self.cfg.fallback_width

        betas = np.linspace(0.0, self.alpha, self.cfg.beta_grid)
        lows = np.quantile(pool, betas)
        highs = np.quantile(pool, 1.0 - self.alpha + betas)
        widths = highs - lows
        idx = int(np.argmin(widths))
        return float(lows[idx]), float(highs[idx])

    def _qrf_beta_interval(self) -> Tuple[float, float]:
        if self._model is None or self._q_levels is None or len(self._residuals) < self.cfg.past_window:
            return self._empirical_beta_interval()

        x = np.asarray(self._residuals[-self.cfg.past_window:], dtype=float).reshape(1, -1)
        pred = self._model.predict(x)
        q_pred = self._format_quantile_predictions(pred, n_quantiles=len(self._q_levels))

        m = self.cfg.beta_grid
        lower_q = q_pred[:m]
        upper_q = q_pred[m:]
        widths = upper_q - lower_q
        idx = int(np.argmin(widths))
        return float(lower_q[idx]), float(upper_q[idx])

    def predict(self, base_prediction, model_uncertainty=None, **kwargs):
        mu = float(base_prediction)
        if not self._in_test:
            return mu - self.cfg.fallback_width, mu + self.cfg.fallback_width

        lo_resid, hi_resid = self._qrf_beta_interval()
        return float(mu + lo_resid), float(mu + hi_resid)

    def update(self, y_true, y_pred, prediction_interval=None, **kwargs):
        resid = float(y_true) - float(y_pred)
        if not np.isfinite(resid):
            return

        self._residuals.append(resid)
        self._resid_pool.append(resid)

        if self._in_test:
            self._t_test += 1
            if (self._t_test % self.cfg.refit_every) == 0:
                self._fit_qrf()
