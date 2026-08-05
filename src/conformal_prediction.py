# conformal_prediction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

try:
    from sklearn.linear_model import QuantileRegressor as _QR
    _HAS_QR = True
except ImportError:
    _HAS_QR = False

@dataclass
class ConformalPredictionConfig:
    # Target risk level and rolling window used by the online calibrator.
    initial_alpha: float = 0.1
    target_coverage: float = 0.9
    window_size: int = 64

    # Latent regime discovery.
    max_regimes: int = 8                
    new_regime_threshold: float = 2.2
    new_regime_patience: int = 3      

    sticky_bonus: float = 0.5             
    min_state_duration: int = 5         

    ewma_beta: float = 0.94         
    jump_q: float = 0.95            
    feature_ema: float = 0.05         

    # Calibration and trust-gating windows.
    calib_window_size: int = 150
    min_calib_size: int = 20
    min_regime_calib_size: int = 20

    min_spectral_size: int = 30

    # ACI learning-rate modulation
    aci_gamma_base: float = 0.05
    aci_spectral_beta: float = 1.0

    # spectral score cap (ensures bounded γ_t for coverage guarantee)
    spectral_score_cap: float = 2.0

    # Wasserstein reweighting (Xu et al., 2025)
    wass_reweight: bool = True
    wass_temperature: float = 0.1

    # CQR-inspired single-score conformal (Romano et al., 2019)
    use_cqr_score: bool = True
    cqr_refit_every: int = 50       # retrain QR model every N update steps
    cqr_l2: float = 0.1             # L2 regularisation for QuantileRegressor
    cqr_split_ratio: float = 0.6    # sequential-split fraction for QR training
    cqr_r_clip: float = 8.0
    cqr_x_clip_quantile: float = 0.01
    cqr_x_std_clip: float = 6.0
    unc_floor_min: float = 1e-3
    unc_floor_window: int = 128
    unc_floor_quantile: float = 0.1
    unc_floor_scale: float = 0.5

    # Residual-space regime discovery and alpha warm start.
    regime_on_residuals: bool = True
    warmstart_blend: float = 0.3

    # Minimum local coverage support before using regime-specific calibration.
    min_regime_cov_size: int = 20

    # Coverage history window for regime trust gating.
    coverage_window: int = 50

    # alpha bounds
    alpha_min: float = 0.01
    alpha_max: float = 0.3

    use_spectral: bool = True
    use_regime: bool = True
    use_adaptive_alpha: bool = True


class _SpectralDrift:
    """FFT-based Wasserstein drift statistic for adjacent error windows."""

    def __init__(self, window_size: int):
        self.window_size = int(window_size)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) == 0 or len(y) == 0:
            return 0.0

        # Convert each sub-window into a normalized power spectrum.
        fx = np.abs(np.fft.rfft(x)) ** 2
        fy = np.abs(np.fft.rfft(y)) ** 2

        sx = fx.sum()
        sy = fy.sum()
        if sx <= 1e-12 or sy <= 1e-12:
            return 0.0

        px = fx / sx
        py = fy / sy

        # One-dimensional Wasserstein distance between cumulative spectra.
        cdf_x = np.cumsum(px)
        cdf_y = np.cumsum(py)

        return float(np.mean(np.abs(cdf_x - cdf_y)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      q: float) -> float:
    """Weighted quantile with finite-sample correction.

    Implements the weighted conformal quantile from Tibshirani et al. (2019)
    Theorem 2.  The test point receives unit weight w_{n+1}=1, so the
    effective quantile level is adjusted to  q · (1 + 1 / Σ w_i).

    Parameters
    ----------
    values : 1-D array of calibration nonconformity scores.
    weights : 1-D array of non-negative importance weights.
    q : nominal quantile level in [0, 1].

    Returns
    -------
    float : the corrected weighted q-quantile.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return 0.0

    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]

    # finite-sample correction: test point weight w_{n+1} = 1
    sum_w = float(weights.sum())
    q_adj = min(q * (1.0 + 1.0 / max(sum_w, 1e-12)), 1.0)

    cum_w = np.cumsum(weights)
    cum_w /= sum_w

    j = int(np.searchsorted(cum_w, q_adj))
    j = min(j, len(values) - 1)
    return float(values[j])


class _ACIAlphaController:
    """
    Regime-aware ACI controller with spectral learning-rate modulation.

    Core update rule (Gibbs & Candès, 2021):
        α_{t+1} = α_t + γ_t · (covered_t − target_coverage)

    Learning-rate modulation:
        γ_t = γ_base · (1 + β · spectral_drift_t)

    When spectral drift is large (distribution shift detected) the step size
    increases, enabling faster adaptation.  When the spectrum is stable the
    learning rate shrinks back to γ_base, preserving the long-run marginal
    coverage guarantee of ACI.
    """

    def __init__(self, n_regimes: int, alpha_init: float,
                 cfg: ConformalPredictionConfig):
        self.cfg = cfg
        self.n_regimes = int(n_regimes)

        # per-regime alpha track
        self.alpha_per_regime = np.full(self.n_regimes, float(alpha_init),
                                        dtype=float)
        # global alpha track
        self.alpha_global = float(alpha_init)

        # Regime-transition bookkeeping for alpha warm starts.
        self.prev_rid: Optional[int] = None
        self.regime_step_counts = np.zeros(self.n_regimes, dtype=int)

        # Mondrian ACI: per-regime EWMA coverage for conditional guarantee
        self._regime_cov_ema = np.full(self.n_regimes, float(cfg.target_coverage),
                                       dtype=float)
        self._regime_cov_n = np.zeros(self.n_regimes, dtype=int)

    def _warmstart_alpha(self, rid: int) -> None:
        """Warm-start a new regime's alpha from cross-regime weighted average."""
        blend = float(getattr(self.cfg, 'warmstart_blend', 0.3))
        if blend <= 0.0:
            return
        total = int(self.regime_step_counts.sum())
        if total == 0:
            return
        weights = self.regime_step_counts.astype(float).copy()
        weights[rid] = 0.0
        w_sum = float(weights.sum())
        if w_sum <= 0:
            return
        weights /= w_sum
        cross_alpha = float(np.dot(weights, self.alpha_per_regime))
        self.alpha_per_regime[rid] = (
            (1.0 - blend) * float(self.alpha_per_regime[rid])
            + blend * cross_alpha
        )

    def choose(self, rid: int, use_regime: bool) -> float:
        """Return the current alpha (deterministic — no sampling noise)."""
        if not use_regime:
            return float(self.alpha_global)
        rid = int(max(0, min(self.n_regimes - 1, rid)))
        return float(self.alpha_per_regime[rid])

    def step(self, rid: int, covered: float, spectral_score: float,
             update_regime: bool) -> float:
        """
        Online ACI update with spectral-modulated learning rate.

        Parameters
        ----------
        rid : int
            Current regime id.
        covered : float
            1.0 if y_true fell inside the last prediction interval, else 0.0.
        spectral_score : float
            Current spectral drift score (≥ 0).
        update_regime : bool
            Whether to also update the regime-specific alpha track.

        Returns
        -------
        float
            The alpha value that should be recorded for this step.
        """
        gamma_base = float(getattr(self.cfg, 'aci_gamma_base', 0.05))
        beta = float(getattr(self.cfg, 'aci_spectral_beta', 1.0))

        # spectral-modulated learning rate (capped for bounded γ_t guarantee)
        s_cap = float(getattr(self.cfg, 'spectral_score_cap', 2.0))
        gamma = gamma_base * (1.0 + beta * min(max(0.0, float(spectral_score)), s_cap))

        target = float(self.cfg.target_coverage)
        # grad > 0  when covered (hit)  → α increases → interval narrows
        # grad < 0  when missed          → α decreases → interval widens
        grad = float(covered) - target

        a_lo = float(self.cfg.alpha_min)
        a_hi = float(self.cfg.alpha_max)

        # global track always updated
        self.alpha_global = float(np.clip(
            self.alpha_global + gamma * grad, a_lo, a_hi))

        # Mondrian ACI: per-regime EWMA coverage tracking
        rid_c = int(max(0, min(self.n_regimes - 1, rid)))
        ema_beta = 0.95
        self._regime_cov_ema[rid_c] = (
            ema_beta * self._regime_cov_ema[rid_c]
            + (1.0 - ema_beta) * float(covered))
        self._regime_cov_n[rid_c] += 1

        # Warm-start a newly entered regime if it has limited local history.
        if self.prev_rid is not None and rid_c != self.prev_rid:
            min_samples = int(getattr(self.cfg, 'min_regime_calib_size', 20))
            if self.regime_step_counts[rid_c] < min_samples:
                self._warmstart_alpha(rid_c)
        self.prev_rid = rid_c
        self.regime_step_counts[rid_c] += 1

        # per-regime track: use REGIME-LOCAL coverage for gradient (Mondrian)
        if update_regime:
            n_r = int(self._regime_cov_n[rid_c])
            if n_r >= 20:
                cov_r = float(self._regime_cov_ema[rid_c])
                grad_r = cov_r - target  # regime-conditional gradient
            else:
                grad_r = grad  # fallback to global signal
            self.alpha_per_regime[rid_c] = float(np.clip(
                self.alpha_per_regime[rid_c] + gamma * grad_r, a_lo, a_hi))
            return float(self.alpha_per_regime[rid_c])

        return float(self.alpha_global)


class _AdaptiveRegimeKernel:
    """
    Online latent-regime discovery from recent residual or value windows.

    The feature vector tracks robust volatility, EWMA volatility, jump rate,
    lag-1 dependence, and normalized trend. Sticky assignment and minimum dwell
    time reduce transient regime switching.

    Returns rid in [0, K-1], where K grows up to cfg.max_regimes.
    """
    def __init__(self, cfg: ConformalPredictionConfig, random_seed: int = 42):
        self.cfg = cfg

        self.K = 0
        self.centers: List[np.ndarray] = []
        self.counts: List[int] = []

        self.prev_rid: Optional[int] = None
        self.dwell: int = 0
        self._new_candidate_hits = 0

        self._ewma_var: Optional[float] = None
        self._ewma_var_resid: Optional[float] = None  # separate state for residual mode

        # online feature standardization (Welford)
        self._feat_n: int = 0
        self._feat_mean = np.zeros(5, dtype=float)
        self._feat_m2 = np.zeros(5, dtype=float)

    def _mad(self, x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return 0.0
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return float(1.4826 * mad + 1e-6)

    def _ewma_vol(self, x: np.ndarray, residual: bool = False) -> float:
        beta = float(self.cfg.ewma_beta)
        x = x[np.isfinite(x)]
        attr = '_ewma_var_resid' if residual else '_ewma_var'
        var = getattr(self, attr)
        if len(x) == 0:
            return float(np.sqrt(var)) if var else 0.0
        if var is None:
            var = float(np.mean(x**2) + 1e-6)
        for v in x[-min(10, len(x)):]:
            var = beta * var + (1.0 - beta) * float(v**2)
        setattr(self, attr, var)
        return float(np.sqrt(var + 1e-12))

    def _jump_rate(self, x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return 0.0
        thr = float(np.quantile(np.abs(x), float(self.cfg.jump_q)))
        if thr <= 1e-12:
            return 0.0
        return float(np.mean(np.abs(x) > thr))

    def _acf1(self, x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 6:
            return 0.0
        a = x[:-1] - np.mean(x[:-1])
        b = x[1:]  - np.mean(x[1:])
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return float(np.dot(a, b) / denom)

    def _features(self, price_window: np.ndarray) -> np.ndarray:
        """
        Optional value-window route used when residual-based regimes are disabled.
        """

        p = np.asarray(price_window, dtype=float).reshape(-1)
        p = p[np.isfinite(p)]
        if p.size < 12:
            return np.zeros(5, dtype=float)

        # log returns (robust to scale)
        eps = 1e-8
        r = np.diff(np.log(np.clip(p, eps, None)))
        r = r[np.isfinite(r)]
        if r.size < 10:
            return np.zeros(5, dtype=float)

        # 1) robust vol (MAD on returns)
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        vol_rob = 1.4826 * mad + 1e-6

        # 2) EWMA vol (on returns)
        vol_ewma = self._ewma_vol(r, residual=False)

        # 3) jump rate (on returns)
        jump = self._jump_rate(r)

        # 4) ACF1 of returns
        ac1 = self._acf1(r)

        # 5) trend slope of price level (normalized)
        t = np.arange(p.size, dtype=float)
        t = t - t.mean()
        denom = (np.dot(t, t) + 1e-6)
        slope = float(np.dot(t, (p - p.mean())) / denom)
        slope = slope / (np.std(p) + 1e-6)

        return np.array([
            np.log1p(vol_rob),
            np.log1p(vol_ewma),
            jump,
            ac1,
            slope,
        ], dtype=float)

    def _residual_features(self, resid_window: np.ndarray) -> np.ndarray:
        """
        Regime features from prediction-error (residual) window.
        Operates on raw residuals instead of log-returns.
        """
        r = np.asarray(resid_window, dtype=float).reshape(-1)
        r = r[np.isfinite(r)]
        if r.size < 12:
            return np.zeros(5, dtype=float)

        vol_rob = self._mad(r)
        vol_ewma = self._ewma_vol(r, residual=True)
        jump = self._jump_rate(r)
        ac1 = self._acf1(r)

        t = np.arange(r.size, dtype=float)
        t = t - t.mean()
        denom = (np.dot(t, t) + 1e-6)
        slope = float(np.dot(t, (r - r.mean())) / denom)
        slope = slope / (np.std(r) + 1e-6)

        return np.array([
            np.log1p(vol_rob),
            np.log1p(vol_ewma),
            jump,
            ac1,
            slope,
        ], dtype=float)

    def _dist2(self, a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))

    def _assign(self, f: np.ndarray) -> Tuple[int, float]:
        if self.K == 0:
            self.K = 1
            self.centers = [f.copy()]
            self.counts = [0]
            return 0, 0.0

        d2 = np.array([self._dist2(f, c) for c in self.centers], dtype=float)
        j = int(np.argmin(d2))
        dmin = float(np.sqrt(max(d2[j], 0.0)))
        return j, dmin

    def _standardize(self, f: np.ndarray) -> np.ndarray:
        """Online Welford standardization of feature vector."""
        self._feat_n += 1
        delta = f - self._feat_mean
        self._feat_mean += delta / self._feat_n
        delta2 = f - self._feat_mean
        self._feat_m2 += delta * delta2
        if self._feat_n < 3:
            return f  # not enough data to standardize
        std = np.sqrt(self._feat_m2 / (self._feat_n - 1) + 1e-8)
        return (f - self._feat_mean) / std

    def _update_and_get_regime(self, window: np.ndarray,
                               residual: bool = False) -> int:
        f_raw = self._residual_features(window) if residual else self._features(window)
        f = self._standardize(f_raw)
        j, dmin = self._assign(f)

        # sticky preference
        if self.prev_rid is not None and self.prev_rid < self.K:
            stay = int(self.prev_rid)
            d_stay = float(np.sqrt(max(self._dist2(f, self.centers[stay]), 0.0)))
            d_stay_eff = max(0.0, d_stay - float(self.cfg.sticky_bonus))
            if d_stay_eff <= dmin:
                j, dmin = stay, d_stay_eff

        # minimum dwell time
        if self.prev_rid is not None and self.dwell < int(self.cfg.min_state_duration):
            j = int(self.prev_rid)

        # new regime creation with patience (anti-spike)
        if (dmin > float(self.cfg.new_regime_threshold)) and (self.K < int(self.cfg.max_regimes)):
            self._new_candidate_hits += 1
            if self._new_candidate_hits >= int(self.cfg.new_regime_patience):
                j = self.K
                self.K += 1
                self.centers.append(f.copy())
                self.counts.append(0)
                self._new_candidate_hits = 0
        else:
            self._new_candidate_hits = 0

        # update center
        eta = float(self.cfg.feature_ema)
        self.centers[j] = (1 - eta) * self.centers[j] + eta * f
        self.counts[j] += 1

        # dwell update
        if self.prev_rid is None or int(j) != int(self.prev_rid):
            self.dwell = 1
        else:
            self.dwell += 1

        self.prev_rid = int(j)
        return int(j)


class AdaptiveConformalPredictor:
    def __init__(self, config: Optional[ConformalPredictionConfig] = None) -> None:
        self.config = config or ConformalPredictionConfig()

        R = int(self.config.max_regimes)

        # Main paper components: spectral drift, regime discovery, and alpha control.
        self._drift = _SpectralDrift(window_size=int(self.config.window_size))
        self._regime = _AdaptiveRegimeKernel(cfg=self.config)
        self._alpha = _ACIAlphaController(n_regimes=R, alpha_init=float(self.config.initial_alpha), cfg=self.config)

        # state
        self.current_state: Optional[int] = None
        self.state_history: List[int] = []

        # rolling errors
        self.prediction_errors: Deque[float] = deque(maxlen=int(self.config.window_size)+1)

        # Calibration buffers are maintained globally and per discovered regime.
        self._init_buffers()

        self.alpha_history: List[float] = []
        self.spectral_q_history: List[float] = []
        self.use_regime_history: List[bool] = []
        self.margin_route_history: List[str] = []

    def _init_buffers(self) -> None:
        R = int(self.config.max_regimes)
        Wc = int(self.config.calib_window_size)

        self.calib_e_lo_global: Deque[float] = deque(maxlen=Wc)
        self.calib_e_hi_global: Deque[float] = deque(maxlen=Wc)
        self.calib_s_global: Deque[float] = deque(maxlen=Wc)

        self.calib_e_lo_by_regime: Dict[int, Deque[float]] = {r: deque(maxlen=Wc) for r in range(R)}
        self.calib_e_hi_by_regime: Dict[int, Deque[float]] = {r: deque(maxlen=Wc) for r in range(R)}
        self.calib_s_by_regime: Dict[int, Deque[float]] = {r: deque(maxlen=Wc) for r in range(R)}

        self.cover_hist_global: Deque[float] = deque(maxlen=int(self.config.coverage_window))
        self.cover_hist_by_regime: Dict[int, Deque[float]] = {r: deque(maxlen=int(self.config.coverage_window)) for r in range(R)}

        # Conditional residual-quantile route.
        self._cqr_X_buf: List[np.ndarray] = []       # covariate history
        self._cqr_r_buf: List[float] = []             # signed normalized residuals
        self._cqr_E_buf: Deque[float] = deque(maxlen=Wc)  # held-out residual-quantile scores
        self._cqr_s_buf: Deque[float] = deque(maxlen=Wc)  # spectral scores aligned with E
        self._cqr_model_lo = None   # QuantileRegressor for α/2
        self._cqr_model_hi = None   # QuantileRegressor for 1-α/2
        self._cqr_x_mean: Optional[np.ndarray] = None
        self._cqr_x_std: Optional[np.ndarray] = None
        self._cqr_x_lo: Optional[np.ndarray] = None
        self._cqr_x_hi: Optional[np.ndarray] = None
        self._cqr_step: int = 0
        self._cqr_fitted: bool = False

    @property
    def alpha(self) -> float:
        if hasattr(self, "alpha_history") and len(self.alpha_history) > 0:
            return float(self.alpha_history[-1])
        return float(self.config.initial_alpha)

    def _use_regime(self, rid: int) -> bool:
        """Whether regime-specific stats are trusted (sample size gates)."""
        if not self.config.use_regime:
            return False
        rid = int(rid)
        
        ok = (
            len(self.calib_e_lo_by_regime[rid]) >= int(self.config.min_regime_calib_size)
            and len(self.calib_e_hi_by_regime[rid]) >= int(self.config.min_regime_calib_size)
            and len(self.cover_hist_by_regime[rid]) >= int(self.config.min_regime_cov_size)
        )
        if not ok:
            return False

        if bool(self.config.use_spectral):
            return len(self.calib_s_by_regime[rid]) >= int(self.config.min_regime_calib_size)

        return True

    def _buffers_global(self):
        """Return global buffers (no gating)."""
        return (
            self.calib_e_lo_global,
            self.calib_e_hi_global,
            self.calib_s_global,
            self.cover_hist_global,
            )

    def _buffers_regime(self, rid: int):
        """Return regime buffers for rid (no gating)."""
        rid = int(rid)
        return (
            self.calib_e_lo_by_regime[rid],
            self.calib_e_hi_by_regime[rid],
            self.calib_s_by_regime[rid],
            self.cover_hist_by_regime[rid],
        )

    def _effective_uncertainty(self, raw_unc: float) -> float:
        unc = float(raw_unc) if raw_unc is not None else 1.0
        floor = float(getattr(self.config, "unc_floor_min", 1e-3))

        if len(self.prediction_errors) >= 8:
            w = int(getattr(self.config, "unc_floor_window", 128))
            recent = np.asarray(list(self.prediction_errors)[-w:], dtype=float)
            recent = recent[np.isfinite(recent) & (recent > 0)]
            if len(recent) > 0:
                q = float(getattr(self.config, "unc_floor_quantile", 0.1))
                scale = float(getattr(self.config, "unc_floor_scale", 0.5))
                base = float(np.quantile(recent, q))
                if np.isfinite(base) and base > 0:
                    floor = max(floor, scale * base)

        if not np.isfinite(unc):
            unc = floor
        return float(max(unc, floor))

    def _clip_cqr_residual(self, r: float) -> float:
        clip = float(getattr(self.config, "cqr_r_clip", 8.0))
        if clip <= 0 or not np.isfinite(clip):
            return float(r)
        return float(np.clip(r, -clip, clip))

    def _transform_cqr_x(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self._cqr_x_mean is None or self._cqr_x_std is None:
            return None
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        if self._cqr_x_lo is not None and self._cqr_x_hi is not None:
            x_arr = np.clip(x_arr, self._cqr_x_lo, self._cqr_x_hi)
        xs = (x_arr - self._cqr_x_mean) / self._cqr_x_std
        std_clip = float(getattr(self.config, "cqr_x_std_clip", 6.0))
        if std_clip > 0 and np.isfinite(std_clip):
            xs = np.clip(xs, -std_clip, std_clip)
        if not np.all(np.isfinite(xs)):
            return None
        return xs
        
    # ---------- conditional residual-quantile route ----------
    def _refit_cqr(self) -> None:
        """Fit conditional residual quantiles on I1 and populate scores from I2."""
        if not _HAS_QR:
            return
        n = len(self._cqr_r_buf)
        min_n = max(int(self.config.min_calib_size) * 2, 40)
        min_e = int(self.config.min_calib_size)
        if n < min_n:
            return

        X = np.array(self._cqr_X_buf, dtype=float)
        r = np.array(self._cqr_r_buf, dtype=float)
        r = np.clip(r, -float(self.config.cqr_r_clip), float(self.config.cqr_r_clip))
        valid_rows = np.all(np.isfinite(X), axis=1) & np.isfinite(r)
        X = X[valid_rows]
        r = r[valid_rows]
        n = len(r)
        if n < min_n:
            self._cqr_fitted = False
            return

        # sequential split (time-series: no look-ahead)
        n1 = int(n * float(self.config.cqr_split_ratio))
        n1 = max(10, min(n1, n - 10))
        X1, r1 = X[:n1], r[:n1]
        X2, r2 = X[n1:], r[n1:]

        xq = float(getattr(self.config, "cqr_x_clip_quantile", 0.01))
        if 0.0 < xq < 0.5:
            self._cqr_x_lo = np.quantile(X1, xq, axis=0)
            self._cqr_x_hi = np.quantile(X1, 1.0 - xq, axis=0)
            X1 = np.clip(X1, self._cqr_x_lo, self._cqr_x_hi)
            X2 = np.clip(X2, self._cqr_x_lo, self._cqr_x_hi)
        else:
            self._cqr_x_lo = None
            self._cqr_x_hi = None

        # standardise features using I1
        self._cqr_x_mean = X1.mean(axis=0)
        self._cqr_x_std = np.maximum(X1.std(axis=0), 1e-8)
        X1s = (X1 - self._cqr_x_mean) / self._cqr_x_std
        X2s = (X2 - self._cqr_x_mean) / self._cqr_x_std
        std_clip = float(getattr(self.config, "cqr_x_std_clip", 6.0))
        if std_clip > 0 and np.isfinite(std_clip):
            X1s = np.clip(X1s, -std_clip, std_clip)
            X2s = np.clip(X2s, -std_clip, std_clip)
        if not (np.all(np.isfinite(X1s)) and np.all(np.isfinite(X2s)) and np.all(np.isfinite(r1)) and np.all(np.isfinite(r2))):
            self._cqr_fitted = False
            return

        a = float(self.config.initial_alpha)
        l2 = float(self.config.cqr_l2)
        try:
            self._cqr_model_lo = _QR(quantile=a / 2.0, alpha=l2, solver="highs")
            self._cqr_model_hi = _QR(quantile=1.0 - a / 2.0, alpha=l2, solver="highs")
            self._cqr_model_lo.fit(X1s, r1)
            self._cqr_model_hi.fit(X1s, r1)
        except Exception:
            self._cqr_fitted = False
            return
        coef_lo = np.asarray(getattr(self._cqr_model_lo, "coef_", []), dtype=float)
        coef_hi = np.asarray(getattr(self._cqr_model_hi, "coef_", []), dtype=float)
        if not (
            np.all(np.isfinite(coef_lo))
            and np.isfinite(getattr(self._cqr_model_lo, "intercept_", np.nan))
            and np.all(np.isfinite(coef_hi))
            and np.isfinite(getattr(self._cqr_model_hi, "intercept_", np.nan))
        ):
            self._cqr_fitted = False
            return

        # Held-out conformal scores for the conditional residual-quantile route.
        try:
            qlo = self._cqr_model_lo.predict(X2s)
            qhi = self._cqr_model_hi.predict(X2s)
        except Exception:
            self._cqr_fitted = False
            return
        if not (np.all(np.isfinite(qlo)) and np.all(np.isfinite(qhi))):
            self._cqr_fitted = False
            return
        E = np.maximum(qlo - r2, r2 - qhi)
        E = E[np.isfinite(E)]
        if len(E) < min_e:
            self._cqr_fitted = False
            return

        # Keep residual-quantile scores aligned with spectral scores.
        self._cqr_E_buf.clear()
        self._cqr_s_buf.clear()
        s_global = list(self.calib_s_global)
        s_offset = max(0, len(s_global) - n)  # align with _cqr_r_buf
        for i in range(len(E)):
            self._cqr_E_buf.append(float(E[i]))
            idx = s_offset + n1 + i
            sv = float(s_global[idx]) if idx < len(s_global) else 0.0
            self._cqr_s_buf.append(sv)

        self._cqr_fitted = True

    def _cqr_score_one(self, x: np.ndarray, r: float) -> Optional[float]:
        """Compute the conditional residual-quantile score for one sample."""
        if not self._cqr_fitted:
            return None
        xs = self._transform_cqr_x(x)
        if xs is None:
            self._cqr_fitted = False
            return None
        try:
            qlo = float(self._cqr_model_lo.predict(xs)[0])
            qhi = float(self._cqr_model_hi.predict(xs)[0])
        except Exception:
            self._cqr_fitted = False
            return None
        if not (np.isfinite(qlo) and np.isfinite(qhi) and np.isfinite(r)):
            self._cqr_fitted = False
            return None
        return float(max(qlo - r, r - qhi))

    # ---------- margins ----------
    def _margins_from_buffers(
        self,
        e_lo_buf,
        e_hi_buf,
        s_buf,
        alpha: float,
        model_uncertainty: float,
    ) -> Tuple[float, float, float]:
        a = float(np.clip(alpha, float(self.config.alpha_min), float(self.config.alpha_max)))

        # warm start (scores are uncertainty-normalized, so use unit scale)
        if len(e_lo_buf) < int(self.config.min_calib_size) or len(e_hi_buf) < int(self.config.min_calib_size):
            scale = 1.0 / max(1e-6, 1.0 - a)
            m = scale  # unit in normalized space; predict() multiplies by unc
            return float(m), float(m), 0.0

        e_lo_raw = np.asarray(list(e_lo_buf), float)
        e_hi_raw = np.asarray(list(e_hi_buf), float)
        n_raw = min(len(e_lo_raw), len(e_hi_raw))

        # Wasserstein reweighting (Xu et al., 2025; Barber et al., 2023)
        use_wass = bool(getattr(self.config, 'wass_reweight', False))
        n_s = len(s_buf)
        if use_wass and self.config.use_spectral and n_s >= n_raw and n_raw > 0:
            s_arr = np.asarray(list(s_buf), float)[-n_raw:]
            s_arr = np.clip(s_arr, 0.0, None)
            rev_cum = np.cumsum(s_arr[::-1])[::-1]
            drift_to_now = np.zeros(n_raw, dtype=float)
            if n_raw > 1:
                drift_to_now[:-1] = rev_cum[1:]
            temp = float(getattr(self.config, 'wass_temperature', 0.1))
            w_raw = np.exp(-temp * drift_to_now)
        else:
            w_raw = np.ones(n_raw, dtype=float)

        # Keep lower and upper scores paired when applying weights.
        mask = (np.isfinite(e_lo_raw[:n_raw])
                & np.isfinite(e_hi_raw[:n_raw]))
        e_lo = e_lo_raw[:n_raw][mask]
        e_hi = e_hi_raw[:n_raw][mask]
        w = w_raw[mask]

        if len(e_lo) == 0:
            return 1.0 / max(1e-6, 1.0 - a), 1.0 / max(1e-6, 1.0 - a), 0.0

        use_wass_active = (use_wass and self.config.use_spectral and n_s >= n_raw)

        if use_wass_active:
            q_lo = float(_weighted_quantile(e_lo, w, 1.0 - a))
            q_hi = float(_weighted_quantile(e_hi, w, 1.0 - a))
        else:
            q_lo = float(np.quantile(e_lo, 1.0 - a))
            q_hi = float(np.quantile(e_hi, 1.0 - a))

        # spectral quantile (diagnostics only)
        q_s = 0.0
        if len(s_buf) >= int(getattr(self.config, "min_spectral_size", self.config.min_calib_size)):
            s = np.asarray(list(s_buf), float)
            s = s[np.isfinite(s)]
            if len(s) > 0:
                q_s = float(np.quantile(s, 1.0 - a))

        # No additive spectral margin. Spectral information only affects the
        # predictor through drift-aware weighting and ACI learning-rate control.
        m_lo = max(0.0, q_lo)
        m_hi = max(0.0, q_hi)
        return float(m_lo), float(m_hi), float(q_s)

    def _margins_global(self, alpha: float, model_uncertainty: float) -> Tuple[float, float, float]:
        e_lo, e_hi, s, _ = self._buffers_global()
        return self._margins_from_buffers(
            e_lo,
            e_hi,
            s,
            alpha,
            model_uncertainty,
        )

    def _margins_regime(self, rid: int, alpha: float, model_uncertainty: float) -> Tuple[float, float, float]:
        e_lo, e_hi, s, _ = self._buffers_regime(rid)
        return self._margins_from_buffers(
            e_lo,
            e_hi,
            s,
            alpha,
            model_uncertainty,
        )

    def _margins_cqr(
        self, alpha: float, model_uncertainty: float, x_new
    ) -> Tuple[float, float, float]:
        """Margins from conditional residual quantiles plus conformal correction."""
        a = float(np.clip(alpha, float(self.config.alpha_min), float(self.config.alpha_max)))
        min_E = int(self.config.min_calib_size)

        # fallback if E_buf too small
        if len(self._cqr_E_buf) < min_E:
            return self._margins_global(alpha=alpha, model_uncertainty=model_uncertainty)

        # Conditional residual quantiles for the current covariates.
        x_flat = np.asarray(x_new, dtype=float).ravel()
        xs = self._transform_cqr_x(x_flat)
        if xs is None:
            self._cqr_fitted = False
            return self._margins_global(alpha=alpha, model_uncertainty=model_uncertainty)
        try:
            q_base_lo = float(self._cqr_model_lo.predict(xs)[0])
            q_base_hi = float(self._cqr_model_hi.predict(xs)[0])
        except Exception:
            self._cqr_fitted = False
            return self._margins_global(alpha=alpha, model_uncertainty=model_uncertainty)
        if not (np.isfinite(q_base_lo) and np.isfinite(q_base_hi)):
            self._cqr_fitted = False
            return self._margins_global(alpha=alpha, model_uncertainty=model_uncertainty)

        E_arr = np.asarray(list(self._cqr_E_buf), dtype=float)
        E_arr = E_arr[np.isfinite(E_arr)]
        n_E = len(E_arr)
        if n_E < min_E:
            return self._margins_global(alpha=alpha, model_uncertainty=model_uncertainty)

        # Wasserstein reweighting on CQR scores
        use_wass = bool(getattr(self.config, 'wass_reweight', False))
        if use_wass and self.config.use_spectral and len(self._cqr_s_buf) >= n_E:
            s_arr = np.asarray(list(self._cqr_s_buf), dtype=float)[-n_E:]
            s_arr = np.clip(s_arr, 0.0, None)
            rev_cum = np.cumsum(s_arr[::-1])[::-1]
            drift = np.zeros(n_E, dtype=float)
            if n_E > 1:
                drift[:-1] = rev_cum[1:]
            temp = float(getattr(self.config, 'wass_temperature', 0.1))
            w = np.exp(-temp * drift)
            Q = float(_weighted_quantile(E_arr, w, 1.0 - a))
        else:
            Q = float(np.quantile(E_arr, 1.0 - a))
        if not np.isfinite(Q):
            self._cqr_fitted = False
            return self._margins_global(alpha=alpha, model_uncertainty=model_uncertainty)

        # Asymmetric margins in normalized score space.
        m_lo = max(0.0, Q - q_base_lo)
        m_hi = max(0.0, q_base_hi + Q)

        # spectral quantile for diagnostics
        q_s = 0.0
        if len(self.calib_s_global) >= int(getattr(self.config, "min_spectral_size", self.config.min_calib_size)):
            s = np.asarray(list(self.calib_s_global), float)
            s = s[np.isfinite(s)]
            if len(s) > 0:
                q_s = float(np.quantile(s, 1.0 - a))

        return float(m_lo), float(m_hi), float(q_s)

    def _extract_price_window(self, x) -> Optional[np.ndarray]:
        if x is None:
            return None
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            return a
        if a.ndim == 2:
            return a[:, 0]
        if a.ndim == 3:
            return a[0, :, 0]
        return a.reshape(-1)

    def _has_true_cqr_context(self, x) -> bool:
        return bool(
            getattr(self.config, 'use_cqr_score', False)
            and self._cqr_fitted
            and x is not None
        )

    def _select_margin_route(self, rid: int, x) -> str:
        if self._has_true_cqr_context(x):
            return "cqr"
        if self._use_regime(rid):
            return "regime"
        return "global"

    # ============================================================
    # Public API: one-step closed loop
    # =========================================================


    def reset(self) -> None:
        self.current_state = None
        self.state_history.clear()
        self.prediction_errors.clear()
        self._init_buffers()
        if hasattr(self._regime, "reset"):
            self._regime.reset()

    def predict(
        self,
        base_prediction: float = None,
        y_pred: float = None,
        model_uncertainty: float = None,
        uncertainty: float = None,
        **kwargs,
        ):

        if getattr(self, "_pending", None) is not None:
            self._dropped_pending = getattr(self, "_dropped_pending", 0) + 1
            self._pending = None

        yp = base_prediction if base_prediction is not None else y_pred
        unc = model_uncertainty if model_uncertainty is not None else uncertainty

        yp = float(yp)
        unc = self._effective_uncertainty(unc)

        x = kwargs.get("x", None)
        if x is None:
            x = kwargs.get("X", None)
        if x is None:
            x = kwargs.get("features", None)

        # Select the current latent regime before observing y_t.
        use_resid = bool(getattr(self.config, 'regime_on_residuals', False))
        regime_window = max(8, int(self.config.window_size))
        if use_resid and len(self.prediction_errors) >= 12:
            resid_win = np.asarray(list(self.prediction_errors)[-regime_window:], float)
            rid = int(self._regime._update_and_get_regime(
                resid_win, residual=True))
        elif (not use_resid) and x is not None:
            pw = self._extract_price_window(x)
            if pw is None or pw.size < 12:
                rid = 0
            else:
                rid = int(self._regime._update_and_get_regime(pw[-regime_window:]))
        else:
            rid = 0

        rid = int(max(0, min(self.config.max_regimes - 1, rid)))
        self.current_state = rid
        self.state_history.append(rid)

        use_r = bool(self._use_regime(rid))

        if not self.config.use_adaptive_alpha:
            alpha = float(self.config.initial_alpha)
        else:
            alpha = float(self._alpha.choose(rid, use_regime=use_r))

        route = self._select_margin_route(rid, x)
        if route == "cqr":
            m_lo, m_hi, q_s = self._margins_cqr(
                alpha=alpha, model_uncertainty=unc, x_new=x)
        elif route == "regime":
            m_lo, m_hi, q_s = self._margins_regime(rid, alpha=alpha, model_uncertainty=unc)
        else:
            m_lo, m_hi, q_s = self._margins_global(alpha=alpha, model_uncertainty=unc)

        # Convert normalized margins back to the target scale.
        lower = float(yp - m_lo * unc)
        upper = float(yp + m_hi * unc)

        self._pending = {
            "rid": rid,
            "use_r": use_r,
            "yp": yp,
            "unc": unc,
            "alpha": alpha,
            "m_lo": float(m_lo),
            "m_hi": float(m_hi),
            "q_s": float(q_s),
            "lower": float(lower),
            "upper": float(upper),
            "x_raw": x,  # covariates used by the residual-quantile route
            "route": route,
        }

        self._last_alpha_sampled = float(alpha)

        return lower, upper
    
    def update(
        self,
        y_true: float = None,
        y: float = None,
        y_obs: float = None,
        **kwargs,
    ):

        yt = y_true if y_true is not None else (y if y is not None else y_obs)
        yt = float(yt)

        p = getattr(self, "_pending", None)

        if p is None:
            return None

        rid = int(p["rid"])
        use_r_pred = bool(p["use_r"])
        yp = float(p["yp"])
        alpha = float(p["alpha"])
        lower = float(p["lower"])
        upper = float(p["upper"])
        q_s = float(p["q_s"])

        e_lo_raw = max(0.0, yp - yt)
        e_hi_raw = max(0.0, yt - yp)
        err = float(e_lo_raw + e_hi_raw)

        # normalize conformal scores by uncertainty (Lei et al., 2018)
        unc = self._effective_uncertainty(float(p["unc"]))
        e_lo = float(e_lo_raw) / unc
        e_hi = float(e_hi_raw) / unc

        self.calib_e_lo_global.append(float(e_lo))
        self.calib_e_hi_global.append(float(e_hi))
        self.calib_e_lo_by_regime[rid].append(float(e_lo))
        self.calib_e_hi_by_regime[rid].append(float(e_hi))

        covered = 1.0 if (lower <= yt <= upper) else 0.0
        self.cover_hist_global.append(float(covered))
        self.cover_hist_by_regime[rid].append(float(covered))

        self.prediction_errors.append(float(err))
        if self.config.use_spectral:
            s = 0.0
            if len(self.prediction_errors) >= int(self.config.window_size):
                window = np.asarray(list(self.prediction_errors)[-int(self.config.window_size):], float)
                half = len(window) // 2
                s = float(self._drift.score(window[:half], window[half:]))
            self.calib_s_global.append(s)
            self.calib_s_by_regime[rid].append(float(s))

        use_r_now = bool(self._use_regime(rid))

        if self.config.use_adaptive_alpha:
            s_now = float(self.calib_s_global[-1]) if (
                self.config.use_spectral and len(self.calib_s_global) > 0) else 0.0
            alpha_state = float(self._alpha.step(
                rid, covered, s_now, update_regime=use_r_now))
        else:
            alpha_state = float(alpha)

        self.alpha_history.append(alpha_state)

        self.spectral_q_history.append(float(q_s))
        self.use_regime_history.append(bool(use_r_pred))
        self.margin_route_history.append(str(p.get("route", "unknown")))

        # Update the conditional residual-quantile route after observing y_t.
        use_cqr = bool(getattr(self.config, 'use_cqr_score', False)) and _HAS_QR
        x_raw = p.get("x_raw", None)
        if use_cqr and x_raw is not None:
            r_signed = self._clip_cqr_residual(float(e_hi - e_lo))
            x_flat = np.asarray(x_raw, dtype=float).ravel()
            self._cqr_X_buf.append(x_flat)
            self._cqr_r_buf.append(r_signed)
            Wc = int(self.config.calib_window_size)
            if len(self._cqr_X_buf) > Wc:
                self._cqr_X_buf = self._cqr_X_buf[-Wc:]
                self._cqr_r_buf = self._cqr_r_buf[-Wc:]

            # If fitted, add the new held-out score before periodic refitting.
            if self._cqr_fitted:
                E_new = self._cqr_score_one(x_flat, r_signed)
                if E_new is not None:
                    self._cqr_E_buf.append(E_new)
                    s_last = float(self.calib_s_global[-1]) if len(self.calib_s_global) > 0 else 0.0
                    self._cqr_s_buf.append(s_last)

            # periodic refit
            self._cqr_step += 1
            if self._cqr_step % int(self.config.cqr_refit_every) == 0:
                self._refit_cqr()

        self._pending = None

        return float(lower), float(upper)

    def start_test(self):
        """Interface hook used by the experiment runner."""
        return
