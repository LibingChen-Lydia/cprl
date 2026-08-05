# src/base_conformal/agaci_cp.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

from .aci_cp import ACICP

Number = Union[int, float]


def pinball_loss(y: float, q: float, beta: float) -> float:
    """
    Pinball loss:
      ρ_β(u) = β*u - min(u, 0), where u = y - q.
    """
    u = float(y - q)
    return float(beta * u - min(u, 0.0))


@dataclass
class AgACIConfig:
    alpha: float = 0.1
    gammas: Tuple[float, ...] = (
        0.0,
        0.000005,
        0.00005,
        0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
        0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    )
    warmup_steps: int = 50

    # ---- expert (time-series ACI) params ----
    T0: int = 500
    min_calib_size: int = 30
    warm_start: int = 50
    fallback_width: float = 3.0
    clip_alpha: bool = False
    eps: float = 1e-6

    # ---- bounded thresholding domain (only applied to finite expert outputs) ----
    M_lower: float = -1e9
    M_upper: float = 1e9

    # ---- BOA stability ----
    agg_eps: float = 1e-12
    clip_eta_max: float = 50.0
    loss_clip: float = 1e6
    eta_floor: float = 1e-8

    seed: int = 0


class _BOAStyleAggregator:
    """
    BOA-style second-order online aggregation (one side: lower OR upper).
    Updates weights with centered losses to improve stability.
    """
    def __init__(
        self,
        n_experts: int,
        eps: float = 1e-12,
        clip_eta_max: float = 50.0,
        loss_clip: float = 1e6,
        eta_floor: float = 1e-8,
    ):
        self.K = int(n_experts)
        self.eps = float(eps)
        self.clip_eta_max = float(clip_eta_max)
        self.loss_clip = float(loss_clip)
        self.eta_floor = float(eta_floor)

        self.w = np.full(self.K, 1.0 / self.K, dtype=float)
        self.cum_sq = np.zeros(self.K, dtype=float)
        self.cum_l = np.zeros(self.K, dtype=float)
        self.max_abs = np.zeros(self.K, dtype=float)
        self.eta = np.zeros(self.K, dtype=float)

    def reset(self):
        self.w[:] = 1.0 / self.K
        self.cum_sq[:] = 0.0
        self.cum_l[:] = 0.0
        self.max_abs[:] = 0.0
        self.eta[:] = 0.0

    def update(self, losses: np.ndarray):
        losses = np.asarray(losses, dtype=float).reshape(-1)
        if losses.shape[0] != self.K:
            raise ValueError(f"losses shape {losses.shape} != (K,) with K={self.K}")

        # safety clip (engineering)
        if np.isfinite(self.loss_clip) and self.loss_clip > 0:
            losses = np.clip(losses, -self.loss_clip, self.loss_clip)

        # centered losses
        loss_bar = float(np.sum(self.w * losses))
        ell = losses - loss_bar  # (K,)

        # second-order stats
        self.cum_sq += ell ** 2
        self.max_abs = np.maximum(self.max_abs, np.abs(ell))

        # BOA-like bounding term e_k
        e_vals = 2.0 ** (np.ceil(np.log2(self.max_abs + self.eps)) + 1.0)

        # adaptive eta_k
        logK = np.log(self.K + 1.0)
        eta = np.minimum(
            1.0 / (e_vals + self.eps),
            np.sqrt(logK / (self.cum_sq + self.eps)),
        )
        eta = np.clip(eta, 0.0, self.clip_eta_max)
        eta = np.maximum(eta, self.eta_floor)
        self.eta = eta

        # BOA-like l-values update
        self.cum_l += 0.5 * (ell * (1.0 + eta * ell) + e_vals * (eta * ell > 0.5))

        # weights update: w_k ∝ eta_k * exp(- eta_k * L_k)
        z = -eta * self.cum_l
        z -= np.max(z)  # stabilize
        w_unnorm = eta * np.exp(z)
        s = float(np.sum(w_unnorm))
        if s <= 0 or not np.isfinite(s):
            self.w[:] = 1.0 / self.K
        else:
            self.w[:] = w_unnorm / (s + self.eps)


class AgACICP:
    """
    AgACI for time series:
      - K time-series ACI experts (ACICP) with different gammas.
      - Aggregate lower/upper bounds with BOA-style online weights.
      - Use pinball losses with beta_low=alpha/2, beta_up=1-alpha/2.

    IMPORTANT:
      - We cache expert bounds from predict() and reuse them in update().
      - Experts producing non-finite bounds (±inf) are masked out from aggregation
        (do NOT replace by huge numbers, which would explode the average).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gammas: Optional[List[float]] = None,
        warmup_steps: int = 50,

        # expert ACI params
        T0: int = 500,
        min_calib_size: int = 30,
        warm_start: int = 50,
        fallback_width: float = 3.0,
        clip_alpha: bool = False,
        eps: float = 1e-6,

        # bounding (only for finite values)
        M_lower: float = -1e9,
        M_upper: float = 1e9,

        # aggregator stability
        agg_eps: float = 1e-12,
        clip_eta_max: float = 50.0,
        loss_clip: float = 1e6,
        eta_floor: float = 1e-8,

        seed: int = 0,
        **kwargs,
    ):
        self.cfg = AgACIConfig(
            alpha=float(alpha),
            gammas=tuple(gammas) if gammas is not None else AgACIConfig().gammas,
            warmup_steps=int(warmup_steps),

            T0=int(T0),
            min_calib_size=int(min_calib_size),
            warm_start=int(warm_start),
            fallback_width=float(fallback_width),
            clip_alpha=bool(clip_alpha),
            eps=float(eps),

            M_lower=float(M_lower),
            M_upper=float(M_upper),

            agg_eps=float(agg_eps),
            clip_eta_max=float(clip_eta_max),
            loss_clip=float(loss_clip),
            eta_floor=float(eta_floor),

            seed=int(seed),
        )

        self.initial_alpha = float(alpha)
        self.alpha = float(alpha)  # AgACI doesn't adapt alpha; keep for logging

        self.gammas = list(self.cfg.gammas)
        self.K = len(self.gammas)

        # Experts: time-series ACI with different step sizes (gamma)
        self.experts: List[ACICP] = []
        for k, g in enumerate(self.gammas):
            self.experts.append(
                ACICP(
                    alpha=self.cfg.alpha,
                    lr=float(g),  # in ACICP: lr is gamma
                    T0=self.cfg.T0,
                    min_calib_size=self.cfg.min_calib_size,
                    warm_start=self.cfg.warm_start,
                    fallback_width=self.cfg.fallback_width,
                    clip_alpha=self.cfg.clip_alpha,
                    eps=self.cfg.eps,
                    seed=self.cfg.seed + 97 * (k + 1),
                )
            )

        self.agg_low = _BOAStyleAggregator(
            self.K,
            eps=self.cfg.agg_eps,
            clip_eta_max=self.cfg.clip_eta_max,
            loss_clip=self.cfg.loss_clip,
            eta_floor=self.cfg.eta_floor,
        )
        self.agg_up = _BOAStyleAggregator(
            self.K,
            eps=self.cfg.agg_eps,
            clip_eta_max=self.cfg.clip_eta_max,
            loss_clip=self.cfg.loss_clip,
            eta_floor=self.cfg.eta_floor,
        )

        # pinball beta for lower/upper
        self.beta_low = self.cfg.alpha / 2.0
        self.beta_up = 1.0 - self.cfg.alpha / 2.0

        self.t = 0

        # cache from last predict()
        self._last_expert_lows: Optional[np.ndarray] = None
        self._last_expert_ups: Optional[np.ndarray] = None
        self._last_finite_mask: Optional[np.ndarray] = None
        self._last_mu: Optional[float] = None
        self._last_unc: Optional[float] = None

    def initialize(self, initial_data=None):
        self.alpha = self.initial_alpha
        self.t = 0
        self.agg_low.reset()
        self.agg_up.reset()

        self._last_expert_lows = None
        self._last_expert_ups = None
        self._last_finite_mask = None
        self._last_mu = None
        self._last_unc = None

        for e in self.experts:
            e.initialize(initial_data=initial_data)

    def _fallback_interval(self, mu: float, unc: float) -> Tuple[float, float]:
        w = self.cfg.fallback_width * max(1e-12, float(unc))
        return float(mu - w), float(mu + w)

    def predict(
        self,
        base_prediction=None,
        model_uncertainty: Number = 1.0,
        x=None,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Return aggregated interval (low_agg, up_agg).
        Cache expert bounds for update().
        """
        mu = float(base_prediction) if base_prediction is not None else 0.0
        unc = float(model_uncertainty)

        use_uniform = (self.t < self.cfg.warmup_steps)
        w_low = np.full(self.K, 1.0 / self.K, dtype=float) if use_uniform else self.agg_low.w.copy()
        w_up = np.full(self.K, 1.0 / self.K, dtype=float) if use_uniform else self.agg_up.w.copy()

        lows = np.full(self.K, np.nan, dtype=float)
        ups = np.full(self.K, np.nan, dtype=float)
        finite_mask = np.zeros(self.K, dtype=bool)

        for k, e in enumerate(self.experts):
            l_k, u_k = e.predict(
                base_prediction=mu,
                model_uncertainty=unc,
                x=x,
            )

            # IMPORTANT: do not replace inf with huge numbers; mask them out.
            if (not np.isfinite(l_k)) or (not np.isfinite(u_k)):
                continue

            l_k = float(np.clip(l_k, self.cfg.M_lower, self.cfg.M_upper))
            u_k = float(np.clip(u_k, self.cfg.M_lower, self.cfg.M_upper))

            lows[k] = l_k
            ups[k] = u_k
            finite_mask[k] = True

        # cache for update()
        self._last_expert_lows = lows.copy()
        self._last_expert_ups = ups.copy()
        self._last_finite_mask = finite_mask.copy()
        self._last_mu = mu
        self._last_unc = unc

        # if no finite experts, fallback
        if not np.any(finite_mask):
            return self._fallback_interval(mu, unc)

        # renormalize weights on finite experts only
        w_low[~finite_mask] = 0.0
        w_up[~finite_mask] = 0.0
        w_low = w_low / (float(w_low.sum()) + 1e-12)
        w_up = w_up / (float(w_up.sum()) + 1e-12)

        low_agg = float(np.nansum(w_low * lows))
        up_agg = float(np.nansum(w_up * ups))

        # final sanity: ensure order
        if low_agg > up_agg:
            low_agg, up_agg = up_agg, low_agg

        return low_agg, up_agg

    def update(
        self,
        y_true,
        y_pred=None,
        prediction_interval=None,
        interval=None,
        x=None,
        model_uncertainty: Number = 1.0,
        **kwargs,
    ):
        """
        Update:
          1) experts (time-series ACI needs y_pred for residual score)
          2) aggregators via pinball losses

        Uses cached expert bounds from the preceding predict().
        """
        if prediction_interval is None:
            prediction_interval = interval
        if prediction_interval is None:
            raise ValueError("AgACICP.update requires prediction_interval (or interval).")
        if y_pred is None and self._last_mu is None:
            raise ValueError("AgACICP.update requires y_pred (or prior predict cache) for ACI experts.")

        yt = float(y_true)

        # enforce same-step context (prefer cache)
        mu = float(self._last_mu) if self._last_mu is not None else float(y_pred)
        unc = float(self._last_unc) if self._last_unc is not None else float(model_uncertainty)

        if self._last_expert_lows is None or self._last_expert_ups is None or self._last_finite_mask is None:
            # fallback: compute once (but do NOT do it twice)
            lows = np.full(self.K, np.nan, dtype=float)
            ups = np.full(self.K, np.nan, dtype=float)
            finite_mask = np.zeros(self.K, dtype=bool)
            for k, e in enumerate(self.experts):
                l_k, u_k = e.predict(base_prediction=mu, model_uncertainty=unc, x=x)
                if (not np.isfinite(l_k)) or (not np.isfinite(u_k)):
                    continue
                lows[k] = float(np.clip(l_k, self.cfg.M_lower, self.cfg.M_upper))
                ups[k] = float(np.clip(u_k, self.cfg.M_lower, self.cfg.M_upper))
                finite_mask[k] = True
        else:
            lows = self._last_expert_lows
            ups = self._last_expert_ups
            finite_mask = self._last_finite_mask

        # 1) update experts (only finite ones have meaningful bounds)
        for k, e in enumerate(self.experts):
            if not bool(finite_mask[k]):
                # even if expert interval is infinite, we can still update it using the observed miss
                # but we must provide a valid interval; use fallback to avoid NaN
                l_k, u_k = self._fallback_interval(mu, unc)
            else:
                l_k, u_k = float(lows[k]), float(ups[k])

            e.update(
                y_true=yt,
                y_pred=mu,
                prediction_interval=(l_k, u_k),
                x=x,
            )

        # 2) pinball losses for aggregation (only for finite experts)
        loss_low = np.zeros(self.K, dtype=float)
        loss_up = np.zeros(self.K, dtype=float)
        for k in range(self.K):
            if not bool(finite_mask[k]):
                # assign neutral-ish loss; we set to 0 so it won't dominate centered loss
                # (the mask will also effectively drop it due to weight=0 after normalization at predict time)
                loss_low[k] = 0.0
                loss_up[k] = 0.0
            else:
                loss_low[k] = pinball_loss(yt, q=float(lows[k]), beta=self.beta_low)
                loss_up[k] = pinball_loss(yt, q=float(ups[k]), beta=self.beta_up)

        # 3) update aggregators after warmup
        if self.t >= self.cfg.warmup_steps:
            self.agg_low.update(loss_low)
            self.agg_up.update(loss_up)

        self.t += 1
        self.alpha = self.initial_alpha  # keep for logging

        # clear cache to avoid accidental reuse across steps
        self._last_expert_lows = None
        self._last_expert_ups = None
        self._last_finite_mask = None
        self._last_mu = None
        self._last_unc = None
