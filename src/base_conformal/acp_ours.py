# src/base_conformal/acp_ours.py

from src.conformal_prediction import ConformalPredictionConfig, AdaptiveConformalPredictor

def build_acp_ours(args) -> AdaptiveConformalPredictor:
    cfg = ConformalPredictionConfig(
        initial_alpha=float(getattr(args, "alpha", 0.1)),
        target_coverage=float(getattr(args, "target_coverage", 1.0 - float(getattr(args, "alpha", 0.1)))),
        window_size=int(getattr(args, "spectral_window", getattr(args, "window_size", 64))),

        max_regimes=int(getattr(args, "max_regimes", 8)),
        new_regime_threshold=float(getattr(args, "new_regime_threshold", 2.2)),
        new_regime_patience=int(getattr(args, "new_regime_patience", 3)),
        sticky_bonus=float(getattr(args, "sticky_bonus", 0.5)),
        min_state_duration=int(getattr(args, "min_state_duration", 5)),
        ewma_beta=float(getattr(args, "ewma_beta", 0.94)),
        jump_q=float(getattr(args, "jump_q", 0.95)),
        feature_ema=float(getattr(args, "feature_ema", 0.05)),

        calib_window_size=int(getattr(args, "calib_window_size", 200)),
        min_calib_size=int(getattr(args, "min_calib_size", 30)),
        min_regime_calib_size=int(getattr(args, "min_regime_calib_size", 50)),
        min_regime_cov_size=int(getattr(args, "min_regime_cov_size", 30)),

        coverage_window=int(getattr(args, "coverage_window", 50)),

        # ACI + spectral modulation
        aci_gamma_base=float(getattr(args, "aci_gamma_base", 0.05)),
        aci_spectral_beta=float(getattr(args, "aci_spectral_beta", 1.0)),
        spectral_score_cap=float(getattr(args, "spectral_score_cap", 2.0)),

        # Wasserstein reweighting
        wass_reweight=bool(getattr(args, "wass_reweight", True)),
        wass_temperature=float(getattr(args, "wass_temperature", 0.1)),

        # CQR-inspired single-score conformal with online QR
        use_cqr_score=bool(getattr(args, "use_cqr_score", True)),
        cqr_refit_every=int(getattr(args, "cqr_refit_every", 50)),
        cqr_l2=float(getattr(args, "cqr_l2", 0.1)),
        cqr_split_ratio=float(getattr(args, "cqr_split_ratio", 0.6)),
        cqr_r_clip=float(getattr(args, "cqr_r_clip", 8.0)),
        cqr_x_clip_quantile=float(getattr(args, "cqr_x_clip_quantile", 0.01)),
        cqr_x_std_clip=float(getattr(args, "cqr_x_std_clip", 6.0)),
        unc_floor_min=float(getattr(args, "unc_floor_min", 1e-3)),
        unc_floor_window=int(getattr(args, "unc_floor_window", 128)),
        unc_floor_quantile=float(getattr(args, "unc_floor_quantile", 0.1)),
        unc_floor_scale=float(getattr(args, "unc_floor_scale", 0.5)),

        # residual-space regime + warm-start
        regime_on_residuals=bool(getattr(args, "regime_on_residuals", True)),
        warmstart_blend=float(getattr(args, "warmstart_blend", 0.3)),

        alpha_min=float(getattr(args, "alpha_min", 0.01)),
        alpha_max=float(getattr(args, "alpha_max", 0.3)),
    )

    mode = str(getattr(args, "ablation_mode", "M0")).upper()

    cfg.use_spectral = True
    cfg.use_regime   = True
    cfg.use_adaptive_alpha = bool(getattr(args, "adaptive_alpha", True))

    if mode == "M0":
        pass
    elif mode == "M1":  # no spectral
        cfg.use_spectral = False
        cfg.wass_reweight = False
    elif mode == "M2":  # no regime
        cfg.use_regime = False
        cfg.regime_on_residuals = False
    elif mode == "M3":  # no adaptive alpha control (fixed alpha)
        cfg.use_adaptive_alpha = False
    elif mode == "M4":  # no Wasserstein reweighting
        cfg.wass_reweight = False
    elif mode == "M5":  # no conditional residual-quantile calibration route
        cfg.use_cqr_score = False
    else:
        raise ValueError(f"Unknown ablation_mode: {mode}")

    return AdaptiveConformalPredictor(config=cfg)
