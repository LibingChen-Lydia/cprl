# src/base_conformal/builder.py

from .aci_cp import ACICP
from .cqr_cp import CQRCP
from .nex_cp import NexCP
from .acp_ours import build_acp_ours
from .agaci_cp import AgACICP
from .dfpi_cp import DFPI
from .enbpi_cp import EnbPICP
from .spci_cp import SPCICP
from .cptc_cp import CPTCCP
from .hopcpt_cp import HopCPTCP
from .cpid_cp import ConformalPIDCP
from .bellman_cp import BellmanCICP

from sklearn.ensemble import RandomForestRegressor


def build_conformal_predictor(args):
    mode = getattr(args, "cp_mode", "acp")

    # ---- global safe defaults (avoid AttributeError) ----
    alpha = float(getattr(args, "alpha", 0.1))
    min_calib_size = int(getattr(args, "min_calib_size", 30))
    calib_window_size = int(getattr(args, "calib_window_size", 200))
    seed = int(getattr(args, "seed", 1103))

    if mode == "acp":
        return build_acp_ours(args)

    elif mode == "aci":
        aci_gamma = getattr(args, "aci_gamma", None)
        if aci_gamma is None:
            aci_gamma = getattr(args, "cp_lr", 0.01)
        return ACICP(
            alpha=alpha,
            T0=int(getattr(args, "aci_T0", calib_window_size)),
            min_calib_size=min_calib_size,
            gamma=float(aci_gamma),
            warm_start=int(getattr(args, "aci_warm_start", min_calib_size)),
            fallback_width=float(getattr(args, "aci_fallback_width", 3.0)),
            clip_alpha=bool(int(getattr(args, "aci_clip_alpha", 1))),
            eps=float(getattr(args, "aci_eps", 1e-6)),
            seed=seed,
        )


    elif mode == "cqr":
        return CQRCP(
            alpha=alpha,
            min_calib_size=int(getattr(args, "min_calib_size", 60)),
            split_ratio=float(getattr(args, "cqr_split_ratio", 0.5)),
            qr_l2=float(getattr(args, "cqr_qr_l2", 0.0)),
            solver=str(getattr(args, "cqr_solver", "highs")),
            standardize_x=bool(int(getattr(args, "cqr_standardize_x", 1))),
            sequential_split=bool(int(getattr(args, "cqr_sequential_split", 0))),
            fallback_width=float(getattr(args, "cqr_fallback_width", 3.0)),
            random_state=seed,
        )

    elif mode == "nex":
        return NexCP(
            alpha=alpha,
            window_size=calib_window_size,
            min_calib_size=min_calib_size,
            gamma=float(getattr(args, "nex_gamma", 0.99)),
        )

    elif mode == "agaci":
        gammas = getattr(args, "agaci_gammas", [
            0.0,
            0.000005,
            0.00005,
            0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
            0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        ])
        return AgACICP(
            alpha=alpha,
            gammas=gammas,
            warmup_steps=int(getattr(args, "agaci_warmup_steps", 50)),

            # --- align expert ACI settings with ACI baseline ---
            T0=int(getattr(args, "aci_T0", calib_window_size)),
            min_calib_size=min_calib_size,
            warm_start=int(getattr(args, "aci_warm_start", min_calib_size)),
            fallback_width=float(getattr(args, "aci_fallback_width", 3.0)),
            clip_alpha=bool(int(getattr(args, "aci_clip_alpha", 1))),
            eps=float(getattr(args, "aci_eps", 1e-6)),

            seed=seed,
        )

    elif mode == "dfpi":
        return DFPI(
            alpha=alpha,
            min_calib_size=min_calib_size,
        )

    elif mode == "enbpi":
        # 关键：这里把括号闭合写清楚，避免语法错误
        enbpi_n_estimators = int(getattr(args, "enbpi_n_estimators", 50))
        enbpi_max_depth = int(getattr(args, "enbpi_max_depth", 3))

        def _fit_func_factory():
            return RandomForestRegressor(
                n_estimators=enbpi_n_estimators,
                max_depth=enbpi_max_depth,
                bootstrap=False,
                n_jobs=-1,
                random_state=seed,
            )

        return EnbPICP(
            alpha=alpha,
            B=int(getattr(args, "enbpi_B", 25)),
            batch_size_s=int(getattr(args, "enbpi_batch_size", 30)),
            min_calib_size=min_calib_size,
            agg=str(getattr(args, "enbpi_agg", "mean")),
            beta_grid=int(getattr(args, "enbpi_beta_grid", 101)),
            random_state=seed,
            fit_func_factory=_fit_func_factory,
        )

    elif mode == "spci":
        return SPCICP(
            alpha=alpha,
            past_window=int(getattr(args, "spci_past_window", 10)),
            min_calib_size=min_calib_size,
            calib_window_size=calib_window_size,
            n_estimators=int(getattr(args, "spci_n_estimators", 100)),
            max_depth=int(getattr(args, "spci_max_depth", 5)),
            max_features=float(getattr(args, "spci_max_features", 1.0)),
            min_samples_leaf=int(getattr(args, "spci_min_samples_leaf", 1)),
            beta_grid=int(getattr(args, "spci_beta_grid", 101)),
            refit_every=int(getattr(args, "spci_refit_every", 1)),
            fallback_width=float(getattr(args, "spci_fallback_width", 3.0)),
            random_state=seed,
        )

    elif mode == "cptc":
        return CPTCCP(
            alpha=alpha,
            gamma=float(getattr(args, "cptc_gamma", 0.2)),
            min_residuals=int(getattr(args, "cptc_min_residuals", 25)),
            max_width=float(getattr(args, "cptc_max_width", 3.0)),
            prob_threshold=float(getattr(args, "cptc_prob_threshold", 0.3)),
            seed=int(getattr(args, "seed", 0)),
        )

    elif mode == "hopcpt":
        return HopCPTCP(
            alpha=alpha,
            min_calib_size=min_calib_size,
            emb_dim=int(getattr(args, "hopcpt_emb_dim", 32)),
            hidden_dim=int(getattr(args, "hopcpt_hidden_dim", 64)),
            train_epochs=int(getattr(args, "hopcpt_train_epochs", 200)),
            lr=float(getattr(args, "hopcpt_lr", 1e-3)),
            beta=float(getattr(args, "hopcpt_beta", 1.0)),
            online_update=bool(int(getattr(args, "hopcpt_online_update", 1))),
            seed=seed,
        )

    elif mode == "cpid":
        return ConformalPIDCP(
            alpha=alpha,
            kp=float(getattr(args, "cpid_kp", 0.05)),
            ki=float(getattr(args, "cpid_ki", 0.01)),
            kd=float(getattr(args, "cpid_kd", 0.0)),
            score_ema=float(getattr(args, "cpid_score_ema", 0.2)),
            score_window=int(getattr(args, "cpid_score_window", 50)),
            min_calib_size=min_calib_size,
            warm_start=int(getattr(args, "cpid_warm_start", min_calib_size)),
            fallback_width=float(getattr(args, "cpid_fallback_width", 3.0)),
            clip_alpha=bool(int(getattr(args, "cpid_clip_alpha", 1))),
            alpha_min=float(getattr(args, "cpid_alpha_min", 1e-6)),
            alpha_max=float(getattr(args, "cpid_alpha_max", 1.0 - 1e-6)),
            seed=seed,
        )

    elif mode == "bellman":
        alpha_grid = getattr(args, "bellman_alpha_grid", None)
        if alpha_grid is not None and isinstance(alpha_grid, str):
            alpha_grid = tuple(float(v) for v in alpha_grid.split(",") if v.strip())
        return BellmanCICP(
            alpha=alpha,
            alpha_grid=alpha_grid,
            horizon=int(getattr(args, "bellman_horizon", 1)),
            width_weight=float(getattr(args, "bellman_width_weight", 1.0)),
            miss_weight=float(getattr(args, "bellman_miss_weight", 3.0)),
            smooth_weight=float(getattr(args, "bellman_smooth_weight", 0.1)),
            score_window=int(getattr(args, "bellman_score_window", 100)),
            min_calib_size=min_calib_size,
            warm_start=int(getattr(args, "bellman_warm_start", min_calib_size)),
            fallback_width=float(getattr(args, "bellman_fallback_width", 3.0)),
            clip_alpha=bool(int(getattr(args, "bellman_clip_alpha", 1))),
            alpha_min=float(getattr(args, "bellman_alpha_min", 1e-6)),
            alpha_max=float(getattr(args, "bellman_alpha_max", 1.0 - 1e-6)),
            seed=seed,
        )
        
    else:
        raise ValueError(f"Unknown cp_mode: {mode}")
