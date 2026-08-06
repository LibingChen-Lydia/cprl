#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TSL_ROOT = ROOT / "time_series_library"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TSL_ROOT) not in sys.path:
    sys.path.insert(0, str(TSL_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cprl_mpl")

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a time_series_library forecaster and export forecast_full.npz."
    )

    parser.add_argument("--task_name", default="long_term_forecast")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="custom")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features", default="MS")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--checkpoints", default="./checkpoints/")
    parser.add_argument("--cache_root", default=None)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--seasonal_patterns", default="Monthly")
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--enc_in", type=int, default=1)
    parser.add_argument("--dec_in", type=int, default=1)
    parser.add_argument("--c_out", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--decomp_method", default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--down_sampling_method", default=None)
    parser.add_argument("--seg_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=16)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--des", default=None)
    parser.add_argument("--loss", default="MSE")
    parser.add_argument("--lradj", default="type1")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--use_dtw", type=bool, default=False)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", default="0,1,2,3")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setting_name(args: argparse.Namespace, itr_index: int = 0) -> str:
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}"
        f"_df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}"
        f"_eb{args.embed}_dt{args.distil}_{args.des}_{itr_index}"
    )


def ordered_loader(exp: Exp_Long_Term_Forecast, flag: str) -> tuple[object, DataLoader]:
    data_set, _ = exp._get_data(flag=flag)
    loader = DataLoader(
        data_set,
        batch_size=exp.args.batch_size,
        shuffle=False,
        num_workers=exp.args.num_workers,
        drop_last=False,
    )
    return data_set, loader


def aggregate_split(exp: Exp_Long_Term_Forecast, flag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_set, loader = ordered_loader(exp, flag)
    pred_len = int(exp.args.pred_len)
    seq_len = int(exp.args.seq_len)
    n_unique = len(data_set) + pred_len - 1

    pred_sum = np.zeros(n_unique, dtype=np.float64)
    true_sum = np.zeros(n_unique, dtype=np.float64)
    count = np.zeros(n_unique, dtype=np.int64)

    exp.model.eval()
    sample_offset = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_size = batch_x.shape[0]
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            if exp.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if exp.args.features == "MS" else 0
            outputs = outputs[:, -pred_len:, :]
            truth = batch_y[:, -pred_len:, :]
            outputs_np = outputs.detach().cpu().numpy()
            truth_np = truth.detach().cpu().numpy()

            if data_set.scale and exp.args.inverse:
                shape = truth_np.shape
                if outputs_np.shape[-1] != truth_np.shape[-1]:
                    outputs_np = np.tile(outputs_np, [1, 1, int(truth_np.shape[-1] / outputs_np.shape[-1])])
                outputs_np = data_set.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                truth_np = data_set.inverse_transform(truth_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

            outputs_np = outputs_np[:, :, f_dim:].reshape(batch_size, pred_len)
            truth_np = truth_np[:, :, f_dim:].reshape(batch_size, pred_len)

            for row in range(batch_size):
                start = sample_offset + row
                for horizon in range(pred_len):
                    out_idx = start + horizon
                    pred_sum[out_idx] += float(outputs_np[row, horizon])
                    true_sum[out_idx] += float(truth_np[row, horizon])
                    count[out_idx] += 1
            sample_offset += batch_size

    keep = count > 0
    y_pred = pred_sum[keep] / count[keep]
    y_true = true_sum[keep] / count[keep]
    time_idx = np.arange(seq_len, seq_len + n_unique, dtype=np.int64)[keep]
    return y_true.astype(np.float32), y_pred.astype(np.float32), time_idx, count[keep]


def cache_path(args: argparse.Namespace, setting: str) -> Path:
    root = Path(args.cache_root) if args.cache_root else ROOT / f"forecast_cache_seed{args.seed}"
    return root / setting / "forecast_full.npz"


def main() -> int:
    args = parse_args()
    if args.des is None:
        args.des = args.model_id.replace("_cache", "")
    if args.gpu_type == "cpu":
        args.use_gpu = False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(x) for x in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    set_seed(args.seed)
    setting = setting_name(args, 0)
    exp = Exp_Long_Term_Forecast(args)

    print(f">>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.train(setting)

    print(f">>>>>>> exporting cache : {setting} <<<<<<<<<<<<<<<<<<<<<<<<")
    val_y_true, val_y_pred, val_time_idx, val_count = aggregate_split(exp, "val")
    test_y_true, test_y_pred, test_time_idx, test_count = aggregate_split(exp, "test")

    out_path = cache_path(args, setting)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        val_y_true_full=val_y_true,
        val_y_pred_full=val_y_pred,
        val_time_idx=val_time_idx,
        val_count=val_count,
        test_y_true_full=test_y_true,
        test_y_pred_full=test_y_pred,
        test_time_idx=test_time_idx,
        test_count=test_count,
    )
    print(f"[Cache] saved to: {out_path}")
    print(f"[Cache] val length={len(val_y_true)} count_max={int(val_count.max()) if len(val_count) else 0}")
    print(f"[Cache] test length={len(test_y_true)} count_max={int(test_count.max()) if len(test_count) else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
