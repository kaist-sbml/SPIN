import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_BiLSTM import BiLSTMRegressor, TARGET_COLS

DYN_EXTRA = ["rpm", "Feed", "delta_t"]
ALL_DYN = TARGET_COLS + DYN_EXTRA

CKPT_PATH = Path("./Models/ckpt_BiLSTM.pt")


def parse_ckpt_dims():
    return {
        "seq_len": 12,       
        "hidden_dim1": 128,
        "hidden_dim2": 64,
        "dropout": 0.2,
    }

def load_example_inputs(dynamic_csv: Path, static_csv: Path):
    df_dyn = pd.read_csv(dynamic_csv).rename(columns=str.strip)
    df_static = pd.read_csv(static_csv).rename(columns=str.strip)

    missing_dyn = [c for c in ALL_DYN if c not in df_dyn.columns]
    if missing_dyn:
        raise ValueError(f"Dynamic CSV is missing required columns: {missing_dyn}")

    if len(df_static) != 1:
        raise ValueError("Static CSV must contain exactly one row.")

    static_cols = df_static.columns.tolist()

    df_dyn[ALL_DYN] = df_dyn[ALL_DYN].apply(pd.to_numeric, errors="coerce")
    df_dyn = df_dyn.sort_values("Time").reset_index(drop=True)
    df_dyn[ALL_DYN] = df_dyn[ALL_DYN].ffill().bfill().fillna(0.0)

    static_vec = df_static.iloc[0].astype(np.float32).to_numpy()
    return df_dyn, static_vec, static_cols

def make_last_window(df_dyn: pd.DataFrame, static_vec: np.ndarray, seq_len: int) -> np.ndarray:
    if len(df_dyn) < seq_len:
        raise ValueError(f"Need at least {seq_len} dynamic rows, got {len(df_dyn)}")

    win_dyn = df_dyn.iloc[-seq_len:][ALL_DYN].to_numpy(dtype=np.float32).copy()
    static_rep = np.repeat(static_vec[None, :], seq_len, axis=0).astype(np.float32)
    x = np.hstack([win_dyn, static_rep]).astype(np.float32)
    return x

def build_model(input_dim: int):
    dims = parse_ckpt_dims()
    model = BiLSTMRegressor(
        input_dim=input_dim,
        hidden_dim1=dims["hidden_dim1"],
        hidden_dim2=dims["hidden_dim2"],
        dropout=dims["dropout"],
        out_activation="relu",
    )
    return model, dims

@torch.no_grad()
def predict_from_window(model, x_window: np.ndarray, device: torch.device) -> np.ndarray:
    # x_window shape: (Seq_len, Input_dim) -> (1, Seq_len, Input_dim)으로 변환
    xb = torch.from_numpy(x_window)[None].to(device)
    pred = model(xb).squeeze(0).cpu().numpy().astype(float)
    return pred


def search_best_rpm(model, x_window: np.ndarray, curr_rpm: float, device: torch.device):
    rpm_deltas = (-200, -150, -100, -50, 0, 50, 100, 150, 200)
    rpm_idx = ALL_DYN.index("rpm")
    bdo_idx = TARGET_COLS.index("BDO")

    base_pred = predict_from_window(model, x_window, device)
    best_bdo = float(base_pred[bdo_idx])
    best_rpm = float(curr_rpm)

    for d in rpm_deltas:
        new_rpm = curr_rpm + d
        if not (150 <= new_rpm <= 650):
            continue

        x_try = x_window.copy()
        x_try[-1, rpm_idx] = new_rpm  

        pred_try = predict_from_window(model, x_try, device)
        bdo_try = float(pred_try[bdo_idx])

        if bdo_try > best_bdo:
            best_bdo = bdo_try
            best_rpm = float(new_rpm)

    return best_rpm, best_bdo


def main(args):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    df_dyn, static_vec, static_cols = load_example_inputs(
        Path(args.dynamic_csv),
        Path(args.static_csv),
    )

    model, dims = build_model(input_dim=len(ALL_DYN) + len(static_cols))
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
    model.to(device)
    model.eval()

    x_last = make_last_window(df_dyn, static_vec, seq_len=dims["seq_len"])

    last_time = float(df_dyn.iloc[-1]["Time"])
    curr_bdo = float(df_dyn.iloc[-1]["BDO"])
    curr_rpm = float(df_dyn.iloc[-1]["rpm"])
    curr_cc = float(df_dyn.iloc[-1]["CarbonContents"])
    curr_feed = float(df_dyn.iloc[-1]["Feed"])
    
    delta_t_idx = ALL_DYN.index("delta_t")
    next_timepoint = last_time + float(x_last[-1, delta_t_idx])

    bdo_idx = TARGET_COLS.index("BDO")
    feed_idx = ALL_DYN.index("Feed")

    pred = predict_from_window(model, x_last, device)
    pred_bdo = float(pred[bdo_idx])

    print(f"Current BDO at {last_time:.1f}h: {curr_bdo:.3f}")
    print(f"Predicted BDO at {next_timepoint:.1f}h: {pred_bdo:.3f}")

    result = {
        "model": "bilstm",
        "checkpoint": str(CKPT_PATH),
        "seq_len": dims["seq_len"],
        "last_time": last_time,
        "next_timepoint": next_timepoint,
        "current_bdo": curr_bdo,
        "predicted_bdo": pred_bdo,
    }

    if pred_bdo >= curr_bdo:
        print("BDO increase predicted, RPM recommendation not performed.")
        result["message"] = "BDO increase predicted, RPM recommendation not performed."
    else:
        if (curr_cc < 20.0) and (curr_feed == 0.0):
            print("Low Carbon contents, re-predicting with additional Feeding.")
            x_feed = x_last.copy()
            x_feed[-1, feed_idx] = 50.0

            pred_feed = predict_from_window(model, x_feed, device)
            pred_bdo_feed = float(pred_feed[bdo_idx])

            result["feeding_added"] = 50.0
            result["predicted_bdo_with_feeding"] = pred_bdo_feed

            if pred_bdo_feed >= curr_bdo:
                print(f"Predicted BDO at {next_timepoint:.1f}h (with 50 g/L Feeding): {pred_bdo_feed:.3f}")
                print("BDO increase predicted, RPM recommendation not performed.")
                result["message"] = "BDO increase predicted, RPM recommendation not performed."
            else:
                print("BDO decrease predicted, RPM recommendation performed.")
                best_rpm, best_bdo = search_best_rpm(model, x_feed, curr_rpm, device)
                result["message"] = "BDO decrease predicted, RPM recommendation performed."
                result["best_rpm"] = best_rpm
                result["best_predicted_bdo"] = best_bdo
        else:
            print("BDO decrease predicted, RPM recommendation performed.")
            best_rpm, best_bdo = search_best_rpm(model, x_last, curr_rpm, device)
            result["message"] = "BDO decrease predicted, RPM recommendation performed."
            result["best_rpm"] = best_rpm
            result["best_predicted_bdo"] = best_bdo

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved prediction JSON to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic_csv", type=str, required=True)
    parser.add_argument("--static_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    
    main(args)
