import json
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

TARGET_COLS = ["OD", "CarbonContents", "Ethanol", "Acetoin", "BDO"]
DYN_EXTRA = ["rpm", "Feed", "delta_t"]
ALL_DYN = TARGET_COLS + DYN_EXTRA
CKPT_PATH = Path("./Models/ckpt_BiLSTM.keras")

def load_example_inputs(dynamic_csv: Path, static_csv: Path):
    df_dyn = pd.read_csv(dynamic_csv).rename(columns=str.strip)
    df_static = pd.read_csv(static_csv).rename(columns=str.strip)
    missing_dyn = [c for c in ALL_DYN if c not in df_dyn.columns]
    if missing_dyn:
        raise ValueError(f"Missing columns: {missing_dyn}")
    df_dyn[ALL_DYN] = df_dyn[ALL_DYN].apply(pd.to_numeric, errors="coerce")
    df_dyn = df_dyn.sort_values("Time").reset_index(drop=True).ffill().bfill().fillna(0.0)
    static_vec = df_static.iloc[0].astype(np.float32).to_numpy()
    return df_dyn, static_vec, df_static.columns.tolist()

def make_last_window(df_dyn, static_vec, seq_len):
    win_dyn = df_dyn.iloc[-seq_len:][ALL_DYN].to_numpy(dtype=np.float32)
    static_rep = np.repeat(static_vec[None, :], seq_len, axis=0)
    return np.hstack([win_dyn, static_rep]).astype(np.float32)

def predict_from_window(model, x_window):
    xb = x_window[np.newaxis, ...]
    return model.predict(xb, verbose=0)[0]

def search_best_rpm(model, x_window, curr_rpm):
    rpm_deltas = (-200, -150, -100, -50, 0, 50, 100, 150, 200)
    rpm_idx = ALL_DYN.index("rpm")
    bdo_idx = TARGET_COLS.index("BDO")
    base_pred = predict_from_window(model, x_window)
    best_bdo = float(base_pred[bdo_idx])
    best_rpm = float(curr_rpm)
    for d in rpm_deltas:
        new_rpm = curr_rpm + d
        if not (150 <= new_rpm <= 650):
            continue
        x_try = x_window.copy()
        x_try[-1, rpm_idx] = new_rpm
        pred_try = predict_from_window(model, x_try)
        if pred_try[bdo_idx] > best_bdo:
            best_bdo = float(pred_try[bdo_idx])
            best_rpm = float(new_rpm)
    return best_rpm, best_bdo

def main(args):
    model = tf.keras.models.load_model(CKPT_PATH, compile=False)
    df_dyn, static_vec, static_cols = load_example_inputs(Path(args.dynamic_csv), Path(args.static_csv))
    seq_len = 12
    x_last = make_last_window(df_dyn, static_vec, seq_len)
    last_time = float(df_dyn.iloc[-1]["Time"])
    curr_bdo = float(df_dyn.iloc[-1]["BDO"])
    curr_rpm = float(df_dyn.iloc[-1]["rpm"])
    curr_cc = float(df_dyn.iloc[-1]["CarbonContents"])
    curr_feed = float(df_dyn.iloc[-1]["Feed"])
    pred = predict_from_window(model, x_last)
    pred_bdo = float(pred[TARGET_COLS.index("BDO")])
    result = {"model": "bilstm", "last_time": last_time, "current_bdo": curr_bdo, "predicted_bdo": pred_bdo}
    if pred_bdo < curr_bdo:
        if curr_cc < 20.0 and curr_feed == 0.0:
            x_feed = x_last.copy()
            x_feed[-1, ALL_DYN.index("Feed")] = 50.0
            pred_feed = predict_from_window(model, x_feed)
            pred_bdo_feed = float(pred_feed[TARGET_COLS.index("BDO")])
            if pred_bdo_feed < curr_bdo:
                best_rpm, best_bdo = search_best_rpm(model, x_feed, curr_rpm)
                result.update({"best_rpm": best_rpm, "best_predicted_bdo": best_bdo})
            else:
                result["predicted_bdo_with_feeding"] = pred_bdo_feed
        else:
            best_rpm, best_bdo = search_best_rpm(model, x_last, curr_rpm)
            result.update({"best_rpm": best_rpm, "best_predicted_bdo": best_bdo})
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic_csv", type=str, required=True)
    parser.add_argument("--static_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str)
    args = parser.parse_args()
    main(args)
