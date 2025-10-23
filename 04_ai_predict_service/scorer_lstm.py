# 04_ai_predict_service/scorer_lstm.py
import numpy as np
import pandas as pd
from .features import add_time_features, encode_with_vocab

def build_windows(df: pd.DataFrame, scaler, window_size=3):
    # 모델 입력: event_type_id, location_id_id, delta_t_sec, hour_sin, hour_cos
    use = ["event_type_id", "location_id_id", "delta_t_sec", "hour_sin", "hour_cos"]
    df = df.copy()
    df["delta_t_sec"] = scaler.transform(df[["delta_t_sec"]]).ravel()
    Xs, ys = [], []
    for _, g in df.sort_values(["epc_code", "event_time"]).groupby("epc_code"):
        feats = g[use].to_numpy(np.float32)
        tgt = g["event_type_id"].to_numpy(np.int64)
        W = window_size
        for i in range(len(g) - W):
            Xs.append(feats[i:i+W])
            ys.append(int(tgt[i+W]))
    if not Xs:
        return np.empty((0, W, 5), np.float32), np.empty((0,), np.int64)
    return np.stack(Xs, 0), np.array(ys, np.int64)

def score_lstm(df_raw: pd.DataFrame, arts, window_size=3):
    if not (arts.model and arts.scaler and arts.vocab_event and arts.vocab_loc):
        raise RuntimeError("Artifacts are not loaded.")
    df = add_time_features(df_raw)
    df["event_type_id"] = encode_with_vocab(df["event_type"], arts.vocab_event)
    df["location_id_id"] = encode_with_vocab(df["location_id"].astype(str), arts.vocab_loc)
    X, y = build_windows(df, arts.scaler, window_size)
    if X.shape[0] == 0:
        return [], float('nan'), []
    probs = arts.model.predict(X, verbose=0)
    import tensorflow as tf
    losses = tf.keras.losses.sparse_categorical_crossentropy(y, probs).numpy()
    rows = []; idx = 0
    for epc, g in df.sort_values(["epc_code", "event_time"]).groupby("epc_code"):
        m = max(0, len(g) - window_size)
        if m == 0: continue
        ep_losses = losses[idx:idx+m]; idx += m
        rows.append({
            "epc_code": epc,
            "samples": int(m),
            "score_mean": float(ep_losses.mean()),
            "score_max": float(ep_losses.max()),
            "score_p95": float(np.percentile(ep_losses, 95)),
        })
    threshold = float(np.percentile([row["score_mean"] for row in rows], 99)) if rows else 0.0
    # 이상치 판단 결과 per_row로 반환
    per_row_results = []
    idx = 0
    for epc, g in df.sort_values(["epc_code", "event_time"]).groupby("epc_code"):
        m = max(0, len(g) - window_size)
        if m == 0: continue
        ep_losses = losses[idx:idx+m]; idx += m
        for i, loss in enumerate(ep_losses):
            per_row_results.append({
                "epc_code": epc,
                "window_idx": i,
                "score": float(loss),
                "is_anomaly": bool(loss > threshold)
            })
    return per_row_results, threshold, rows
