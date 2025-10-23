# 04_ai_predict_service/features.py
import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    out = out.sort_values(["epc_code", "event_time"])
    out["delta_t_sec"] = out.groupby("epc_code")["event_time"].diff().dt.total_seconds().fillna(0.0)
    h = out["event_time"].dt.hour + out["event_time"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    return out

def encode_with_vocab(series: pd.Series, vocab: dict) -> pd.Series:
    tok2id = {tok: i for i, tok in enumerate(vocab["id2token"])}
    unk = vocab["reserved"].index("<UNK>") if "<UNK>" in vocab["reserved"] else 1
    return series.astype(str).map(lambda x: tok2id.get(str(x), unk)).astype("int32")
