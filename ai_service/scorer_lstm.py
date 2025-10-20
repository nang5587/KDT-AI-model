# ai_service/scorer_lstm.py
# ---------------------------------------------------------
# LSTM 분류기 기반 이상 점수 계산기
# - 입력: 원본 이벤트 DF + 아티팩트(모델/스케일러/어휘집)
# - 처리: 전처리 → ID인코딩 → 윈도우 구성 → 예측손실(−log p) 산출
# - 출력: EPC별 score_mean/max/p95 및 임계치
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from .features import add_time_features, encode_with_vocab
from .settings import settings

def build_windows_for_lstm(df: pd.DataFrame, scaler) -> tuple[np.ndarray, np.ndarray]:
    # LSTM 입력 피처 순서(훈련 시와 동일해야 함)
    use = ["event_type_id","location_id_id","delta_t_sec","hour_sin","hour_cos"]

    # Δt 스케일링(훈련에서 fit된 스케일러로 transform만 실행)
    df = df.copy()
    df["delta_t_sec"] = scaler.transform(df[["delta_t_sec"]]).ravel()

    # EPC/시간 정렬 후 슬라이딩 윈도우 구성
    Xs, ys = [], []
    W = settings.window_size
    for _, g in df.sort_values(["epc_code","event_time"]).groupby("epc_code", sort=False):
        if len(g) <= W:
            continue
        feats = g[use].to_numpy(np.float32)       # (T, F)
        tgt = g["event_type_id"].to_numpy(np.int64)  # (T,)
        for i in range(len(g)-W):
            Xs.append(feats[i:i+W])               # (W, F)
            ys.append(int(tgt[i+W]))              # 다음 스텝 라벨(스칼라)

    if not Xs: 
        # 윈도우 생성 불가 시 빈 배열 반환
        return np.empty((0,W,5),np.float32), np.empty((0,),np.int64)

    return np.stack(Xs,0), np.array(ys,np.int64)

def score_lstm_per_epc(df_raw: pd.DataFrame, arts) -> tuple[pd.DataFrame, float]:
    # 필수 아티팩트 검증
    if not (arts.lstm_model and arts.scaler and arts.vocab_event and arts.vocab_loc):
        raise RuntimeError("LSTM 아티팩트 부족(model/scaler/vocab)")

    # 전처리 + ID 인코딩
    df = add_time_features(df_raw)
    df["event_type_id"]  = encode_with_vocab(df["event_type"], arts.vocab_event)
    df["location_id_id"] = encode_with_vocab(df["location_id"].astype(str), arts.vocab_loc)

    # 윈도우 구성
    X, y = build_windows_for_lstm(df, arts.scaler)
    if X.shape[0] == 0:
        return pd.DataFrame(), float("nan")

    # 예측확률 → 각 샘플 손실(−log p(y_true|X))
    probs = arts.lstm_model.predict(X, verbose=0)  # (N, C)
    losses = tf.keras.losses.sparse_categorical_crossentropy(y, probs).numpy()  # (N,)

    # 손실을 EPC 단위로 재집계(평균/최대/p95)
    rows = []; idx=0; W=settings.window_size
    for epc, g in df.sort_values(["epc_code","event_time"]).groupby("epc_code", sort=False):
        m = max(0, len(g)-W)           # 윈도우 개수
        if m==0: 
            continue
        ep = losses[idx:idx+m]; idx+=m
        rows.append({
            "epc_code": epc,
            "samples":  int(m),
            "score_mean": float(ep.mean()),
            "score_max":  float(ep.max()),
            "score_p95":  float(np.percentile(ep,95)),
        })
    per_epc = pd.DataFrame(rows)

    # 임계치: 파일이 있으면 사용, 없으면 score_mean p99를 임시 기준으로
    thr = arts.lstm_threshold if arts.lstm_threshold is not None else (
        float(np.percentile(per_epc["score_mean"],99)) if not per_epc.empty else 0.0)

    return per_epc, thr
