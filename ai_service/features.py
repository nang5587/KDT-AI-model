# ai_service/features.py
# ---------------------------------------------------------
# 전처리 공통 함수
# - 학습/서빙 간 전처리 불일치를 막기 위해 단일 모듈에서 관리
# - 시간 피처(Δt, hour_sin/cos), 어휘집 기반 인코딩 제공
# ---------------------------------------------------------
import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # event_time 파싱(실패 시 NaT → 예외)
    out = df.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    if out["event_time"].isna().any():
        raise ValueError("event_time 파싱 실패 행이 존재합니다.")

    # EPC/시간 정렬(윈도우링/Δt 일관성 보장)
    out.sort_values(["epc_code","event_time"], inplace=True, kind="stable")

    # Δt(초) 첫 이벤트는 0.0
    out["delta_t_sec"] = out.groupby("epc_code")["event_time"].diff().dt.total_seconds().fillna(0.0)

    # 시각의 주기성 피처(sin/cos). 분 단위까지 반영
    h = out["event_time"].dt.hour + out["event_time"].dt.minute/60.0
    out["hour_sin"] = np.sin(2*np.pi*h/24.0)
    out["hour_cos"] = np.cos(2*np.pi*h/24.0)
    return out

def encode_with_vocab(series: pd.Series, vocab: dict) -> pd.Series:
    # <PAD>=0, <UNK>=1 규칙을 가정
    tok2id = {tok:i for i, tok in enumerate(vocab["id2token"])}
    unk = vocab["reserved"].index("<UNK>") if "<UNK>" in vocab["reserved"] else 1
    # 미등록 토큰은 UNK로 안전하게 치환
    return series.astype(str).map(lambda x: tok2id.get(str(x), unk)).astype("int32")
