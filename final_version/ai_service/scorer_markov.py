# ai_service/scorer_markov.py
# ---------------------------------------------------------
# Markov(+Δt 버킷) 기반 이상 점수 계산기
# - 입력: 원본 이벤트 DF + (vocab, markov_model)
# - 처리: 전처리 → ID 인코딩 → 전이별 −logP 계산 → EPC 집계
# - 출력: EPC별 score_mean/max/p95 및 임계치
# ---------------------------------------------------------
import numpy as np
import pandas as pd
from .features import add_time_features, encode_with_vocab

def score_markov_per_epc(df_raw: pd.DataFrame, arts) -> tuple[pd.DataFrame, float]:
    # 모델/어휘집 체크
    model = arts.markov_model
    if model is None or not (arts.vocab_event and arts.vocab_loc):
        raise RuntimeError("Markov 아티팩트 부족(model/vocab)")

    # 전이 확률표와 Δt 버킷 경계 로드
    probs = model["probs"]                            # {state: {"combos": {...}, "__UNK__": p}}
    edges = np.array(model["bin_edges"], np.float64)  # 분위수 기반 경계

    def digitize(logdt): 
        # log1p(Δt)를 버킷 색인(0..B-1)으로 변환
        b = np.digitize(logdt, edges, right=False)
        return np.clip(b,0,len(edges))

    # 전처리 + ID 인코딩
    df = add_time_features(df_raw)
    df["event_type_id"]  = encode_with_vocab(df["event_type"], arts.vocab_event)
    df["location_id_id"] = encode_with_vocab(df["location_id"].astype(str), arts.vocab_loc)

    rows=[]
    for epc, g in df.sort_values(["epc_code","event_time"]).groupby("epc_code", sort=False):
        g = g.reset_index(drop=True)
        if len(g)<2: 
            continue
        loc=g["location_id_id"].to_numpy(int)
        evt=g["event_type_id"].to_numpy(int)
        dt=g["delta_t_sec"].to_numpy(float)
        b = digitize(np.log1p(np.clip(dt,0,None)))

        # 전이별 −logP 누적
        losses=[]
        for i in range(len(g)-1):
            s  = f"{loc[i]}|{evt[i]}"        # 현재 상태
            sp = f"{loc[i+1]}|{evt[i+1]}"    # 다음 상태
            bk = int(b[i+1])                 # 다음으로 가는 데 걸린 Δt 버킷
            key= f"{sp}|{bk}"
            if s in probs:
                p = probs[s]["combos"].get(key, probs[s]["__UNK__"])
            else:
                p = 1e-8                     # 미관측 시작상태 백오프
            losses.append(-float(np.log(max(p,1e-12))))

        if losses:
            losses=np.array(losses,float)
            rows.append({
                "epc_code": epc,
                "samples": int(len(losses)),
                "score_mean": float(losses.mean()),
                "score_max":  float(losses.max()),
                "score_p95":  float(np.percentile(losses,95)),
            })

    per_epc = pd.DataFrame(rows)

    # 임계치: 파일값 우선, 없으면 score_max p99 사용(희귀 전이에 민감)
    thr_dict = arts.markov_thresholds or {}
    thr = thr_dict.get("epc_score_max_p99")
    if thr is None and not per_epc.empty:
        thr = float(np.percentile(per_epc["score_max"],99))
    return per_epc, float(thr or 0.0)
