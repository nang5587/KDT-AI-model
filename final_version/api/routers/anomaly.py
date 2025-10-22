# api/routers/anomaly.py
# ---------------------------------------------------------
# FastAPI 라우터(이상탐지)
# - JSON 입력으로 LSTM/Markov 점수 계산 후 EPC별 판정 반환
# - API 계층은 얇게 유지하고 ai_service 모듈을 호출
# ---------------------------------------------------------
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from ai_service.types import PredictRequest, PredictResponse
from ai_service.settings import settings
from ai_service.artifacts import ARTS
from ai_service.scorer_lstm import score_lstm_per_epc
from ai_service.scorer_markov import score_markov_per_epc

router = APIRouter(prefix="/predict", tags=["predict"])

@router.post("/anomaly", response_model=PredictResponse)
def predict_anomaly(req: PredictRequest):
    # 입력 검증: records 필수
    if not req.records:
        raise HTTPException(status_code=400, detail="records가 비어있습니다.")

    # 리스트[dict] → DataFrame
    df = pd.DataFrame([r.dict() for r in req.records])

    # 필수 컬럼 점검
    need = {"epc_code","event_time","event_type","location_id"}
    missing = need - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"필수 컬럼 없음: {sorted(missing)}")

    # 모드 분기
    if req.mode == "lstm":
        per_epc, thr = score_lstm_per_epc(df, ARTS)
        if per_epc.empty:
            raise HTTPException(status_code=422, detail="점수 계산 불가(윈도우 부족)")
        per_epc["status"] = np.where(per_epc["score_mean"]>thr, "Anomaly", "Normal")
        return PredictResponse(
            mode="lstm",
            version=settings.version,
            threshold=float(thr),
            per_epc=per_epc.to_dict("records"),
            info={"window_size": settings.window_size}
        )

    elif req.mode == "markov":
        per_epc, thr = score_markov_per_epc(df, ARTS)
        if per_epc.empty:
            raise HTTPException(status_code=422, detail="점수 계산 불가")
        per_epc["status"] = np.where(per_epc["score_max"]>thr, "Anomaly", "Normal")
        return PredictResponse(
            mode="markov",
            version=settings.version,
            threshold=float(thr),
            per_epc=per_epc.to_dict("records"),
            info={}  # 필요 시 버킷 수 등 메타 추가
        )

    else:
        # 허용되지 않은 모드
        raise HTTPException(status_code=400, detail="mode는 'lstm' 또는 'markov' 여야 합니다.")
