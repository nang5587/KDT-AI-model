# 04_ai_predict_service/router_anomaly.py
from fastapi import APIRouter, HTTPException
from .types import PredictRequest, PredictResponse
from .scorer_lstm import score_lstm
from .artifacts import ARTS
from .settings import settings

router = APIRouter(prefix="api/manager/", tags=["predict"])

@router.post("/ai_module", response_model=PredictResponse)
def predict_anomaly(req: PredictRequest):
    import pandas as pd
    if not req.records:
        raise HTTPException(status_code=400, detail="records가 비어 있습니다.")
    df = pd.DataFrame([r.dict() for r in req.records])
    results, threshold, summary = score_lstm(df, ARTS, window_size=settings.window_size)
    if not results:
        raise HTTPException(status_code=422, detail="입력 데이터가 너무 적습니다(윈도우 부족).")
    return PredictResponse(
        results=results,
        threshold=threshold,
        summary=summary
    )
