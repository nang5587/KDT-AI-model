# 04_ai_predict_service/types.py
from pydantic import BaseModel
from typing import List, Dict, Any

class EventRecord(BaseModel):
    epc_code: str
    event_time: str
    event_type: str
    location_id: int

class PredictRequest(BaseModel):
    records: List[EventRecord]

class AnomalyResult(BaseModel):
    epc_code: str
    window_idx: int
    score: float
    is_anomaly: bool

class EpccSummary(BaseModel):
    epc_code: str
    samples: int
    score_mean: float
    score_max: float
    score_p95: float

class PredictResponse(BaseModel):
    results: List[AnomalyResult]
    threshold: float
    summary: List[EpccSummary]
