# ai_service/types.py
# ---------------------------------------------------------
# Pydantic 데이터 모델 정의
# - FastAPI 요청/응답 스키마를 모듈로 분리해 재사용
# - 타입 검증/문서화 자동 지원
# ---------------------------------------------------------
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class EventRecord(BaseModel):
    # 단일 이벤트 입력 스키마
    epc_code: str
    event_time: str  # ISO8601 문자열(서버에서 파싱)
    event_type: str
    location_id: int

class PredictRequest(BaseModel):
    # 이상탐지 요청 모델
    mode: str = Field("lstm", description="'lstm' 또는 'markov'")
    records: List[EventRecord]

class PredictResponse(BaseModel):
    # 이상탐지 응답 모델
    mode: str
    version: str
    threshold: float
    per_epc: List[Dict[str, Any]]
    info: Dict[str, Any]
