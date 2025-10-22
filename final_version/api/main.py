# api/main.py
# ---------------------------------------------------------
# FastAPI 애플리케이션 엔트리포인트
# - 시작 시 아티팩트 1회 로드
# - /health, /reload, /predict 라우터 제공
# ---------------------------------------------------------
from fastapi import FastAPI
from ai_service.artifacts import ARTS
from ai_service.settings import settings
from .routers.anomaly import router as anomaly_router

def create_app() -> FastAPI:
    # 앱 생성(메타 정보)
    app = FastAPI(title="AI Anomaly Service", version=settings.version)

    @app.on_event("startup")
    def _startup():
        # 프로세스 시작 시 아티팩트 초기 로딩
        ARTS.load()

    @app.get("/health")
    def health():
        # 간단한 헬스체크 엔드포인트
        return {"status": "ok", "version": settings.version}

    @app.post("/reload")
    def reload_artifacts():
        # 수동 재로딩(모델 핫스왑/임계치 갱신 시 사용)
        ARTS.load()
        return {"status": "reloaded"}

    # 이상탐지 라우터 등록
    app.include_router(anomaly_router)
    return app

# uvicorn 실행 진입점
# - 개발: uvicorn api.main:app --reload --port 8000
app = create_app()


@app.get("/artifacts")
def artifacts_status():
    ve = (ARTS.vocab_event or {}).get("id2token", [])
    vl = (ARTS.vocab_loc or {}).get("id2token", [])
    return {
        "artifact_dir": str(ARTS and ARTS.__dict__.get("lstm_model") and
                            __import__("os").getenv("ARTIFACT_DIR", "preprocessed")),
        "lstm_model": ARTS.lstm_model is not None,
        "scaler": ARTS.scaler is not None,
        "vocab_event_size": len(ve),
        "vocab_loc_size": len(vl),
        "lstm_threshold": ARTS.lstm_threshold,
        "markov_loaded": ARTS.markov_model is not None,
    }