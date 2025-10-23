# 04_ai_predict_service/main.py
from fastapi import FastAPI
from .artifacts import ARTS
from .settings import settings
from .router_anomaly import router as anomaly_router

app = FastAPI(title="AI Anomaly Service", version=settings.version)

@app.on_event("startup")
def startup():
    ARTS.load()

app.include_router(anomaly_router)
