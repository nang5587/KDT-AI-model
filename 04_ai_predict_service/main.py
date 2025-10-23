# 04_ai_predict_service/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .artifacts import ARTS
from .settings import settings
from .router_anomaly import router as anomaly_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    ARTS.load()
    yield

app = FastAPI(title="AI Anomaly Service", version=settings.version, lifespan=lifespan)
app.include_router(anomaly_router)
