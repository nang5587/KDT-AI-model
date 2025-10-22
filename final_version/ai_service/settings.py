# ai_service/settings.py
# ---------------------------------------------
# 공통 설정을 보관하는 모듈
# - 코드 곳곳에서 동일한 값을 참조하도록 중앙집중화
# - 환경변수로 쉽게 값 교체(운영/개발 전환)
# ---------------------------------------------
from pydantic import BaseModel
from pathlib import Path
import os

class Settings(BaseModel):
    # 서비스 버전(응답/로그에 표기)
    version: str = "v2.0.0"
    # 학습 산출물(모델/스케일러/어휘집) 폴더
    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "preprocessed"))
    # LSTM 윈도우 길이(학습과 동일해야 함)
    window_size: int = int(os.getenv("WINDOW_SIZE", "3"))

# 전역 설정 인스턴스(다른 모듈에서 import 해서 사용)
settings = Settings()