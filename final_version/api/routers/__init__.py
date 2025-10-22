# api 패키지 초기화
# - FastAPI 앱/라우터를 외부에서 쉽게 임포트할 수 있게 단축 경로 제공

from .main import app, create_app

__all__ = ["app", "create_app"]