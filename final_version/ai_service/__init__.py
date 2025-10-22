# ai_service 패키지 초기화
# - 패키지 버전/메타데이터 노출
# - 자주 쓰는 심볼들을 얇게 재노출(re-export)
# - 무거운 로직(모델 로드 등)은 절대 금지

__version__ = "2.0.0"

# 외부에서 편하게 쓰라고 재노출
# from ai_service import features, scorer_lstm ...
from . import features as _features
from . import scorer_lstm as _scorer_lstm
from . import scorer_markov as _scorer_markov
from .settings import settings

# 사용자가 ai_service.features 처럼 접근 가능
features = _features
scorer_lstm = _scorer_lstm
scorer_markov = _scorer_markov

# 공개 API 한정 (IDE 자동완성/문서화에 도움)
__all__ = [
    "__version__",
    "features",
    "scorer_lstm",
    "scorer_markov",
    "settings",
]