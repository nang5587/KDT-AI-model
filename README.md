your-project/
├─ ai_service/                 # ← 모델/전처리/스코어링 "순수 파이썬 모듈"
│  ├─ __init__.py
│  ├─ settings.py             # 경로/윈도우사이즈/모드 등 설정
│  ├─ artifacts.py            # 아티팩트 로더(모델/스케일러/어휘집/임계치)
│  ├─ features.py             # 전처리(시간 피처, vocab 인코딩)
│  ├─ scorer_lstm.py          # LSTM 기반 윈도우/점수 계산
│  ├─ scorer_markov.py        # Markov(+Δt) 기반 점수 계산
│  └─ types.py                # Pydantic 모델(입력/출력 스키마)
├─ api/
│  ├─ __init__.py
│  ├─ main.py                 # FastAPI 앱(라우터만) → ai_service 호출
│  └─ routers/
│     └─ anomaly.py           # /predict/anomaly 라우터
├─ preprocessed/              # 아티팩트(.keras/.joblib/.json)
├─ requirements.txt
└─ README.md