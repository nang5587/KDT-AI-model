# 파일 트리
``` bash
01_DB/
    ├─ hws.csv
    ├─ icn.csv
    ├─ kum.csv
    └─ ygs.csv

02_preprocessing_learning/
    ├─ 01_Data_Preprocessing.ipynb
    ├─ 02_Model_Training_And_Evaluation.ipynb
    ├─ 03_DB_preprocessing_ensemble.ipynb
    └─ preprocessed/
          ├─ epc_predictor_final_model.keras
          ├─ delta_t_scaler.joblib
          ├─ event_type.vocab.json
          └─ location_id.vocab.json

03_experiments/
    └─ (실험용 ipynb)

4_ai_predict_service/
    ├─ artifacts.py        # 모델, 스케일러, vocab 파일(이벤트, 로케이션) 로딩 클래스
    ├─ features.py         # 입력 데이터 전처리/특성 생성 및 어휘 인코딩 함수
    ├─ scorer_lstm.py      # LSTM 기반 시계열 이상치 스코어 산출 함수 (윈도우별 anomaly 점수)
    ├─ types.py            # FastAPI용 Pydantic 입력/출력 스키마
    ├─ settings.py         # 서비스 공통 설정 (예: window_size)
    ├─ main.py             # FastAPI 앱 진입점. lifespan에서 모델 등 로딩, router 등록
    └─ router_anomaly.py   # "/predict/anomaly" 엔드포인트. 이상치 예측 및 결과 반환

05_ai_batch_predict/
    ├─ batch_predict.py
    └─ batch_utils.py

requirements.txt
README.md

```