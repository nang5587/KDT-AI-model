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

## 04_ai_predict_service/ 각 파일별 상세 역할

---

### 1. artifacts.py

- **주요 역할:**  
  모델, 스케일러, 이벤트/로케이션 vocab 등  
  AI 추론에 필요한 모든 외부 아티팩트(산출물) 파일을 메모리로 로딩/보관하는 객체(`ARTS`)를 제공.

- **상세 동작:**  
    - `epc_predictor_final_model.keras`: 학습된 LSTM 모델(keras) 파일을 load_model로 메모리 적재
    - `delta_t_scaler.joblib`: 이벤트 간 시간 간격(`delta_t_sec`) 정규화용 스케일러 파일을 joblib으로 로드
    - `event_type.vocab.json`, `location_id.vocab.json`: 이벤트타입, 위치코드 등 문자열을 모델 입력용 정수 인덱스로 변환하는 맵핑(vocab) json 파일을 읽어옴
    - **ARTS 객체:** FastAPI 전체에서 단일 인스턴스로 사용(모델/스케일러/어휘집 등 리소스를 매 API마다 새로 읽지 않고 효율적으로 관리)

---

### 2. features.py

- **주요 역할:**  
  입력 데이터(DataFrame)에 LSTM 입력에 필요한 전처리/파생특성/어휘 인코딩을 적용하는 함수 모음

- **상세 동작:**  
    - `add_time_features(df)`:  
        - event_time을 datetime으로 변환, epc_code별로 시간 순 정렬
        - 이벤트 간 시간 간격(`delta_t_sec`) 파생
        - 이벤트 발생 시간으로 hour_sin/hour_cos 파생 (주기성 캡처)
    - `encode_with_vocab(series, vocab)`:  
        - event_type, location_id 등의 문자열 컬럼을 vocab json을 참고해 정수 인덱스(토큰)로 변환 (모델 입력에 필수)
        - 미등록 값은 <UNK>로 매핑

- **의의:**  
    - 학습/서비스시 데이터 전처리 차이 없이 동일한 인코딩/정규화 로직 보장

---

### 3. scorer_lstm.py

- **주요 역할:**  
  입력 데이터(전처리/인코딩 완료)를 LSTM 모델로 예측, anomaly score와 임계치, 이상/정상 여부를 산출하는 핵심 점수 계산 파트

- **상세 동작:**  
    - 슬라이딩 윈도우 방식으로 epc_code별로 window_size 만큼 시계열 분할
    - 각 윈도우에 대해
        - LSTM 모델에 입력 → softmax 예측값 도출
        - 정답 event_type과의 cross-entropy(−logp) loss 계산(=anomaly score)
    - epc_code별로 score_mean, score_max, score_p95 등 통계값 산출
    - 전체 score_mean 분포에서 99% 위치(임계치, threshold) 산출
    - 각 EPC, 샘플별 score > threshold이면 이상(Anomaly), 아니면 정상(Normal)으로 판정
    - **결과:**  
        - 전체 row별(score, is_anomaly, 인덱스 등) 결과
        - epc별 통계 summary
        - 전체 임계치(threshold)

---

### 4. types.py

- **주요 역할:**  
  입력(JSON) 및 출력(예측결과) 데이터 구조를 Pydantic 모델로 정의.  
  FastAPI에서 요청/응답 데이터의 스키마 검증 및 문서화 자동화

- **상세 동작:**  
    - `EventRecord`: epc_code, event_time, event_type, location_id 등 실제 이벤트 1건의 구조 정의
    - `PredictRequest`: records(List[EventRecord])로 여러 이벤트를 한 번에 입력 받을 수 있게 설계
    - `AnomalyResult`: 결과 row별(epc_code, index, score, is_anomaly 등) 이상치 판정 및 메타 데이터 구조 정의
    - `EpcSummary`: epc_code별 전체 윈도우 결과 통계 (평균, 최대, 95%, 이상치 비율 등)
    - `PredictResponse`:
        - 결과 리스트(results: List[AnomalyResult])
        - 임계치(threshold)
        - epc별 summary(summary: List[EpcSummary])
        - 기타 info(추가 메타정보)

- **의의:**  
    - Spring(v2)와의 데이터 포맷 싱크/자동 검증 및 API 문서 자동화에 활용

---

### 5. settings.py

- **주요 역할:**  
  모델 추론, 점수 계산 등 전체 서비스에 필요한 공통 환경 설정 값 관리

- **상세 동작:**  
    - `window_size`: LSTM 슬라이딩 윈도우 크기 (ex: 3)
    - `version`: 서비스/모델 버전 명시
    - (환경변수/외부설정 확장 가능)

- **의의:**  
    - 코드 곳곳에 magic number 없이, 설정 변경을 한 곳에서 관리

---

### 6. main.py

- **주요 역할:**  
  FastAPI 전체 애플리케이션 진입점.  
  서버 실행시 모델 등 리소스 초기화, 라우터 등록

- **상세 동작:**  
    - lifespan 이벤트 핸들러(최신 FastAPI 패턴)로 서버 실행/재기동 시 ARTS.load() 호출(모델 등 메모리 적재)
    - router_anomaly.py의 router를 등록하여 `/predict/anomaly` 엔드포인트 활성화

- **의의:**  
    - 전체 서비스의 초기화, 라우팅, 확장 포인트 제공

---

### 7. router_anomaly.py

- **주요 역할:**  
  /predict/anomaly API 라우트 정의.  
  Spring 등에서 전달한 JSON(PredictRequest) 입력 받아,  
  전처리 → 모델 추론 → 이상치 판단 → JSON(PredictResponse) 결과 반환

- **상세 동작:**  
    - 요청(request: PredictRequest)에서 records 리스트 추출
    - features.py로 전처리/인코딩, scorer_lstm.py로 이상치 점수/임계치/summary 계산
    - per-row/epc별 결과를 types.py 스키마로 직렬화, JSON 반환
    - 입력 부족/포맷 이상 등 예외시 HTTPException(422) 등 명확한 에러 반환

- **의의:**  
    - 실제 v2 백엔드와 연동될 AI 추론(이상탐지) 메인 엔드포인트 구현부



