# ai_service/artifacts.py
# ---------------------------------------------------------
# 아티팩트 로더
# - 모델(.keras), 스케일러(.joblib), 어휘집/임계치(.json) 로딩
# - 일부 파일이 없어도 동작하도록 '선택적 로딩' 설계
# - FastAPI에서 /reload 로 재로딩 가능
# ---------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import json
import joblib
from tensorflow.keras.models import load_model
from .settings import settings

class Artifacts:
    def __init__(self):
        # LSTM용
        self.lstm_model = None     # keras model
        self.scaler = None         # StandardScaler
        self.vocab_event = None    # dict(json)
        self.vocab_loc = None      # dict(json)
        self.lstm_threshold = None # float
        # Markov용
        self.markov_model = None         # dict(json)
        self.markov_thresholds = None    # dict(json)

    def load(self):
        # 아티팩트 경로 루트
        d: Path = settings.artifact_dir

        # ----- LSTM 관련 파일 경로 -----
        mp = d / "epc_predictor_final_model.keras"  # 모델
        sp = d / "delta_t_scaler.joblib"            # 스케일러
        ve = d / "event_type.vocab.json"            # 이벤트 어휘집
        vl = d / "location_id.vocab.json"           # 로케이션 어휘집
        th = d / "lstm_threshold.json"               # LSTM 임계치(옵션)

        # 존재 시에만 개별 로딩
        if mp.exists(): self.lstm_model = load_model(mp)
        if sp.exists(): self.scaler = joblib.load(sp)
        if ve.exists(): self.vocab_event = json.loads(ve.read_text(encoding="utf-8"))
        if vl.exists(): self.vocab_loc = json.loads(vl.read_text(encoding="utf-8"))
        if th.exists():
            try:
                self.lstm_threshold = float(json.loads(th.read_text()).get("threshold"))
            except Exception:
                # 파일 형식 문제 등으로 파싱 실패해도 서비스 동작은 지속
                self.lstm_threshold = None

        # ----- Markov 관련 파일 경로 -----
        mm = d / "markov_dt_model.json"         # 학습된 전이확률/버킷 모델
        mt = d / "markov_dt_thresholds.json"    # 임계치(퍼센타일)

        if mm.exists(): self.markov_model = json.loads(mm.read_text(encoding="utf-8"))
        if mt.exists(): self.markov_thresholds = json.loads(mt.read_text(encoding="utf-8"))

# 전역 아티팩트 핸들
ARTS = Artifacts()
