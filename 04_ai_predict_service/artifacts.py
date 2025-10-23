# 04_ai_predict_service/artifacts.py
from pathlib import Path
import json
import joblib
from tensorflow.keras.models import load_model

BASE = Path(__file__).parent.parent / "02_preprocessing_learning" / "preprocessed"

class Artifacts:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.vocab_event = None
        self.vocab_loc = None

    def load(self):
        self.model = load_model(BASE / "epc_predictor_final_model.keras")
        self.scaler = joblib.load(BASE / "delta_t_scaler.joblib")
        with open(BASE / "event_type.vocab.json", encoding="utf-8") as f:
            self.vocab_event = json.load(f)
        with open(BASE / "location_id.vocab.json", encoding="utf-8") as f:
            self.vocab_loc = json.load(f)

ARTS = Artifacts()
