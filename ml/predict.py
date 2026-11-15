
from pathlib import Path

import joblib
import pandas as pd

from feature_extraction import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "model.joblib"


def predict(review):
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    features = bundle["features"]

    feats = extract_features(review)
    X = pd.DataFrame([feats])[features]

    prob_suspicious = model.predict_proba(X)[0, 1]
    label = "suspicious" if prob_suspicious >= 0.5 else "genuine"

    return label, prob_suspicious, feats
