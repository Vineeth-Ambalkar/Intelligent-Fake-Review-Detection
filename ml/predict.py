"""
predict.py
-----------
Simulates the backend prediction service in the Workflow diagram:
"Extract features -> ML model predicts Genuine/Suspicious -> store &
display result". Run this after train_model.py to test the trained
model on new, hand-written examples.
"""

import sys, os
import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from feature_extraction import extract_features, FEATURE_ORDER

MODEL_PATH = "/home/claude/fake_review_detection/ml/model.joblib"


def predict(review):
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]
    feats = extract_features(review)
    X = pd.DataFrame([feats])[features]
    prob_suspicious = model.predict_proba(X)[0, 1]
    label = "suspicious" if prob_suspicious >= 0.5 else "genuine"
    return label, prob_suspicious, feats


if __name__ == "__main__":
    examples = [
        {
            "review_text": "Amazing product! Exceeded my expectations, works perfectly.",
            "rating": 5, "was_returned": 1, "return_reason": "defective",
            "days_to_return": 2, "review_burst": 0,
        },
        {
            "review_text": "Terrible product, stopped working within a week. Returning it.",
            "rating": 1, "was_returned": 1, "return_reason": "defective",
            "days_to_return": 5, "review_burst": 0,
        },
        {
            "review_text": "Great product, does exactly what I needed, no complaints.",
            "rating": 5, "was_returned": 0, "return_reason": "none",
            "days_to_return": 0, "review_burst": 0,
        },
        {
            "review_text": "Perfect! Best purchase ever, love it so much!!!",
            "rating": 5, "was_returned": 1, "return_reason": "not_as_described",
            "days_to_return": 1, "review_burst": 1,
        },
    ]
    for ex in examples:
        label, prob, feats = predict(ex)
        print("-" * 70)
        print("Review:", ex["review_text"])
        print(f"Rating: {ex['rating']}  Returned: {bool(ex['was_returned'])}  "
              f"Reason: {ex['return_reason']}")
        print(f"--> Prediction: {label.upper()}  (suspicious probability: {prob:.2f})")
