"""
train_model.py
----------------
Loads the dataset, extracts text + behavioural features, trains two
models (Logistic Regression baseline + Gradient Boosted Trees - the
scikit-learn equivalent used here since this offline environment can't
pip-install XGBoost, but it is a drop-in swap: replace
GradientBoostingClassifier with xgboost.XGBClassifier and the rest of
the pipeline is unchanged), evaluates them, and saves the best model +
a feature-importance report for the "Explainable AI dashboard" future
enhancement.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)

import sys, os
sys.path.append(os.path.dirname(__file__))
from feature_extraction import extract_features, FEATURE_ORDER

DATA_PATH = "/home/claude/fake_review_detection/data/reviews_dataset.csv"
MODEL_PATH = "/home/claude/fake_review_detection/ml/model.joblib"
REPORT_PATH = "/home/claude/fake_review_detection/ml/evaluation_report.json"

df = pd.read_csv(DATA_PATH)

feat_rows = [extract_features(r) for _, r in df.iterrows()]
X = pd.DataFrame(feat_rows)[FEATURE_ORDER]
y = (df["label"] == "suspicious").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---- Baseline: Logistic Regression ----
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train_s, y_train)
logreg_probs = logreg.predict_proba(X_test_s)[:, 1]
logreg_preds = logreg.predict(X_test_s)

# ---- Main model: Gradient Boosted Trees (XGBoost-equivalent) ----
gbc = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.08, max_depth=3, random_state=42
)
gbc.fit(X_train, y_train)  # tree models don't need scaling
gbc_probs = gbc.predict_proba(X_test)[:, 1]
gbc_preds = gbc.predict(X_test)

results = {}
for name, y_pred, y_prob in [
    ("logistic_regression", logreg_preds, logreg_probs),
    ("gradient_boosted_trees", gbc_preds, gbc_probs),
]:
    report = classification_report(y_test, y_pred, target_names=["genuine", "suspicious"], output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    results[name] = {"classification_report": report, "roc_auc": auc, "confusion_matrix": cm}
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["genuine", "suspicious"]))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion matrix [rows=true, cols=pred] (genuine, suspicious):")
    print(np.array(cm))

# feature importances from the tree model, for the "explainable AI" view
importances = dict(zip(FEATURE_ORDER, gbc.feature_importances_.tolist()))
importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
print("\nFeature importances (Gradient Boosted Trees):")
for k, v in importances.items():
    print(f"  {k:35s} {v:.4f}")

results["feature_importances"] = importances

with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)

joblib.dump({"model": gbc, "scaler": None, "features": FEATURE_ORDER}, MODEL_PATH)
print(f"\nSaved trained model to {MODEL_PATH}")
print(f"Saved evaluation report to {REPORT_PATH}")
