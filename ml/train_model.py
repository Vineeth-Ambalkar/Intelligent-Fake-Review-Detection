
"""
train_model.py
---------------
Loads the review dataset, extracts text and behavioural features,
trains two machine learning models, evaluates them, and saves
the best model and feature importance report.
"""

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import sys

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "reviews_dataset.csv"
MODEL_PATH = BASE_DIR / "ml" / "model.joblib"
REPORT_PATH = BASE_DIR / "ml" / "evaluation_report.json"

# Ensure output directory exists
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. IMPORT FEATURE EXTRACTION
# --------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_extraction import extract_features, FEATURE_ORDER

# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

print("Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please run: python3 data/generate_dataset.py"
    )

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("The dataset is empty.")

print("Dataset loaded successfully!")
print("Total records:", len(df))

# --------------------------------------------------
# 4. FEATURE EXTRACTION
# --------------------------------------------------

print("\nExtracting features...")

feat_rows = [
    extract_features(row)
    for _, row in df.iterrows()
]

X = pd.DataFrame(feat_rows)[FEATURE_ORDER]

y = (df["label"] == "suspicious").astype(int)

print("Feature extraction completed!")
print("Number of features:", len(FEATURE_ORDER))

# --------------------------------------------------
# 5. SPLIT DATASET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 6. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --------------------------------------------------
# 7. BASELINE MODEL - LOGISTIC REGRESSION
# --------------------------------------------------

print("\nTraining Logistic Regression...")

logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logreg.fit(X_train_s, y_train)

logreg_probs = logreg.predict_proba(X_test_s)[:, 1]
logreg_preds = logreg.predict(X_test_s)

# --------------------------------------------------
# 8. MAIN MODEL - GRADIENT BOOSTING
# --------------------------------------------------

print("\nTraining Gradient Boosting Classifier...")

gbc = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=3,
    random_state=42
)

gbc.fit(X_train, y_train)

gbc_probs = gbc.predict_proba(X_test)[:, 1]
gbc_preds = gbc.predict(X_test)

# --------------------------------------------------
# 9. MODEL EVALUATION
# --------------------------------------------------

print("\n========== MODEL EVALUATION ==========")

results = {}

models = [
    ("logistic_regression", logreg_preds, logreg_probs),
    ("gradient_boosted_trees", gbc_preds, gbc_probs)
]

for name, y_pred, y_prob in models:

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["genuine", "suspicious"],
        output_dict=True,
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).tolist()

    results[name] = {
        "classification_report": report,
        "roc_auc": float(auc),
        "confusion_matrix": cm
    }

    print(f"\n=== {name} ===")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["genuine", "suspicious"],
            zero_division=0
        )
    )

    print("ROC-AUC:", round(auc, 4))

    print("Confusion Matrix:")
    print(np.array(cm))

# --------------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------------

print("\n========== FEATURE IMPORTANCE ==========")

importances = dict(
    zip(
        FEATURE_ORDER,
        gbc.feature_importances_.tolist()
    )
)

importances = dict(
    sorted(
        importances.items(),
        key=lambda item: -item[1]
    )
)

for feature, importance in importances.items():
    print(f"{feature:35s} {importance:.4f}")

results["feature_importances"] = importances

# --------------------------------------------------
# 11. SAVE EVALUATION REPORT
# --------------------------------------------------

with open(REPORT_PATH, "w") as file:
    json.dump(results, file, indent=2)

print("\nEvaluation report saved successfully!")
print("Location:", REPORT_PATH)

# --------------------------------------------------
# 12. SAVE TRAINED MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": gbc,
        "scaler": None,
        "features": FEATURE_ORDER
    },
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)

# --------------------------------------------------
# 13. FINAL STATUS
# --------------------------------------------------

print("\n======================================")
print("   FAKE REVIEW DETECTION TRAINING DONE")
print("======================================")
print("Dataset:", DATA_PATH)
print("Model:", MODEL_PATH)
print("Report:", REPORT_PATH)
print("\nAll operations completed successfully!")
