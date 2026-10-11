
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "outputs" / "evaluation_predictions.csv"

df = pd.read_csv(DATA_PATH)

# Evaluate only records where at least one rule fired
covered = df[df["risk_category"] != "insufficient_evidence"].copy()

# Map fuzzy risk categories to binary predictions
covered["predicted_target"] = (
    covered["risk_category"].map({
        "low": 0,
        "moderate": 1,
        "high": 1,
    })
)

y_true = covered["actual_target"].astype(int)
y_pred = covered["predicted_target"].astype(int)

print("Covered records:", len(covered))
print("Excluded records:", len(df) - len(covered))
print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

print("\nConfusion Matrix:")
print("Rows = Actual [No Disease, Disease]")
print("Columns = Predicted [No Disease, Disease]")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["No Disease", "Disease"],
    zero_division=0,
))