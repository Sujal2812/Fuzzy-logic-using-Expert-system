
from pathlib import Path

import pandas as pd

from src.expert_system import predict_patient_risk


# Load the dataset from the project folder
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_disease_data.csv"
df = pd.read_csv(DATA_PATH)

results = []

for _, row in df.iterrows():
    patient = {
        "age": row["age"],
        "trestbps": row["trestbps"],
        "chol": row["chol"],
        "thalach": row["thalach"],
        "oldpeak": row["oldpeak"],
        "exang": row["exang"],
    }

    prediction = predict_patient_risk(patient)

    results.append({
        "actual_target": row["target"],
        "risk_score": prediction["risk_score"],
        "risk_category": prediction["risk_category"],
    })

results_df = pd.DataFrame(results)

total = len(results_df)
covered = (
    results_df["risk_category"] != "insufficient_evidence"
).sum()

print("Total records:", total)
print("Covered records:", covered)
print("Uncovered records:", total - covered)
print(f"Coverage: {covered / total * 100:.2f}%")

print("\nRisk category distribution:")
print(results_df["risk_category"].value_counts())

# Save the evaluation predictions
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "evaluation_predictions.csv"
)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved predictions to:", OUTPUT_PATH)