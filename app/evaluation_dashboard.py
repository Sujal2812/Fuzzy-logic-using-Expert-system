
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from pathlib import Path


# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(
    page_title="System Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Fuzzy System Evaluation Dashboard")
st.caption(
    "Evaluation of the rule-based fuzzy logic heart-disease "
    "expert system using the saved dataset predictions."
)

st.warning(
    "Academic evaluation only. These results are not clinical "
    "validation and must not guide medical decisions."
)


# ------------------------------------------
# LOAD SAVED EVALUATION RESULTS
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "outputs" / "evaluation_predictions.csv"

if not DATA_PATH.exists():
    st.error(
        "Evaluation predictions were not found. "
        "Run: python -m src.evaluate_system"
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = {
    "actual_target",
    "risk_score",
    "risk_category",
}

if not required_columns.issubset(df.columns):
    st.error(
        "The evaluation CSV does not contain all required columns."
    )
    st.stop()


# ------------------------------------------
# CALCULATE COVERAGE
# ------------------------------------------
total_records = len(df)

covered = df[
    df["risk_category"] != "insufficient_evidence"
].copy()

uncovered_count = total_records - len(covered)

coverage = (
    len(covered) / total_records * 100
    if total_records else 0
)


# ------------------------------------------
# MAP RISK CATEGORIES TO BINARY CLASSES
# ------------------------------------------
covered["predicted_target"] = covered["risk_category"].map({
    "low": 0,
    "moderate": 1,
    "high": 1,
})

covered = covered.dropna(subset=["predicted_target"])

y_true = covered["actual_target"].astype(int)
y_pred = covered["predicted_target"].astype(int)

accuracy = (
    (y_true == y_pred).mean() * 100
    if len(covered) else 0
)


# ------------------------------------------
# SUMMARY METRIC CARDS
# ------------------------------------------
st.subheader("Overall Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{total_records}")
c2.metric("Covered Records", f"{len(covered)}")
c3.metric("Rule Coverage", f"{coverage:.2f}%")
c4.metric("Covered-Case Accuracy", f"{accuracy:.2f}%")

st.progress(min(coverage / 100, 1.0))

st.info(
    f"{uncovered_count} records have insufficient evidence "
    "because none of the fuzzy rules fired. Accuracy is calculated "
    "only on covered records."
)


# ------------------------------------------
# RISK CATEGORY DISTRIBUTION
# ------------------------------------------
st.divider()
st.subheader("Risk Category Distribution")

category_order = [
    "high",
    "moderate",
    "low",
    "insufficient_evidence",
]

category_counts = (
    df["risk_category"]
    .value_counts()
    .reindex(category_order, fill_value=0)
)

distribution_df = pd.DataFrame({
    "Category": [
        "High",
        "Moderate",
        "Low",
        "Insufficient Evidence",
    ],
    "Records": category_counts.values,
})

left, right = st.columns(2)

with left:
    st.bar_chart(
        distribution_df.set_index("Category")
    )

with right:
    st.dataframe(
        distribution_df,
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------
st.divider()
st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Disease", "Actual: Disease"],
    columns=["Predicted: No Disease", "Predicted: Disease"],
)

st.dataframe(cm_df, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)

m1.metric("True Negatives (TN)", tn)
m2.metric("False Positives (FP)", fp)
m3.metric("False Negatives (FN)", fn)
m4.metric("True Positives (TP)", tp)


# ------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------
st.divider()
st.subheader("Classification Metrics")

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["No Disease", "Disease"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()
st.dataframe(
    report_df.round(3),
    use_container_width=True,
)

st.caption(
    "Binary mapping used: low → 0; moderate/high → 1. "
    "Insufficient-evidence records are excluded from binary "
    "performance metrics and reported separately."
)