import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from src.expert_system import predict_patient_risk


# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Fuzzy Logic Expert System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]
EVALUATION_PATH = BASE_DIR / "outputs" / "evaluation_predictions.csv"
DATASET_PATH = BASE_DIR / "data" / "heart_disease_data.csv"

# ==================================================
# CUSTOM UI STYLING
# ==================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #164e63);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #dbeafe;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 0.8rem;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SHARED HEADER
# ==================================================
def show_header():
    st.markdown(
        """
        <div class="hero">
            <h1>Medical Diagnostic Expert System</h1>
            <p>
                Rule-Based Fuzzy Logic for Educational
                Heart-Disease Risk Analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "This academic prototype is not clinically validated. "
        "Its output must not be used to diagnose disease or "
        "make medical decisions."
    )


# ==================================================
# PAGE 1: RISK ANALYSIS
# ==================================================
def show_risk_analysis():
    st.subheader("Patient Feature Inputs")

    st.caption(
        "Use synthetic sample values for demonstration. "
        "These inputs are not suitable for real-world diagnosis."
    )

    selected_patient = st.session_state.get(
        "selected_dataset_patient", {}
    )

    with st.form("risk_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=20,
                max_value=80,
                value=int(selected_patient.get("age", 30)),
            )

            trestbps = st.number_input(
                "Resting blood pressure",
                min_value=80,
                max_value=200,
                value=int(selected_patient.get("trestbps", 125)),
            )

        with col2:
            chol = st.number_input(
                "Cholesterol",
                min_value=100,
                max_value=600,
                value=int(selected_patient.get("chol", 300)),
            )

            thalach = st.number_input(
                "Maximum heart rate",
                min_value=60,
                max_value=220,
                value=int(selected_patient.get("thalach", 190)),
            )

        with col3:
            oldpeak = st.number_input(
                "Oldpeak",
                min_value=0.0,
                max_value=6.2,
                value=float(selected_patient.get("oldpeak", 2.0)),
                step=0.1,
            )

            exang = st.selectbox(
                "Exercise-induced angina",
                options=[0, 1],
                index=int(selected_patient.get("exang", 0)),
                format_func=lambda x: (
                    "No (0)" if x == 0 else "Yes (1)"
                ),
            )

        submitted = st.form_submit_button(
            "Run Fuzzy Analysis",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        patient = {
            "age": age,
            "trestbps": trestbps,
            "chol": chol,
            "thalach": thalach,
            "oldpeak": oldpeak,
            "exang": exang,
        }

        result = predict_patient_risk(patient)

        st.session_state["last_result"] = result
        st.session_state["last_patient"] = patient

    if "last_result" not in st.session_state:
        st.info("Enter the sample values and click Run Fuzzy Analysis.")
        return

    result = st.session_state["last_result"]

    st.divider()
    st.subheader("Analysis Results")

    if result["risk_score"] is None:
        st.warning(
            "Insufficient evidence: none of the rules fired. "
            "The system cannot assign a risk category for these inputs."
        )
    else:
        score_col, category_col = st.columns(2)

        score_col.metric(
            "Fuzzy Risk Score",
            f"{result['risk_score']:.2f} / 100",
        )

        category_col.metric(
            "Rule-Based Category",
            result["risk_category"].title(),
        )

    st.markdown(
        '<div class="section-title">Aggregated Rule Strengths</div>',
        unsafe_allow_html=True,
    )

    aggregated = pd.DataFrame.from_dict(
        result["aggregated_risk"],
        orient="index",
        columns=["Strength"],
    )
    aggregated.index.name = "Risk Category"

    st.bar_chart(aggregated)
    st.dataframe(aggregated, use_container_width=True)

    st.markdown(
        '<div class="section-title">Fuzzy Membership Analysis</div>',
        unsafe_allow_html=True,
    )

    membership_rows = []

    for feature, categories in result["fuzzy_data"].items():
        for category_name, value in categories.items():
            membership_rows.append({
                "Feature": feature,
                "Membership": category_name,
                "Value": round(float(value), 4),
            })

    membership_df = pd.DataFrame(membership_rows)

    st.dataframe(
        membership_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Rule-by-Rule Explanation</div>',
        unsafe_allow_html=True,
    )

    rule_results = result["rule_results"].copy()
    rule_results["firing_strength"] = (
        rule_results["firing_strength"].round(4)
    )

    fired = rule_results[
        rule_results["firing_strength"] > 0
    ]

    rule_col1, rule_col2 = st.columns(2)
    rule_col1.metric("Total Rules", len(rule_results))
    rule_col2.metric("Rules Fired", len(fired))

    st.dataframe(
        rule_results,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View only the fired rules"):
        if fired.empty:
            st.info("No rules fired for these input values.")
        else:
            st.dataframe(
                fired,
                use_container_width=True,
                hide_index=True,
            )


# ==================================================
# PAGE 2: SYSTEM EVALUATION
# ==================================================
def show_system_evaluation():
    st.subheader("Fuzzy System Evaluation Dashboard")

    st.caption("Evaluation using saved dataset predictions.")

    if not EVALUATION_PATH.exists():
        st.error(
            "Evaluation predictions were not found.\n\n"
            "Run the following command from your project root:\n\n"
            "python -m src.evaluate_system"
        )
        return

    try:
        df = pd.read_csv(EVALUATION_PATH)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not read the evaluation CSV: {error}")
        return

    required_columns = {
        "actual_target",
        "risk_score",
        "risk_category",
    }

    if not required_columns.issubset(df.columns):
        st.error(
            "The evaluation CSV does not contain all required "
            "columns: actual_target, risk_score, risk_category."
        )
        return

    if df.empty:
        st.warning("The evaluation CSV contains no records.")
        return

    # Calculate rule coverage
    total_records = len(df)

    covered = df[
        df["risk_category"] != "insufficient_evidence"
    ].copy()

    uncovered_count = total_records - len(covered)

    coverage = (
        len(covered) / total_records * 100
        if total_records else 0
    )

    # Convert fuzzy categories into binary predictions
    covered["predicted_target"] = covered["risk_category"].map({
        "low": 0,
        "moderate": 1,
        "high": 1,
    })

    covered = covered.dropna(
        subset=["actual_target", "predicted_target"]
    )

    y_true = covered["actual_target"].astype(int)
    y_pred = covered["predicted_target"].astype(int)

    accuracy = (
        (y_true == y_pred).mean() * 100
        if len(covered) else 0
    )

    # Summary metrics
    st.subheader("Overall Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", total_records)
    c2.metric("Covered Records", len(covered))
    c3.metric("Rule Coverage", f"{coverage:.2f}%")
    c4.metric("Covered-Case Accuracy", f"{accuracy:.2f}%")

    st.progress(min(coverage / 100, 1.0))

    st.info(
        f"{uncovered_count} records have insufficient evidence "
        "because none of the fuzzy rules fired. Accuracy is "
        "calculated only on covered records."
    )

    # Risk category distribution
    st.divider()
    st.subheader("Risk Category Distribution")

    category_order = [
        "high",
        "moderate",
        "low",
        "insufficient_evidence",
    ]

    category_labels = [
        "High",
        "Moderate",
        "Low",
        "Insufficient Evidence",
    ]

    category_counts = (
        df["risk_category"]
        .value_counts()
        .reindex(category_order, fill_value=0)
    )

    distribution_df = pd.DataFrame({
        "Category": category_labels,
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

    if covered.empty:
        st.warning(
            "There are no covered records with valid targets "
            "to calculate binary performance metrics."
        )
        return

    # Confusion matrix
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
        columns=[
            "Predicted: No Disease",
            "Predicted: Disease",
        ],
    )

    st.dataframe(cm_df, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("True Negatives (TN)", int(tn))
    m2.metric("False Positives (FP)", int(fp))
    m3.metric("False Negatives (FN)", int(fn))
    m4.metric("True Positives (TP)", int(tp))

    # Classification report
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
        "Binary mapping: low → 0; moderate/high → 1. "
        "Insufficient-evidence records are excluded from binary "
        "performance metrics and reported separately."
    )

    st.warning(
        "These are exploratory results on the dataset used during "
        "rule development, not independent clinical validation."
    )


# ==================================================
# PAGE 3: ABOUT THE SYSTEM
# ==================================================
def show_about():
    st.subheader("Project Overview")

    st.write(
        "This project demonstrates how fuzzy membership functions, "
        "IF–THEN rules, rule aggregation, and defuzzification can "
        "be combined into an interpretable expert-system prototype."
    )

    st.subheader("System Pipeline")

    st.code(
        "Input Features\n"
        "      ↓\n"
        "Fuzzy Membership Functions\n"
        "      ↓\n"
        "12 IF–THEN Rules\n"
        "      ↓\n"
        "Rule Firing Strengths\n"
        "      ↓\n"
        "Aggregation\n"
        "      ↓\n"
        "Defuzzification\n"
        "      ↓\n"
        "Risk Category or Insufficient Evidence",
        language="text",
    )

    st.subheader("Input Features")

    st.markdown(
        """
        - Age
        - Resting blood pressure
        - Cholesterol
        - Maximum heart rate
        - Oldpeak
        - Exercise-induced angina
        """
    )

    st.subheader("Recorded Evaluation Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Records", "606")
    c2.metric("Rule Coverage", "73.60%")
    c3.metric("Covered-Case Accuracy", "73.99%")

    st.info(
        "These are the recorded evaluation results. Coverage is "
        "the percentage of records for which at least one rule "
        "fired. Accuracy is calculated only on covered records, "
        "not all 606 records."
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn for evaluation metrics
        - Streamlit for the interactive interface
        - Rule-based fuzzy logic for inference
        """
    )

    st.warning(
        "This is an academic educational prototype. It is not "
        "clinically validated and must not be used for diagnosis "
        "or treatment decisions."
    )


# ==================================================
# PAGE 4: DATASET EXPLORER
# ==================================================
def show_dataset_explorer():
    st.subheader("Dataset Explorer")
    st.caption(
        "Inspect individual dataset records and compare their "
        "actual targets with the fuzzy system's outputs."
    )

    if not DATASET_PATH.exists():
        st.error(f"Dataset file not found: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    st.write(f"**Total records:** {len(df)}")
    st.write(f"**Total columns:** {len(df.columns)}")

    # View the complete dataset
    with st.expander("View complete dataset"):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Inspect an Individual Record")

    row_number = st.number_input(
        "Enter dataset row number (starting from 0)",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    record = df.iloc[int(row_number)]

    st.markdown(f"### Dataset Record: {int(row_number)}")

    # Display the original values
    record_df = record.to_frame(name="Value")
    record_df.index.name = "Feature"

    st.dataframe(
        record_df,
        use_container_width=True,
    )

    # Callback function to perform navigation safely prior to rendering widgets
    def navigate_to_risk_analysis():
        st.session_state["selected_dataset_patient"] = {
            "age": record["age"],
            "trestbps": record["trestbps"],
            "chol": record["chol"],
            "thalach": record["thalach"],
            "oldpeak": record["oldpeak"],
            "exang": record["exang"],
        }
        st.session_state["selected_dataset_row"] = int(row_number)
        st.session_state["page_navigation"] = "Risk Analysis"

    # Trigger navigation using an on_click callback
    st.button(
        "Analyze this record",
        type="primary",
        on_click=navigate_to_risk_analysis,
    )

    st.divider()
    st.subheader("Actual Target vs Fuzzy Output")

    actual_target = int(record["target"])

    target_label = (
        "Disease (target = 1)"
        if actual_target == 1
        else "No Disease (target = 0)"
    )

    # Prepare the six features used by the fuzzy system
    patient = {
        "age": record["age"],
        "trestbps": record["trestbps"],
        "chol": record["chol"],
        "thalach": record["thalach"],
        "oldpeak": record["oldpeak"],
        "exang": record["exang"],
    }

    result = predict_patient_risk(patient)

    actual_col, score_col, category_col = st.columns(3)

    actual_col.metric("Actual Dataset Target", target_label)

    if result["risk_score"] is None:
        score_col.metric("Fuzzy Risk Score", "Unavailable")
        category_col.metric(
            "Fuzzy Category",
            "Insufficient Evidence",
        )
    else:
        score_col.metric(
            "Fuzzy Risk Score",
            f"{result['risk_score']:.2f} / 100",
        )
        category_col.metric(
            "Fuzzy Category",
            result["risk_category"].title(),
        )

    # Apply the same binary mapping used by the evaluation dashboard
    category = result["risk_category"]

    predicted_target = {
        "low": 0,
        "moderate": 1,
        "high": 1,
    }.get(category)

    if predicted_target is None:
        st.info(
            "No rule fired for this record. The system cannot "
            "make a binary comparison for this row."
        )
    else:
        st.write(
            f"**Binary prediction mapping:** "
            f"{predicted_target} "
            f"({'Disease' if predicted_target == 1 else 'No Disease'})"
        )

        if predicted_target == actual_target:
            st.success(
                "The binary prediction matches the dataset target."
            )
        else:
            st.warning(
                "The binary prediction differs from the dataset target."
            )

    # Display rule-by-rule explanation
    st.subheader("Rules Fired for This Record")

    rule_results = result["rule_results"].copy()

    rule_results["firing_strength"] = (
        rule_results["firing_strength"].round(4)
    )

    fired_rules = rule_results[
        rule_results["firing_strength"] > 0
    ]

    st.write(f"**Rules fired:** {len(fired_rules)} out of {len(rule_results)}")

    if fired_rules.empty:
        st.info("None of the fuzzy rules fired for this record.")
    else:
        st.dataframe(
            fired_rules,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Row numbers refer to zero-based positions in the CSV. "
        "This comparison is for academic analysis only and is "
        "not a clinical assessment."
    )


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
if "page_navigation" not in st.session_state:
    st.session_state["page_navigation"] = "Risk Analysis"

with st.sidebar:
    st.title("🫀 Fuzzy Expert System")
    st.caption("Academic AI / Fuzzy Logic OEP")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dataset Explorer",           
            "Risk Analysis",        
            "System Evaluation",
            "About the System",
        ],
        key="page_navigation",
    )

    st.divider()

    st.markdown("### System Information")
    st.write("**Method:** Rule-based fuzzy logic")
    st.write("**Knowledge base:** 12 rules")
    st.write("**Dataset:** 606 records")
    st.write("**Recorded rule coverage:** 73.60%")

    st.caption(
        "Educational prototype only. Not for medical diagnosis "
        "or treatment decisions."
    )


# ==================================================
# MAIN ROUTER
# ==================================================
show_header()

if page == "Risk Analysis":
    show_risk_analysis()
elif page == "Dataset Explorer":
    show_dataset_explorer()
elif page == "System Evaluation":
    show_system_evaluation()
elif page == "About the System":
    show_about()

st.divider()
st.caption(
    "Academic Open-Ended Problem | Rule-Based Fuzzy Logic | "
    "Educational use only"
)
