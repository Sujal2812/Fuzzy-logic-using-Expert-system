
from src.membership_functions import fuzzify_patient
from src.rules import final_rules
from src.inference_engine import evaluate_all_rules
from src.defuzzification import aggregate_risk, defuzzify_risk, get_risk_category


def predict_patient_risk(patient):
    """
    Run the complete fuzzy inference pipeline for one patient.

    Required patient keys:
    age, trestbps, chol, thalach, oldpeak, exang
    """

    # Step 1: Convert patient inputs into fuzzy membership values
    fuzzy_data = fuzzify_patient(
        age=patient["age"],
        trestbps=patient["trestbps"],
        chol=patient["chol"],
        thalach=patient["thalach"],
        oldpeak=patient["oldpeak"],
        exang=patient["exang"]
    )

    # Step 2: Evaluate the 12 fuzzy rules
    rule_results = evaluate_all_rules(fuzzy_data, final_rules)

    # Step 3: Aggregate firing strengths by risk category
    aggregated_risk = aggregate_risk(rule_results)

    # Step 4: Calculate the risk score
    risk_score = defuzzify_risk(aggregated_risk)

    # Step 5: Determine the category
    risk_category = get_risk_category(risk_score)

    return {
        "risk_score": risk_score,
        "risk_category": risk_category,
        "aggregated_risk": aggregated_risk,
        "rule_results": rule_results,
        "fuzzy_data": fuzzy_data
    }