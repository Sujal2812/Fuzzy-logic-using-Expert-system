
def aggregate_risk(rule_results):
    """
    Aggregate rule firing strengths by risk category.
    Uses the maximum firing strength within each category.
    """
    high_strength = rule_results[
        rule_results["risk"] == "high"
    ]["firing_strength"].max()

    moderate_strength = rule_results[
        rule_results["risk"] == "moderate"
    ]["firing_strength"].max()

    low_strength = rule_results[
        rule_results["risk"] == "low"
    ]["firing_strength"].max()

    high_strength = 0.0 if high_strength != high_strength else high_strength
    moderate_strength = 0.0 if moderate_strength != moderate_strength else moderate_strength
    low_strength = 0.0 if low_strength != low_strength else low_strength

    return {
        "high": float(high_strength),
        "moderate": float(moderate_strength),
        "low": float(low_strength)
    }


def defuzzify_risk(aggregated_risk):
    """
    Calculate a weighted risk score from aggregated strengths.
    """
    risk_centers = {
        "low": 20.0,
        "moderate": 50.0,
        "high": 80.0
    }

    numerator = sum(
        aggregated_risk[risk] * risk_centers[risk]
        for risk in aggregated_risk
    )

    denominator = sum(aggregated_risk.values())

    if denominator == 0:
        return None

    return numerator / denominator


def get_risk_category(score):
    """
    Convert the risk score to a category.
    A missing score means no rule supplied evidence.
    """
    if score is None:
        return "insufficient_evidence"
    elif score < 35:
        return "low"
    elif score < 65:
        return "moderate"
    else:
        return "high"