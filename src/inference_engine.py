
import pandas as pd


def evaluate_rule(rule, fuzzy_data):
    """
    Calculate a rule's firing strength using the MIN operator.

    For an AND rule, the smallest membership value determines
    how strongly the rule fires.
    """
    memberships = []

    for feature, category in rule["conditions"]:
        membership_value = fuzzy_data[feature][category]
        memberships.append(membership_value)

    firing_strength = min(memberships)
    return firing_strength


def evaluate_all_rules(fuzzy_data, rules):
    """
    Evaluate all rules and return their firing strengths.
    """
    results = []

    for rule in rules:
        firing_strength = evaluate_rule(rule, fuzzy_data)

        results.append({
            "rule": rule["name"],
            "risk": rule["risk"],
            "firing_strength": firing_strength
        })

    return pd.DataFrame(results)