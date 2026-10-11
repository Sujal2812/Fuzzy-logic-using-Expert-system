
import numpy as np


# Dataset feature bounds used by the fuzzy system
age_min, age_max = 20, 80
bp_min, bp_max = 80, 200
chol_min, chol_max = 100, 600
hr_min, hr_max = 60, 220
oldpeak_min, oldpeak_max = 0.0, 6.2


def triangular_membership(x, a, b, c):
    x = np.asarray(x, dtype=float)
    membership = np.zeros_like(x)

    if b != a:
        mask = (x >= a) & (x <= b)
        membership[mask] = (x[mask] - a) / (b - a)

    if c != b:
        mask = (x >= b) & (x <= c)
        membership[mask] = (c - x[mask]) / (c - b)

    membership[x == b] = 1.0
    return membership


def trapezoidal_membership(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    membership = np.zeros_like(x)

    if b != a:
        mask = (x >= a) & (x <= b)
        membership[mask] = (x[mask] - a) / (b - a)

    mask = (x >= b) & (x <= c)
    membership[mask] = 1.0

    if d != c:
        mask = (x >= c) & (x <= d)
        membership[mask] = (d - x[mask]) / (d - c)

    return membership


def fuzzify_exang(exang):
    """
    Converts exercise-induced angina into fuzzy membership values.
    0 -> no: 1.0, yes: 0.0
    1 -> no: 0.0, yes: 1.0
    """
    if exang == 1:
        return {"no": 0.0, "yes": 1.0}
    else:
        return {"no": 1.0, "yes": 0.0}


def fuzzify_patient(age, trestbps, chol, thalach, oldpeak, exang):
    result = {
        "age": {
            "young": float(trapezoidal_membership(
                np.array([age]), age_min, age_min, 35, 45
            )[0]),
            "middle": float(triangular_membership(
                np.array([age]), 35, 50, 65
            )[0]),
            "old": float(trapezoidal_membership(
                np.array([age]), 55, 65, age_max, age_max
            )[0])
        },
        "blood_pressure": {
            "normal": float(trapezoidal_membership(
                np.array([trestbps]), bp_min, bp_min, 120, 135
            )[0]),
            "high": float(triangular_membership(
                np.array([trestbps]), 120, 140, 160
            )[0]),
            "very_high": float(trapezoidal_membership(
                np.array([trestbps]), 145, 160, bp_max, bp_max
            )[0])
        },
        "cholesterol": {
            "normal": float(trapezoidal_membership(
                np.array([chol]), chol_min, chol_min, 200, 230
            )[0]),
            "high": float(triangular_membership(
                np.array([chol]), 200, 240, 280
            )[0]),
            "very_high": float(trapezoidal_membership(
                np.array([chol]), 260, 280, chol_max, chol_max
            )[0])
        },
        "heart_rate": {
            "low": float(trapezoidal_membership(
                np.array([thalach]), hr_min, hr_min, 100, 130
            )[0]),
            "normal": float(triangular_membership(
                np.array([thalach]), 110, 150, 180
            )[0]),
            "high": float(trapezoidal_membership(
                np.array([thalach]), 160, 180, hr_max, hr_max
            )[0])
        },
        "oldpeak": {
            "low": float(trapezoidal_membership(
                np.array([oldpeak]), oldpeak_min, oldpeak_min, 1.0, 1.5
            )[0]),
            "moderate": float(triangular_membership(
                np.array([oldpeak]), 1.0, 2.0, 3.0
            )[0]),
            "high": float(trapezoidal_membership(
                np.array([oldpeak]), 2.5, 3.0, oldpeak_max, oldpeak_max
            )[0])
        },
        "exercise_angina": fuzzify_exang(exang)
    }

    return result