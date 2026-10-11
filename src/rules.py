
final_rules = [
    # HIGH RISK RULES
    {
        "name": "H1",
        "conditions": [("heart_rate", "high"), ("age", "young")],
        "risk": "high"
    },
    {
        "name": "H2",
        "conditions": [("heart_rate", "high"), ("oldpeak", "moderate")],
        "risk": "high"
    },
    {
        "name": "H3",
        "conditions": [("heart_rate", "high"), ("exercise_angina", "no")],
        "risk": "high"
    },
    {
        "name": "H4",
        "conditions": [
            ("age", "young"),
            ("oldpeak", "low"),
            ("exercise_angina", "no")
        ],
        "risk": "high"
    },
    {
        "name": "H5",
        "conditions": [("cholesterol", "very_high"), ("heart_rate", "high")],
        "risk": "high"
    },

    # MODERATE RISK RULES
    {
        "name": "M1",
        "conditions": [
            ("age", "middle"),
            ("blood_pressure", "high"),
            ("oldpeak", "moderate")
        ],
        "risk": "moderate"
    },
    {
        "name": "M2",
        "conditions": [
            ("cholesterol", "high"),
            ("heart_rate", "normal"),
            ("oldpeak", "moderate")
        ],
        "risk": "moderate"
    },
    {
        "name": "M3",
        "conditions": [
            ("age", "middle"),
            ("oldpeak", "moderate"),
            ("heart_rate", "normal")
        ],
        "risk": "moderate"
    },

    # LOW RISK RULES
    {
        "name": "L1",
        "conditions": [("oldpeak", "high"), ("exercise_angina", "yes")],
        "risk": "low"
    },
    {
        "name": "L2",
        "conditions": [("heart_rate", "low"), ("oldpeak", "high")],
        "risk": "low"
    },
    {
        "name": "L3",
        "conditions": [("cholesterol", "very_high"), ("exercise_angina", "yes")],
        "risk": "low"
    },
    {
        "name": "L4",
        "conditions": [("age", "old"), ("exercise_angina", "yes")],
        "risk": "low"
    }
]