"""
Feature engineering for the patient churn dataset.

Handles categorical encoding, missing-value checks, and derived features
that reflect common healthcare-analytics engineering practice: recency,
utilization ratios, and risk flags.
"""
import pandas as pd


CATEGORICAL_COLS = ["gender", "region", "insurance_type"]
NUMERIC_COLS = [
    "age",
    "chronic_conditions",
    "tenure_months",
    "visits_last_12mo",
    "missed_appointments_last_12mo",
    "avg_days_between_visits",
    "portal_logins_last_90d",
    "claims_denied_last_12mo",
    "days_since_last_contact",
    "satisfaction_score",
]
TARGET = "churned"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived features
    df["missed_appt_rate"] = df["missed_appointments_last_12mo"] / (
        df["visits_last_12mo"] + df["missed_appointments_last_12mo"] + 1e-6
    )
    df["low_engagement_flag"] = (
        (df["portal_logins_last_90d"] == 0) & (df["visits_last_12mo"] <= 1)
    ).astype(int)
    df["high_risk_contact_gap"] = (df["days_since_last_contact"] > 90).astype(int)
    df["claims_friction"] = df["claims_denied_last_12mo"] > 0
    df["claims_friction"] = df["claims_friction"].astype(int)

    return df


def build_design_matrix(df: pd.DataFrame, fit_encoder=None):
    """One-hot encodes categoricals and returns (X, y, encoder)."""
    df = engineer_features(df)

    engineered_numeric = NUMERIC_COLS + [
        "missed_appt_rate",
        "low_engagement_flag",
        "high_risk_contact_gap",
        "claims_friction",
    ]

    X_numeric = df[engineered_numeric]
    X_categorical = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)

    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df[TARGET] if TARGET in df.columns else None

    return X, y
