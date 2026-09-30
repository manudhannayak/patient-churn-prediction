"""
Synthetic healthcare patient-engagement dataset generator.

Simulates patient demographic, utilization, and engagement data with a
realistic churn signal (patients disengaging from care), suitable for
supervised classification. No real patient data is used or required --
this generator creates a reproducible, privacy-safe dataset that mimics
the structure of real healthcare utilization data (visit counts, claims,
appointment adherence, etc.).

Usage:
    python src/generate_data.py --n_patients 5000 --out data/patients.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate_patients(n_patients: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, size=n_patients)
    gender = rng.choice(["F", "M"], size=n_patients, p=[0.52, 0.48])
    region = rng.choice(["Northeast", "South", "Midwest", "West"], size=n_patients)
    insurance_type = rng.choice(
        ["Commercial", "Medicare", "Medicaid", "Uninsured"],
        size=n_patients,
        p=[0.45, 0.30, 0.20, 0.05],
    )

    chronic_conditions = rng.poisson(lam=1.2, size=n_patients).clip(0, 8)
    tenure_months = rng.integers(1, 96, size=n_patients)

    # Utilization signals
    visits_last_12mo = rng.poisson(lam=4, size=n_patients)
    missed_appointments_last_12mo = rng.poisson(lam=1.1, size=n_patients)
    avg_days_between_visits = rng.normal(60, 25, size=n_patients).clip(7, 365)
    portal_logins_last_90d = rng.poisson(lam=2.5, size=n_patients)
    claims_denied_last_12mo = rng.poisson(lam=0.4, size=n_patients)
    days_since_last_contact = rng.exponential(scale=45, size=n_patients).clip(0, 730)
    satisfaction_score = rng.normal(7.5, 1.8, size=n_patients).clip(1, 10)

    # --- Churn signal: a weighted logistic function of the above features ---
    # Higher days_since_last_contact, missed appts, denied claims, and low
    # satisfaction / engagement drive churn probability up.
    z = (
        -0.9
        + 0.014 * days_since_last_contact
        + 0.30 * missed_appointments_last_12mo
        + 0.45 * claims_denied_last_12mo
        - 0.16 * visits_last_12mo
        - 0.11 * portal_logins_last_90d
        - 0.17 * satisfaction_score
        + 0.006 * avg_days_between_visits
        - 0.006 * tenure_months
        + rng.normal(0, 0.75, size=n_patients)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "patient_id": [f"P{100000+i}" for i in range(n_patients)],
            "age": age,
            "gender": gender,
            "region": region,
            "insurance_type": insurance_type,
            "chronic_conditions": chronic_conditions,
            "tenure_months": tenure_months,
            "visits_last_12mo": visits_last_12mo,
            "missed_appointments_last_12mo": missed_appointments_last_12mo,
            "avg_days_between_visits": avg_days_between_visits.round(1),
            "portal_logins_last_90d": portal_logins_last_90d,
            "claims_denied_last_12mo": claims_denied_last_12mo,
            "days_since_last_contact": days_since_last_contact.round(1),
            "satisfaction_score": satisfaction_score.round(1),
            "churned": churned,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient churn dataset")
    parser.add_argument("--n_patients", type=int, default=5000)
    parser.add_argument("--out", type=str, default="data/patients.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_patients(args.n_patients, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"Churn rate: {df['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
