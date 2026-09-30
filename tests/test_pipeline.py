"""Basic smoke tests for the data generation and feature pipeline."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import build_design_matrix
from generate_data import generate_patients


def test_generate_patients_shape():
    df = generate_patients(200, seed=1)
    assert len(df) == 200
    assert "churned" in df.columns
    assert set(df["churned"].unique()) <= {0, 1}


def test_churn_rate_reasonable():
    df = generate_patients(3000, seed=1)
    rate = df["churned"].mean()
    assert 0.05 < rate < 0.40, f"Churn rate {rate} out of expected range"


def test_build_design_matrix_no_nan():
    df = generate_patients(500, seed=2)
    X, y = build_design_matrix(df)
    assert X.isnull().sum().sum() == 0
    assert len(X) == len(y)


def test_build_design_matrix_categoricals_encoded():
    df = generate_patients(300, seed=3)
    X, _ = build_design_matrix(df)
    # gender/region/insurance_type should be one-hot encoded, not raw strings
    assert all(X.dtypes.apply(lambda d: d.kind in "biufc"))


if __name__ == "__main__":
    test_generate_patients_shape()
    test_churn_rate_reasonable()
    test_build_design_matrix_no_nan()
    test_build_design_matrix_categoricals_encoded()
    print("All tests passed.")
