"""
Run inference with the trained churn model on new patient records.

Usage:
    python src/predict.py --input data/new_patients.csv --output reports/predictions.csv
"""
import argparse

import joblib
import pandas as pd

from features import build_design_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="reports/predictions.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    model = joblib.load(f"{args.models_dir}/best_model.joblib")
    scaler = joblib.load(f"{args.models_dir}/scaler.joblib")
    feature_names = joblib.load(f"{args.models_dir}/feature_names.joblib")

    df = pd.read_csv(args.input)
    X, _ = build_design_matrix(df)

    # Align columns with training-time feature set
    X = X.reindex(columns=feature_names, fill_value=0)

    is_linear = hasattr(model, "coef_")
    X_input = pd.DataFrame(scaler.transform(X), columns=X.columns) if is_linear else X

    proba = model.predict_proba(X_input)[:, 1]
    preds = model.predict(X_input)

    out = df[["patient_id"]].copy() if "patient_id" in df.columns else pd.DataFrame(index=df.index)
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = preds
    out = out.sort_values("churn_probability", ascending=False)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
