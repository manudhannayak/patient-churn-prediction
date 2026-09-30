"""
Train and evaluate Logistic Regression, Random Forest, and XGBoost models
on the patient churn dataset. Reports ROC-AUC, F1, and precision/recall,
and saves the best model + a SHAP summary plot.

Usage:
    python src/train.py --data data/patients.csv
"""
import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import build_design_matrix


def evaluate(name, model, X_test, y_test, results):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.4f}   F1: {f1:.4f}")
    print(classification_report(y_test, preds, digits=3))
    results[name] = {"roc_auc": round(auc, 4), "f1": round(f1, 4)}
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/patients.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.join(args.reports_dir, "figures"), exist_ok=True)

    df = pd.read_csv(args.data)
    X, y = build_design_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Logistic Regression (baseline) ---
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    evaluate("Logistic Regression", logreg, X_test_scaled, y_test, results)

    # --- Random Forest (grid search) ---
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [1, 3],
    }
    rf = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        rf_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"\nBest RF params: {rf.best_params_}")
    evaluate("Random Forest", rf.best_estimator_, X_test, y_test, results)

    # --- XGBoost (grid search) ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    xgb = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        ),
        xgb_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    print(f"\nBest XGB params: {xgb.best_params_}")
    best_xgb = xgb.best_estimator_
    xgb_auc = evaluate("XGBoost", best_xgb, X_test, y_test, results)

    # --- Pick best model by ROC-AUC ---
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model: {best_name} (ROC-AUC {results[best_name]['roc_auc']})")

    model_map = {"Logistic Regression": logreg, "Random Forest": rf.best_estimator_, "XGBoost": best_xgb}
    best_model = model_map[best_name]

    joblib.dump(best_model, os.path.join(args.models_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(args.models_dir, "scaler.joblib"))
    joblib.dump(list(X.columns), os.path.join(args.models_dir, "feature_names.joblib"))

    with open(os.path.join(args.reports_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- SHAP explainability on the best tree-based model (or XGB as fallback) ---
    shap_model = best_xgb if best_name != "Logistic Regression" else best_xgb
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.reports_dir, "figures", "shap_summary.png"), dpi=150)
    plt.close()

    # Top feature importance table
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(os.path.join(args.reports_dir, "feature_importance.csv"), index=False)

    print("\nTop 10 predictive drivers (SHAP):")
    print(importance.head(10).to_string(index=False))

    print(f"\nSaved best model ({best_name}) to {args.models_dir}/best_model.joblib")
    print(f"Saved SHAP summary plot to {args.reports_dir}/figures/shap_summary.png")


if __name__ == "__main__":
    main()
