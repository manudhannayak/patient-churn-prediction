"""
Exploratory data analysis for the patient churn dataset.
Generates summary statistics and key visualizations saved to reports/figures/.

Usage:
    python src/eda.py --data data/patients.csv
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/patients.csv")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    fig_dir = os.path.join(args.reports_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(args.data)

    print("=== Dataset shape ===")
    print(df.shape)
    print("\n=== Churn rate ===")
    print(df["churned"].value_counts(normalize=True).round(3))
    print("\n=== Summary statistics ===")
    print(df.describe().round(2))

    # Churn rate by insurance type
    plt.figure(figsize=(7, 4))
    churn_by_insurance = df.groupby("insurance_type")["churned"].mean().sort_values(ascending=False)
    sns.barplot(x=churn_by_insurance.index, y=churn_by_insurance.values, color="#3E5FD8")
    plt.ylabel("Churn rate")
    plt.title("Churn Rate by Insurance Type")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_by_insurance.png"), dpi=150)
    plt.close()

    # Days since last contact distribution by churn status
    plt.figure(figsize=(7, 4))
    sns.kdeplot(
        data=df, x="days_since_last_contact", hue="churned", fill=True, common_norm=False, alpha=0.4
    )
    plt.title("Days Since Last Contact by Churn Status")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "days_since_contact_dist.png"), dpi=150)
    plt.close()

    # Correlation heatmap of numeric features
    numeric_df = df.select_dtypes(include="number").drop(columns=["churned"], errors="ignore")
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()

    print(f"\nSaved 3 EDA figures to {fig_dir}/")


if __name__ == "__main__":
    main()
