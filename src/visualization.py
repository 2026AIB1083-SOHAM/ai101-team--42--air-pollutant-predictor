import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(_file_)))

def run_eda_and_visualize(df, X, y, target_col):
    os.makedirs("outputs", exist_ok=True)

    # 1. Lecture Note Implementation: Box Plot & 1.5 * IQR Outlier Rule
    s = df[target_col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (s < lower) | (s > upper)
    flagged = df.loc[mask]

    print("\n--- [EDA] Box Plot & 1.5 * IQR Outlier Detection ---")
    print(f"Q1 (25th percentile): {q1:.2f} | Median: {s.median():.2f} | Q3 (75th percentile): {q3:.2f}")
    print(f"IQR (Q3 - Q1): {iqr:.2f}")
    print(f"Outlier Limits: Lower = {lower:.2f}, Upper = {upper:.2f}")
    print(f"Flagged outlier records: {len(flagged)} out of {len(s)}")

    # Graph 1: Box Plot (Exact Lecture Note Syntax)
    plt.figure(figsize=(6, 7))
    plt.boxplot(s, whis=1.5, showfliers=True)
    plt.xticks([1], [target_col])
    plt.ylabel(f"{target_col} Concentration (dataset units)")
    plt.title(f"{target_col} Box Plot & Outlier Detection")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "1_boxplot_outliers.png"))
    plt.close()

    # 2. Lecture Note Implementation: 2x2 NumPy Correlation Matrix (Cell [0, 1])
    print("\n--- [EDA] 2x2 NumPy Correlation Matrix (np.corrcoef) ---")
    correlations = {}
    for col in X.columns:
        corr_matrix = np.corrcoef(X[col], y)
        correlations[col] = corr_matrix[0, 1]
        print(f"Correlation ({col:12s} vs {target_col}): {correlations[col]:+.4f}")

    # Graph 2: Target Concentration Distribution Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(y, bins=30, color="skyblue", edgecolor="black")
    plt.axvline(np.mean(y), color="red", linestyle="--", label=f"Mean ({np.mean(y):.2f})")
    plt.axvline(np.median(y), color="navy", linestyle="-", label=f"Median ({np.median(y):.2f})")
    plt.xlabel(f"{target_col} Concentration")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {target_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "2_target_histogram.png"))
    plt.close()

    # Graph 3: Feature Correlation Bar Chart
    corr_series = pd.Series(correlations).sort_values()
    plt.figure(figsize=(8, 5))
    plt.barh(corr_series.index, corr_series.values, color="teal", edgecolor="black")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Pearson Correlation Coefficient (r)")
    plt.title(f"Correlation of Predictors with {target_col}")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "3_correlation_bar_chart.png"))
    plt.close()

    print("[Visualization] Saved Graphs 1, 2, and 3 to outputs/")
    return correlations


if _name_ == "_main_":
    from cleaning import load_and_clean_data
    df, X, y, target_col = load_and_clean_data()
    run_eda_and_visualize(df, X, y, target_col)