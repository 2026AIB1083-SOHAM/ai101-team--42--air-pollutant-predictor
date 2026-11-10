import os
import sys

# Ensure Python locates all component modules inside src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cleaning import load_and_clean_data
from visualization import run_eda_and_visualize
from regression_model import train_regression_model
from model_evaluation import evaluate_regression_model

def main():
    os.makedirs("outputs", exist_ok=True)

    print("====================================================")
    print("   AIR POLLUTANT PREDICTION PIPELINE (AI101)        ")
    print("====================================================")

    # Step 1: Data Cleaning (pandas, numpy)
    df, X, y, target_col = load_and_clean_data("data/air_quality.csv", "CO(GT)")

    # Step 2: Outlier Detection, Correlation & Visualization (numpy, pandas, matplotlib)
    correlations = run_eda_and_visualize(df, X, y, target_col)

    # Step 3: Regression Model Training (scikit-learn, numpy, pandas, matplotlib)
    model, X_train, X_test, y_train, y_test = train_regression_model(X, y, target_col)

    # Step 4: Model Evaluation & Overfitting Check (scikit-learn, numpy, pandas, matplotlib)
    metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, target_col)

    print("\n====================================================")
    print("Pipeline Complete! All 7 Matplotlib graphs saved in outputs/")
    print("====================================================")

if __name__ == "__main__":
    main()