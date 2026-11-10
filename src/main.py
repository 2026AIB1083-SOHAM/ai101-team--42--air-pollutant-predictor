import os
import sys

# Add the source directory to Python's module search path
# so the individual project components can be imported.
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

    # Step 1: Load the air-quality dataset, handle missing
    # observations and prepare the predictor and target variables.
    df, X, y, target_col = load_and_clean_data("data/air_quality.csv", "CO(GT)")

    # Step 2: Explore the cleaned dataset through IQR-based
    # outlier detection, correlation analysis and visualizations.
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
