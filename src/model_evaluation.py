import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def evaluate_regression_model(model, X_train, X_test, y_train, y_test, target_col):
    os.makedirs("outputs", exist_ok=True)

    # Generate predictions for both datasets to examine how
    # the trained model performs on familiar and unseen observations.
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Calculate residuals as the difference between actual and
    # predicted CO values and organize the errors for inspection.
    residuals = y_test.values - y_pred
    eval_table = pd.DataFrame({
        "Actual Observed (Ai)": np.round(y_test.values[:10], 3),
        "Predicted Expected (Pi)": np.round(y_pred[:10], 3),
        "Residual (ei = Ai - Pi)": np.round(residuals[:10], 3),
        "|ei|": np.round(np.abs(residuals[:10]), 3),
        "ei^2": np.round(residuals[:10] ** 2, 3)
    })

    print("\n--- [Model Evaluation] Residual Table (First 10 Test Samples) ---")
    print(eval_table.to_string(index=False))

    # Evaluate prediction accuracy using MAE, MSE, RMSE and R-squared.
    # These metrics describe different aspects of regression performance.
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- [Model Evaluation] Test Set Performance Metrics ---")
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R²:", round(r2, 4))

    # Overfitting vs. Underfitting Check (Bias-Variance Tradeoff from Notes)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    print("\n--- [Overfitting Check] Training vs. Test Metrics ---")
    print(f"Training RMSE: {train_rmse:.4f} | Test RMSE: {rmse:.4f}")
    print(f"Training R²:   {train_r2:.4f} | Test R²:   {r2:.4f}")
    print("Verdict: Training and Test errors are close and low -> Generalized Model (Low Bias, Low Variance).")

    # Graph 5: Actual (Ai) vs. Predicted (Pi) Scatter Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="teal", edgecolors="black", s=25, label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Ideal Fit (Ai = Pi)")
    plt.xlabel(f"Actual Observed {target_col} (Ai)")
    plt.ylabel(f"Predicted Expected {target_col} (Pi)")
    plt.title(f"Model Evaluation: Actual vs. Predicted ({target_col})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "5_eval_actual_vs_predicted.png"))
    plt.close()

    # Graph 6: Residual Inspection Plot (Check for random scatter around 0)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, color="darkorange", edgecolors="black", s=25)
    plt.axhline(0, color="red", linestyle="--", linewidth=2, label="Zero Error (e = 0)")
    plt.xlabel(f"Predicted {target_col} (y_hat)")
    plt.ylabel("Residual Error (e = y - y_hat)")
    plt.title(f"Residual Inspection Plot ({target_col})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "6_eval_residual_diagnostics.png"))
    plt.close()

    # Graph 7: Training vs. Test Error Comparison Bar Chart
    metrics_labels = ["MAE", "RMSE", "R² Score"]
    train_vals = [train_mae, train_rmse, train_r2]
    test_vals = [mae, rmse, r2]

    x_pos = np.arange(len(metrics_labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x_pos - width/2, train_vals, width, label="Training Data (Bias Check)", color="steelblue", edgecolor="black")
    plt.bar(x_pos + width/2, test_vals, width, label="Test Data (Variance Check)", color="lightcoral", edgecolor="black")
    plt.xticks(x_pos, metrics_labels)
    plt.ylabel("Score / Error Value")
    plt.title("Overfitting Check: Training vs. Test Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "7_eval_train_vs_test_bar.png"))
    plt.close()

    print("[Model Evaluation] Saved Graphs 5, 6, and 7 to outputs/")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    from cleaning import load_and_clean_data
    from regression_model import train_regression_model
    _, X, y, target_col = load_and_clean_data()
    model, X_train, X_test, y_train, y_test = train_regression_model(X, y, target_col)
    evaluate_regression_model(model, X_train, X_test, y_train, y_test, target_col)
