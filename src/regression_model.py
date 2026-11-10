import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_regression_model(X, y, target_col):
    os.makedirs("outputs", exist_ok=True)

    # Divide the available observations into training and testing
    # subsets, keeping 20% aside for evaluating unseen data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the multiple linear regression model to learn the
    # intercept and coefficients associated with each input feature.
    model = LinearRegression().fit(X_train, y_train)

    b0 = model.intercept_
    b_coeffs = pd.Series(model.coef_, index=X.columns)

    print("\n--- [Model Training] Parametric Linear Regression Parameters ---")
    print(f"Intercept (b0): {b0:.4f}")
    for col, val in b_coeffs.items():
        print(f"Slope/Coefficient for {col:12s} : {val:+.4f}")

    # Graph 4: Best Fit Line & Vertical Residuals (e_i = y_i - y_hat_i)
    best_col = max(X.columns, key=lambda c: abs(np.corrcoef(X[c], y)[0, 1]))
    simple_model = LinearRegression().fit(X_train[[best_col]], y_train)

    sample_X = X_test[[best_col]].iloc[:25]
    sample_y = y_test.iloc[:25]
    sample_pred = simple_model.predict(sample_X)

    plt.figure(figsize=(7, 5))
    x_line = np.linspace(sample_X[best_col].min(), sample_X[best_col].max(), 100)
    y_line = simple_model.intercept_ + simple_model.coef_[0] * x_line
    plt.plot(x_line, y_line, color="green", linewidth=2, label="Best Fit Line")

    plt.scatter(sample_X[best_col], sample_y, color="royalblue", s=45, label="Actual Observed (y)")
    plt.scatter(sample_X[best_col], sample_pred, color="white", edgecolors="green", s=35, label="Predicted on Line (y_hat)")

    for x_val, actual_val, pred_val in zip(sample_X[best_col], sample_y, sample_pred):
        plt.plot([x_val, x_val], [actual_val, pred_val], color="red", linestyle="--", linewidth=1)

    plt.xlabel(best_col)
    plt.ylabel(target_col)
    plt.title(f"Best Fit Line & Vertical Residuals ({target_col} vs {best_col})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "4_best_fit_and_vertical_residuals.png"))
    plt.close()

    print("[Model Training] Saved Graph 4 to outputs/")
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from cleaning import load_and_clean_data
    _, X, y, target_col = load_and_clean_data()
    train_regression_model(X, y, target_col)
