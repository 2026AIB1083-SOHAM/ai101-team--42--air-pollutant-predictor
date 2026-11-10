import os
import numpy as np
import pandas as pd

def load_and_clean_data(filepath="data/air_quality.csv", target_col="CO(GT)"):
   # Resolve the dataset location so the file can be accessed
   # correctly from different working directories.
    if not os.path.exists(filepath):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath}. Place air_quality.csv inside data/")

    # Identify the CSV separator from the first line because
    # the original UCI dataset uses semicolons instead of commas.
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    decimal = "," if sep == ";" else "."

    df = pd.read_csv(filepath, sep=sep, decimal=decimal)

    # Remove empty unnamed columns and completely blank rows
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(how="all")
    print(f"[Cleaning] Raw dataset shape: {df.shape}")

    # Convert the dataset's -200 missing-value marker into NaN
    # so pandas can handle missing sensor observations consistently.
    df = df.replace(-200, np.nan).replace(-200.0, np.nan)

    # Keep numeric columns and drop rows with missing values
    num_df = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
    print(f"[Cleaning] Cleaned dataset shape: {num_df.shape}")

    if target_col not in num_df.columns:
        target_col = num_df.columns[0]

    # Separate input variables (X) and target output variable (y)
    X = num_df.drop(columns=[target_col])
    y = num_df[target_col]

    return num_df, X, y, target_col


if __name__ == "__main__":
    df, X, y, target_col = load_and_clean_data()
    print(f"Target Variable (y): {target_col}")
    print(f"Predictor Features (X): {list(X.columns)}")
