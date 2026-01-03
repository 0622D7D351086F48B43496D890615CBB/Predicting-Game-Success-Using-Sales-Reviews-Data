import pandas as pd
import numpy as np


def descriptive_statistics(df):
    """
    Calculate Mean, Median, Mode, and Standard Deviation
    for numerical columns
    """

    numeric_cols = df.select_dtypes(include=["int64", "float64"])

    print("\n📊 Descriptive Statistics\n")

    for col in numeric_cols.columns:
        mean_val = np.mean(numeric_cols[col])
        median_val = np.median(numeric_cols[col])
        mode_val = numeric_cols[col].mode()[0]
        std_val = np.std(numeric_cols[col])

        print(f"🔹 Column: {col}")
        print(f"   Mean               : {mean_val:.2f}")
        print(f"   Median             : {median_val:.2f}")
        print(f"   Mode               : {mode_val:.2f}")
        print(f"   Standard Deviation : {std_val:.2f}")
        print("-" * 40)
