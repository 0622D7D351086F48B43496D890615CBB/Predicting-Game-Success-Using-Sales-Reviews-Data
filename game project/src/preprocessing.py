import pandas as pd

def clean_data(df):
    """Clean missing values and duplicates"""
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Fill numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill text columns
    text_cols = df.select_dtypes(include=["object"]).columns
    df[text_cols] = df[text_cols].fillna("Unknown")

    print("✅ Data cleaned successfully")
    return df
