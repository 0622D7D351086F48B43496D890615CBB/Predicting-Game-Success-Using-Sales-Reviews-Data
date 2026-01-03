import pandas as pd
filepath = 'data/all_games.csv'
def load_data(filepath):
    """Load dataset"""
    try:
        df = pd.read_csv(filepath, encoding="latin1")
        print(df.head())
        print(df.info())
        print("✅ Dataset loaded successfully")
        return df
    except FileNotFoundError:
        print("❌ File not found:", filepath)
        return None
