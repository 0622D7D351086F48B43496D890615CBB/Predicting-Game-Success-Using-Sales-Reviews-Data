import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(df):
    """
    Plot histogram for Meta Score and User Review
    """

    print("📊 Generating Histogram...")

    # Convert columns to numeric safely
    df['meta_score'] = pd.to_numeric(df['meta_score'], errors='coerce')
    df['user_review'] = pd.to_numeric(df['user_review'], errors='coerce')

    # Drop missing values
    df = df.dropna(subset=['meta_score', 'user_review'])

    if df.empty:
        print("❌ No valid numeric data for histogram")
        return

    plt.figure(figsize=(10, 5))

    # Meta Score Histogram
    plt.hist(
        df['meta_score'],
        bins=10,
        alpha=0.7,
        label='Meta Score'
    )

    # User Review Histogram
    plt.hist(
        df['user_review'],
        bins=10,
        alpha=0.7,
        label='User Review'
    )

    plt.xlabel("Score Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Game Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    print("✅ Histogram displayed successfully")
