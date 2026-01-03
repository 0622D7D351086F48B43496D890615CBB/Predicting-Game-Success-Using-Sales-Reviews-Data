import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde(df):
    """
    KDE plot for Meta Score and User Review
    """

    print("📊 Generating KDE Plot...")

    # Convert to numeric safely (handles 'tbd')
    df['meta_score'] = pd.to_numeric(df['meta_score'], errors='coerce')
    df['user_review'] = pd.to_numeric(df['user_review'], errors='coerce')

    # Remove missing values
    df = df.dropna(subset=['meta_score', 'user_review'])

    if df.empty:
        print("❌ No valid numeric data for KDE plot")
        return

    plt.figure(figsize=(10, 5))

    # KDE for Meta Score
    sns.kdeplot(
        df['meta_score'],
        fill=True,
        label='Meta Score',
        linewidth=2
    )

    # KDE for User Review
    sns.kdeplot(
        df['user_review'],
        fill=True,
        label='User Review',
        linewidth=2
    )

    plt.xlabel("Score Value")
    plt.ylabel("Density")
    plt.title("KDE Plot of Game Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    print("✅ KDE Plot displayed successfully")
