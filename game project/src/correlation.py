import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_heatmap(df, save=False):
    """
    Generates a correlation heatmap for numeric features
    """

    print("📊 Generating Correlation Heatmap...")

    # Convert columns to numeric safely
    for col in ['meta_score', 'user_review']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.empty:
        print("❌ No numeric columns available for correlation")
        return

    # Compute correlation matrix
    corr = numeric_df.corr()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        linewidths=0.5,
        fmt=".2f"
    )
    plt.title("Correlation Heatmap of Game Dataset")
    plt.tight_layout()

    # Save if required
    if save:
        plt.savefig("outputs/correlation_heatmap.png")

    plt.show()
    print("✅ Correlation Heatmap displayed successfully")
