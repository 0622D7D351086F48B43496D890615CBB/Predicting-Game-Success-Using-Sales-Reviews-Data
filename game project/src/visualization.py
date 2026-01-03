import matplotlib.pyplot as plt
import seaborn as sns

def platform_score_plot(df):
    """Bar chart: Average Meta Score by Platform"""
    if 'platform' not in df.columns or 'meta_score' not in df.columns:
        print("❌ Required columns not found for visualization")
        return

    platform_score = df.groupby('platform')['meta_score'].mean().sort_values(ascending=False).reset_index()

    plt.figure(figsize=(10,5))
    sns.barplot(data=platform_score, x='platform', y='meta_score')
    plt.title("Average Meta Score by Platform")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
