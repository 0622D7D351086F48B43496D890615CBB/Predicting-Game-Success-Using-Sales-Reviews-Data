import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def cluster_games(df, n_clusters=3, plot=True):
    """
    K-Means clustering on meta_score vs user_review
    Handles non-numeric values like 'tbd'
    """

    # Check required columns
    if 'meta_score' not in df.columns or 'user_review' not in df.columns:
        print("❌ Required columns for clustering not found")
        return df

    # Convert to numeric, coerce errors to NaN
    df['meta_score'] = pd.to_numeric(df['meta_score'], errors='coerce')
    df['user_review'] = pd.to_numeric(df['user_review'], errors='coerce')

    # Drop rows with NaN in either column
    X = df[['meta_score', 'user_review']].dropna()
    if X.empty:
        print("❌ No valid numeric data for clustering")
        return df

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster labels
    df.loc[X.index, 'cluster'] = clusters

    print(f"✅ K-Means clustering completed with {n_clusters} clusters")

    # Plot
    if plot:
        plt.figure(figsize=(8,6))
        sns.scatterplot(
            x=X['meta_score'], y=X['user_review'],
            hue=clusters, palette='Set2', s=100
        )
        plt.title("K-Means Clustering of Games")
        plt.xlabel("Meta Score")
        plt.ylabel("User Review")
        plt.legend(title='Cluster')
        plt.show()

    return df
