from src.data_loader import load_data
from src.preprocessing import clean_data
from src.model import train_model
from src.visualization import platform_score_plot
from src.clustering import cluster_games
from src.correlation import correlation_heatmap
from src.histogram import plot_histogram
from src.kde_plot import plot_kde
from src.statistics import descriptive_statistics

DATA_PATH = "data/all_games.csv"

def main():
    df = load_data(DATA_PATH)
    if df is None:
        return

    df = clean_data(df)

    descriptive_statistics(df)
    correlation_heatmap(df)
    plot_histogram(df)
    plot_kde(df)
    # Hit/Flop classification
    if 'meta_score' in df.columns:
        df['hit_flop'] = df['meta_score'].apply(lambda x: 'Hit' if x >= 75 else 'Flop')

    # Train model
    model = train_model(df)

    # Visualizations
    platform_score_plot(df)

    # K-Means clustering
    df = cluster_games(df, n_clusters=3)

if __name__ == "__main__":
    main()
