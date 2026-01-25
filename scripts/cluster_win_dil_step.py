import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import duckdb

# Directory and file paths
FEATURES_DIR = Path("../data/features")
CLUSTERING_RESULTS_DIR = Path("../result/clustering_scores")


def cluster_and_score(step_size=100):
    """Cluster all feature parquet files with KMeans k=5 and compute silhouette scores"""

    feature_files = list(FEATURES_DIR.glob(f"emg_len*_dil*_step{step_size}.parquet"))

    results = []

    for file_path in feature_files:
        # Extract parameters from filename
        filename = file_path.stem
        parts = filename.split("_")
        window_size = int(parts[1].replace("len", ""))
        dilation = int(parts[2].replace("dil", ""))

        # Load feature data
        with duckdb.connect() as conn:
            df = conn.execute(
                f"""
                SELECT * FROM '{file_path.as_posix()}'
            """
            ).df()

        # Extract feature columns (CH*_rms or PC*)
        feature_column_names = [
            col for col in df.columns if col.endswith("_rms") or col.startswith("PC")
        ]

        if not feature_column_names:
            continue

        X = df[feature_column_names].values

        try:
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Calculate silhouette score with sample limit for memory efficiency
            if len(X) > 10000:
                # Sample for large datasets to avoid memory issues
                sample_indices = np.random.choice(len(X), size=10000, replace=False)
                X_sample = X[sample_indices]
                labels_sample = labels[sample_indices]
                score = silhouette_score(X_sample, labels_sample)
            else:
                score = silhouette_score(X, labels)

        except Exception as e:
            score = 0.0  # Default score for failed calculations

        results.append(
            {
                "window_size": window_size,
                "dilation": dilation,
                "silhouette_score": score,
            }
        )

    # Create pivot table
    results_df_per_step = pd.DataFrame(results)

    if results_df_per_step.empty:
        print(f"No results for step size {step_size}")
        return

    pivot_table = results_df_per_step.pivot(
        index="window_size", columns="dilation", values="silhouette_score"
    )

    # Save to CSV
    CLUSTERING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CLUSTERING_RESULTS_DIR / f"silhouette_scores_step{step_size}.csv"
    pivot_table.to_csv(output_file)


def get_available_step_sizes():
    """Extract all unique step sizes from available feature parquet files"""
    feature_files = list(FEATURES_DIR.glob("emg_len*_dil*_step*.parquet"))

    step_sizes = set()
    for file_path in feature_files:
        filename = file_path.stem  # Remove .parquet extension
        parts = filename.split("_")
        if len(parts) >= 4 and parts[3].startswith("step"):
            try:
                step_size = int(parts[3].replace("step", ""))
                step_sizes.add(step_size)
            except ValueError:
                continue

    return sorted(list(step_sizes))


if __name__ == "__main__":
    step_sizes = get_available_step_sizes()  # Dynamically get all available step sizes

    if not step_sizes:
        print("No feature files found!")
        exit(1)

    for step_size in step_sizes:
        print(f"Processing step size: {step_size}")
        cluster_and_score(step_size)
