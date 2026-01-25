import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import duckdb
import sys

# Directory and file paths
FEATURES_DIR = Path("../data/features")
CLUSTERING_RESULTS_DIR = Path("../result/clustering_scores")
CLUSTERS_RESULTS_DIR = Path("../result/clusters")


def find_best_params_for_step(step_size):
    """Find best window size and dilation for a specific step size"""
    scores_file = CLUSTERING_RESULTS_DIR / f"silhouette_scores_step{step_size}.csv"
    if not scores_file.exists():
        print(f"Warning: {scores_file} not found!")
        return None, None, None

    df = pd.read_csv(scores_file, index_col=0)
    max_pos = df.stack().idxmax()
    max_score = df.stack().max()
    return int(max_pos[0]), int(max_pos[1]), max_score


def cluster_analysis(window_size, dilation, step_size):
    """Perform clustering analysis for k=2 to k=10"""

    # Load feature data
    feature_file = (
        FEATURES_DIR / f"emg_len{window_size}_dil{dilation}_step{step_size}.parquet"
    )
    if not feature_file.exists():
        print(f"Warning: {feature_file} not found!")
        return None

    with duckdb.connect() as conn:
        df = conn.execute(
            f"""
            SELECT * FROM '{feature_file.as_posix()}'
        """
        ).df()

    # Extract features and metadata
    rms_cols = [col for col in df.columns if col.endswith("_rms")]
    pca_cols = [col for col in df.columns if col.startswith("PC")]

    # Use PCA features if available, otherwise use RMS features
    if pca_cols:
        feature_cols = pca_cols
        print(f"    Using {len(pca_cols)} PCA features")
    elif rms_cols:
        feature_cols = rms_cols
        print(f"    Using {len(rms_cols)} RMS features")
    else:
        print(f"    Error: No valid feature columns found!")
        return None

    X = df[feature_cols].values

    results = []
    cluster_results = df[["session_id", "label", "t"]].copy()

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Add cluster labels to dataframe
        cluster_results[f"cluster_k{k}"] = labels
        sil_score = silhouette_score(X, labels)
        elbow_score = kmeans.inertia_

        results.append({"k": k, "silhouette": sil_score, "elbow": elbow_score})

    # Save cluster results to parquet
    CLUSTERS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cluster_output = (
        CLUSTERS_RESULTS_DIR
        / f"emg_len{window_size}_dil{dilation}_step{step_size}_clusters.parquet"
    )

    with duckdb.connect() as conn:
        conn.register("cluster_temp", cluster_results)

        # Build column list for proper typing
        select_cols = [
            "session_id::BIGINT as session_id",
            "label::INTEGER as label",
            "t::INTEGER as t",
        ]
        for k in range(2, 11):
            select_cols.append(f"cluster_k{k}::INTEGER as cluster_k{k}")

        conn.execute(
            f"""
            COPY (
                SELECT {', '.join(select_cols)}
                FROM cluster_temp
                ORDER BY session_id, t
            ) TO '{Path(cluster_output).as_posix()}' (FORMAT PARQUET)
        """
        )

    return pd.DataFrame(results)


def plot_results(results_df, window_size, dilation, step_size):
    """Create elbow and silhouette plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow plot
    ax1.plot(results_df["k"], results_df["elbow"], "bo-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.grid(True)

    # Silhouette plot
    ax2.plot(results_df["k"], results_df["silhouette"], "ro-")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.grid(True)

    plt.suptitle(
        f"Clustering Analysis: window_size={window_size}, dilation={dilation}, step_size={step_size}"
    )
    plt.tight_layout()

    CLUSTERS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        CLUSTERS_RESULTS_DIR
        / f"cluster_analysis_len{window_size}_dil{dilation}_step{step_size}.png"
    )
    plt.savefig(output_file)
    plt.close()  # Close instead of show to prevent blocking

    return output_file


def get_available_step_sizes():
    """Extract all unique step sizes from available silhouette score CSV files"""
    score_files = list(CLUSTERING_RESULTS_DIR.glob("silhouette_scores_step*.csv"))

    step_sizes = set()
    for file_path in score_files:
        filename = file_path.stem  # Remove .csv extension
        parts = filename.split("_")
        if len(parts) >= 3 and parts[2].startswith("step"):
            try:
                step_size = int(parts[2].replace("step", ""))
                step_sizes.add(step_size)
            except ValueError:
                continue

    return sorted(list(step_sizes))


if __name__ == "__main__":
    step_sizes = (
        get_available_step_sizes()
    )  # Dynamically get step sizes with available results

    if not step_sizes:
        print("No clustering results found! Run cluster_win_dil_step.py first.")
        exit(1)

    print("Multi-Step Detailed Clustering Analysis")
    print("=" * 60)
    print(f"Found step sizes with results: {step_sizes}")
    print(f"Total step sizes to analyze: {len(step_sizes)}")
    print("=" * 60)

    for step_size in step_sizes:
        print(f"\nProcessing step size: {step_size}")

        # Find best parameters for this step size
        window_size, dilation, max_score = find_best_params_for_step(step_size)

        if window_size is None:
            print(f"Skipping step size {step_size} - no silhouette scores found")
            continue

        print(
            f"Best parameters: window_size={window_size}, dilation={dilation}, score={max_score:.6f}"
        )

        # Run detailed clustering analysis
        results_df = cluster_analysis(window_size, dilation, step_size)

        if results_df is None:
            print(f"Skipping step size {step_size} - no feature file found")
            continue

        # Save CSV results
        CLUSTERS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_csv = (
            CLUSTERS_RESULTS_DIR
            / f"cluster_results_len{window_size}_dil{dilation}_step{step_size}.csv"
        )
        results_df.to_csv(output_csv, index=False)

        # Create plots
        plot_file = plot_results(results_df, window_size, dilation, step_size)

        print(f"Results saved:")
        print(f"  CSV: {output_csv}")
        print(f"  Plot: {plot_file}")
        print("-" * 60)
