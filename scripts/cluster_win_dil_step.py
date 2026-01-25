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
    
    print(f"Looking for files with step size {step_size}")
    print(f"Found {len(feature_files)} feature files")
    
    results = []
    
    for file_path in feature_files:
        # Extract parameters from filename
        filename = file_path.stem
        parts = filename.split('_')
        window_size = int(parts[1].replace('len', ''))
        dilation = int(parts[2].replace('dil', ''))
        
        print(f"  Processing: window={window_size}, dilation={dilation}, step={step_size}")
        
        # Load feature data
        conn = duckdb.connect()
        df = conn.execute(f"""
            SELECT * FROM '{file_path.as_posix()}'
        """).df()
        conn.close()
        
        # Extract feature columns (CH*_rms or PC*)
        feature_cols = [col for col in df.columns if col.endswith('_rms') or col.startswith('PC')]
        
        if not feature_cols:
            print(f"    Warning: No feature columns found in {filename}")
            print(f"    Available columns: {df.columns.tolist()}")
            continue
            
        X = df[feature_cols].values
        
        print(f"    Using {len(feature_cols)} features: {feature_cols[:3]}{'...' if len(feature_cols) > 3 else ''}")
        
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
            print(f"    Error computing silhouette score for {window_size}-{dilation}: {e}")
            score = 0.0  # Default score for failed calculations
        
        results.append({
            'window_size': window_size,
            'dilation': dilation,
            'silhouette_score': score
        })
    
    # Create pivot table
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print(f"No results for step size {step_size}")
        return None, None, None
        
    pivot_table = results_df.pivot(index='window_size', columns='dilation', values='silhouette_score')
    
    # Save to CSV
    CLUSTERING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CLUSTERING_RESULTS_DIR / f"silhouette_scores_step{step_size}.csv"
    pivot_table.to_csv(output_file)
    
    print(f"Results saved to: {output_file}")
    
    # Print maximum score
    max_val = pivot_table.max().max()
    max_pos = pivot_table.stack().idxmax()
    print(f"Maximum score: {max_val:.6f} at window_size={max_pos[0]}, dilation={max_pos[1]}")
    print("-" * 60)
    
    return pivot_table, max_val, max_pos

if __name__ == "__main__":
    step_sizes = [100, 200, 300, 400]  # Analyze all step sizes
    
    print("Multi-Step Clustering Analysis")
    print("=" * 60)
    
    best_results = []
    
    for step_size in step_sizes:
        print(f"\nProcessing step size: {step_size}")
        result = cluster_and_score(step_size)
        
        if result[0] is not None:  # Check if pivot_table is not None
            pivot_table, max_score, max_pos = result
            best_results.append({
                'step_size': step_size,
                'best_window': max_pos[0],
                'best_dilation': max_pos[1],
                'best_score': max_score
            })
        else:
            print(f"Skipping step size {step_size} - no valid results")
    
    # Summary of best results
    print("\n" + "=" * 60)
    print("SUMMARY OF BEST PARAMETERS FOR EACH STEP SIZE:")
    print("=" * 60)
    
    for result in best_results:
        print(f"Step {result['step_size']:3d}: window={result['best_window']:3d}, "
              f"dilation={result['best_dilation']:3d}, score={result['best_score']:.6f}")
    
    # Find overall best
    overall_best = max(best_results, key=lambda x: x['best_score'])
    print(f"\nOverall Best: Step {overall_best['step_size']}, "
          f"window={overall_best['best_window']}, dilation={overall_best['best_dilation']}, "
          f"score={overall_best['best_score']:.6f}")