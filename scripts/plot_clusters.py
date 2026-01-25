import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Directory and file paths
CLUSTERING_RESULTS_DIR = Path("../result/clustering_scores")
CLUSTERS_RESULTS_DIR = Path("../result/clusters")
DICT_RESULTS_DIR = Path("../result/dict")
TEMPORAL_FIGURES_DIR = Path("../result/temporal_figures")

def find_best_params(step_size=200):
    """Find best window size and dilation from silhouette scores CSV"""
    scores_file = CLUSTERING_RESULTS_DIR / f"silhouette_scores_step{step_size}.csv"
    df = pd.read_csv(scores_file, index_col=0)
    max_pos = df.stack().idxmax()
    return int(max_pos[0]), int(max_pos[1]), step_size

def generate_dictionary_file(df, k, window_size, dilation, step_size):
    """Generate dictionary CSV file showing highest percentage cluster for each label at each time step"""
    
    # Get unique labels and time values
    labels = sorted(df['label'].unique())
    t_values = sorted(df['t'].unique())
    
    # Initialize dictionary data structure
    dictionary_data = {'label': labels}
    
    # First pass: calculate average peak percentage for each label across all time steps
    label_avg_peaks = {}
    for label in labels:
        peak_percentages = []
        for t in t_values:
            label_t_data = df[(df['label'] == label) & (df['t'] == t)]
            if len(label_t_data) == 0:
                continue
                
            cluster_counts = label_t_data[f'cluster_k{k}'].value_counts()
            total_count = len(label_t_data)
            
            # Find max percentage for this t
            max_percentage = 0
            for cluster in range(k):
                count = cluster_counts.get(cluster, 0)
                percentage = (count / total_count) * 100
                max_percentage = max(max_percentage, percentage)
            
            peak_percentages.append(max_percentage)
        
        # Calculate average peak percentage for this label
        label_avg_peaks[label] = np.mean(peak_percentages) if peak_percentages else 0
    
    # Second pass: For each time step, find the highest percentage cluster for each label
    for t in t_values:
        t_column = []
        
        for label in labels:
            # Filter data for this label and time step
            label_t_data = df[(df['label'] == label) & (df['t'] == t)]
            
            if len(label_t_data) == 0:
                t_column.append('')  # Empty if no data
                continue
            
            # Count clusters for this label at this time step
            cluster_counts = label_t_data[f'cluster_k{k}'].value_counts()
            total_count = len(label_t_data)
            
            # Find cluster with highest percentage
            max_cluster = -1
            max_percentage = 0
            
            for cluster in range(k):
                count = cluster_counts.get(cluster, 0)
                percentage = (count / total_count) * 100
                if percentage > max_percentage:
                    max_percentage = percentage
                    max_cluster = cluster
            
            # Use 0.8 times the average peak percentage for this label as threshold
            threshold = 0.8 * label_avg_peaks[label]
            
            # Only use cluster if peak percentage > 0.8 * average peak for this label
            if max_percentage > threshold:
                t_column.append(max_cluster)
            else:
                t_column.append('')  # Empty if peak is below 0.8 * average
        
        dictionary_data[str(t)] = t_column
    
    # Create DataFrame and save
    dict_df = pd.DataFrame(dictionary_data)
    
    # Generate filename following the same convention
    dict_filename = DICT_RESULTS_DIR / f'step{step_size}_win{window_size}_dil{dilation}_k{k}_dictionary.csv'
    
    # Ensure output directory exists
    dict_filename.parent.mkdir(parents=True, exist_ok=True)
    
    dict_df.to_csv(dict_filename, index=False)
    
    print(f"Dictionary saved to: {dict_filename}")
    return dict_filename

def create_cluster_percentage_plots():
    """Create plots showing cluster assignment percentages over time for each label for k=4,5,6,7"""
    
    # Process multiple step sizes
    step_sizes = [100, 200, 300, 400]
    
    # K values to plot
    k_values = [4, 5, 6, 7]
    
    for step_size in step_sizes:
        print(f"\n{'='*60}")
        print(f"Processing step size: {step_size}")
        print(f"{'='*60}")
        
        # Get best parameters for this step size
        window_size, dilation, step_size = find_best_params(step_size)
        print(f"Using optimal parameters: window_size={window_size}, dilation={dilation}, step_size={step_size}")
        
        for k in k_values:
            print(f"\nCreating plots for k={k}")
            
            # Load cluster data using best parameters
            cluster_file = CLUSTERS_RESULTS_DIR / f"emg_len{window_size}_dil{dilation}_step{step_size}_clusters.parquet"
            conn = duckdb.connect()
            df = conn.execute(f"SELECT session_id, label, t, cluster_k{k} FROM '{cluster_file.as_posix()}'").df()
            conn.close()
            
            print(f"Loaded {len(df)} data points for k={k} visualization")
            
            # Get unique labels
            labels = sorted(df['label'].unique())
            
            # Create subplots - 2 rows, 5 columns for 10 labels
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for i, label in enumerate(labels):
                ax = axes[i]
                
                # Filter data for this label
                label_data = df[df['label'] == label]
                
                # Group by t and calculate cluster percentages
                cluster_percentages = []
                t_values = sorted(label_data['t'].unique())
                
                for t in t_values:
                    t_data = label_data[label_data['t'] == t]
                    cluster_counts = t_data[f'cluster_k{k}'].value_counts()
                    total_count = len(t_data)
                    
                    # Calculate percentages for each cluster
                    percentages = {}
                    for cluster in range(k):  # k clusters means clusters 0 to k-1
                        count = cluster_counts.get(cluster, 0)
                        percentages[cluster] = (count / total_count) * 100
                    
                    cluster_percentages.append(percentages)
                
                # Convert to arrays for plotting and find peak percentages
                cluster_data = {cluster: [] for cluster in range(k)}
                peak_percentages = []  # Track maximum percentage at each time step
                peak_clusters = []     # Track which cluster has the peak at each time step
                
                for perc_dict in cluster_percentages:
                    # Find the cluster with maximum percentage at this time step
                    max_cluster = max(perc_dict, key=perc_dict.get)
                    max_percentage = perc_dict[max_cluster]
                    peak_percentages.append(max_percentage)
                    peak_clusters.append(max_cluster)
                    
                    for cluster in range(k):
                        cluster_data[cluster].append(perc_dict[cluster])
                
                # Calculate average peak percentage for this label
                avg_peak_percentage = np.mean(peak_percentages)
                
                # Create line plot for each cluster (without markers for non-peak)
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                
                for cluster in range(k):
                    color = colors[cluster % len(colors)]  # Cycle through colors if needed
                    ax.plot(t_values, cluster_data[cluster], 
                           color=color, label=f'Cluster {cluster}', 
                           linewidth=2, alpha=0.7)  # No markers on lines
                
                # Add dots only for peak percentages at each time step
                for j, (t_val, peak_cluster, peak_perc) in enumerate(zip(t_values, peak_clusters, peak_percentages)):
                    color = colors[peak_cluster % len(colors)]
                    ax.scatter(t_val, peak_perc, color=color, s=50, zorder=5, edgecolors='black', linewidth=1)
                
                ax.set_title(f'Label {label} (Avg Peak: {avg_peak_percentage:.1f}%)', fontsize=10)
                ax.set_xlabel('Time Index (t)')
                ax.set_ylabel('Percentage (%)')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                
                # Add legend only to first plot
                if i == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Calculate overall average peak percentage across all labels
            all_peak_percentages = []
            for label in labels:
                label_data = df[df['label'] == label]
                t_values_label = sorted(label_data['t'].unique())
                
                for t in t_values_label:
                    t_data = label_data[label_data['t'] == t]
                    cluster_counts = t_data[f'cluster_k{k}'].value_counts()
                    total_count = len(t_data)
                    
                    max_percentage = 0
                    for cluster in range(k):
                        count = cluster_counts.get(cluster, 0)
                        percentage = (count / total_count) * 100
                        max_percentage = max(max_percentage, percentage)
                    
                    all_peak_percentages.append(max_percentage)
            
            overall_avg_peak = np.mean(all_peak_percentages)
            
            # Create comprehensive title with filename and statistics
            filename = f'step{step_size}_win{window_size}_dil{dilation}_k{k}_cluster.png'
            plt.suptitle(f'{filename}\nOverall Avg Peak: {overall_avg_peak:.1f}% | Window={window_size}, Dilation={dilation}, Step={step_size}', 
                        fontsize=14, y=0.98)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Make room for the title
            
            # Save with correct path and parameters in filename
            output_file = TEMPORAL_FIGURES_DIR / f'step{step_size}_win{window_size}_dil{dilation}_k{k}_cluster.png'
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")
            plt.close()  # Close the figure to free memory
            
            # Generate dictionary file showing highest percentage cluster for each label at each time step
            generate_dictionary_file(df, k, window_size, dilation, step_size)

if __name__ == "__main__":
    create_cluster_percentage_plots()