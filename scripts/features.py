"""
EMG Feature Extraction from Processed Parquet Database
Calculates dilated RMS features with configurable window sizes and dilation parameters.
"""

import os
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Directory and file paths
DB_DIR = Path("../data/db")
PROCESSED_DB_DIR = Path("../data/processed")
FEATURES_DIR = Path("../data/features")
INPUT_PARQUET = PROCESSED_DB_DIR / "emg_processed.parquet"


def process_session_data(session_data, window_size=400, step_size=100, dilation=100):
    """Process a single session and calculate dilated RMS for all channels
    
    Args:
        session_data: DataFrame containing one session's data
        window_size: Size of center window in samples
        step_size: Step between windows in samples
        dilation: Dilation gap in samples
    
    Returns:
        DataFrame with format: session_id, label, t, CH1_rms, CH2_rms, ..., CH8_rms (z-score normalized)
    """
    session_id = session_data['id'].iloc[0]
    label = session_data['label'].iloc[0]
    
    # Extract channel columns (these should be CH1_voltage, CH2_voltage, etc. from processed data)
    channel_cols = [f'CH{i}_voltage' for i in range(1, 9)]
    
    # Check which channels are available
    available_channels = [col for col in channel_cols if col in session_data.columns]
    
    if not available_channels:
        print(f"    Warning: No voltage channels found for session {session_id}")
        return None
    
    # Sort by time_index to ensure proper order
    session_data = session_data.sort_values('time_index').reset_index(drop=True)
    
    # Calculate required buffer for dilated windows
    half_window = window_size // 2
    required_buffer = dilation + half_window
    
    # Calculate valid midpoint range
    midpoint_start = required_buffer
    midpoint_end = len(session_data) - required_buffer
    
    if midpoint_start >= midpoint_end:
        print(f"    Signal too short for dilated windows (need {2*required_buffer} samples, have {len(session_data)})")
        return None
    
    # Get valid midpoint positions
    midpoint_positions = list(range(midpoint_start, midpoint_end + 1, step_size))
    
    if not midpoint_positions:
        print(f"    No valid dilated windows")
        return None
    
    # Calculate dilated RMS for all channels at each valid midpoint position
    results = []
    
    for midpoint in midpoint_positions:
        # Calculate window positions around midpoint (fixed calculations)
        before_start = midpoint - dilation - half_window
        before_end = midpoint - dilation
        
        center_start = midpoint - half_window
        center_end = midpoint + half_window
        
        after_start = midpoint + dilation
        after_end = midpoint + dilation + half_window
        
        # Calculate dilated RMS for each channel
        rms_values = {}
        
        for channel in available_channels:
            signal = session_data[channel].values
            
            # Extract and concatenate the three windows
            window_before = signal[before_start:before_end]
            window_center = signal[center_start:center_end] 
            window_after = signal[after_start:after_end]
            
            # Concatenate and calculate single RMS
            combined_signal = np.concatenate([window_before, window_center, window_after])
            dilated_rms = np.sqrt(np.mean(combined_signal**2))
            
            # Store with simplified column name (CH1_rms instead of CH1_voltage_rms)
            channel_num = channel.split('_')[0]  # Extract CH1, CH2, etc.
            rms_values[f'{channel_num}_rms'] = dilated_rms
        
        # Store feature vector for this time window
        result_row = {
            'session_id': session_id,
            'label': label,
            't': midpoint
        }
        result_row.update(rms_values)
        
        results.append(result_row)
    
    result_df = pd.DataFrame(results)
    
    # Apply z-score normalization to RMS columns within this session
    rms_columns = [col for col in result_df.columns if col.endswith('_rms')]
    if rms_columns:
        for col in rms_columns:
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            if std_val > 0:
                result_df[col] = (result_df[col] - mean_val) / std_val
            else:
                result_df[col] = 0  # Handle case where std is 0
    
    return result_df


def process_all_sessions(window_size=400, step_size=100, dilation=100, apply_pca=True, n_components=None):
    """Process all sessions from processed parquet database and calculate dilated RMS features
    
    Args:
        window_size: Size of center window in samples
        step_size: Step between windows in samples  
        dilation: Dilation gap in samples
        apply_pca: Whether to apply PCA dimensionality reduction to RMS features
        n_components: Number of PCA components (None for auto-selection based on variance)
        
    Returns:
        Path to output parquet file
    """
    print("Dilated RMS Feature Extraction from Parquet Database")
    print("=" * 60)
    print(f"Window size: {window_size} samples")
    print(f"Step size: {step_size} samples")
    print(f"Dilation gap: {dilation} samples")
    print(f"Apply PCA: {apply_pca}")
    if apply_pca:
        print(f"PCA components: {n_components or 'auto-select'}")
    
    # Ensure features directory exists
    FEATURES_DIR.mkdir(exist_ok=True)
    
    # Generate output filename
    output_filename = f"emg_len{window_size}_dil{dilation}_step{step_size}.parquet"
    output_parquet = FEATURES_DIR / output_filename
    
    print(f"Input: {INPUT_PARQUET}")
    print(f"Output: {output_parquet}")
    
    # Check if input file exists
    if not INPUT_PARQUET.exists():
        print(f"Error: Input file {INPUT_PARQUET} not found!")
        print("Please run prep.py first to create the processed database.")
        return None
    
    # Read data from parquet using DuckDB
    print("\n1. Loading processed data from parquet...")
    conn = duckdb.connect()
    
    # Load the data
    df = conn.execute(f"""
        SELECT * FROM '{INPUT_PARQUET.as_posix()}'
        ORDER BY id, time_index
    """).df()
    
    print(f"   Loaded {len(df):,} records")
    print(f"   Sessions: {df['id'].nunique()}")
    print(f"   Labels: {sorted(df['label'].unique())}")
    
    # Process each session separately
    print(f"\n2. Processing sessions...")
    all_features = []
    unique_sessions = df['id'].unique()
    processed_sessions = 0
    
    for i, session_id in enumerate(unique_sessions, 1):
        session_data = df[df['id'] == session_id].copy()
        
        if i % 100 == 0:  # Print progress every 100 sessions
            print(f"   Processing session {i}/{len(unique_sessions)}...")
        
        try:
            features = process_session_data(session_data, window_size, step_size, dilation)
            
            if features is not None:
                all_features.append(features)
                processed_sessions += 1
            
        except Exception as e:
            print(f"    Error processing session {session_id}: {e}")
            continue
    
    # Combine all results
    if all_features:
        print(f"\n3. Combining results from {len(all_features)} sessions...")
        combined_features = pd.concat(all_features, ignore_index=True)
        
        print(f"   Total feature vectors: {len(combined_features)}")
        
        # Apply PCA to RMS features if requested
        if apply_pca:
            print(f"\n4. Applying PCA dimensionality reduction...")
            
            # Get RMS columns
            rms_columns = [col for col in combined_features.columns if col.endswith('_rms')]
            print(f"   Original RMS features: {len(rms_columns)} ({rms_columns})")
            
            if len(rms_columns) > 0:
                # Extract RMS feature matrix
                X_rms = combined_features[rms_columns].values
                
                # Standardize features before PCA
                scaler = StandardScaler()
                X_rms_scaled = scaler.fit_transform(X_rms)
                
                # Apply PCA
                if n_components is None:
                    # Auto-select components to retain 95% of variance
                    pca = PCA()
                    pca.fit(X_rms_scaled)
                    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
                    n_components = np.argmax(cumsum_var >= 0.95) + 1
                    print(f"   Auto-selected {n_components} components to retain 95% variance")
                
                # Apply PCA with selected number of components
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_rms_scaled)
                
                print(f"   PCA components: {n_components}")
                print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
                print(f"   Total variance retained: {pca.explained_variance_ratio_.sum():.3f}")
                
                # Replace RMS columns with PCA components
                # Remove original RMS columns
                combined_features = combined_features.drop(columns=rms_columns)
                
                # Add PCA components
                for i in range(n_components):
                    combined_features[f'PC{i+1}'] = X_pca[:, i]
                
                print(f"   Replaced {len(rms_columns)} RMS features with {n_components} PCA components")
            else:
                print("   Warning: No RMS columns found for PCA!")
        
        # Save to parquet using DuckDB
        step_num = "5" if apply_pca else "4"
        print(f"\n{step_num}. Saving features to {output_parquet}...")
        
        conn.register('features_temp', combined_features)
        
        # Create column list for proper typing
        if apply_pca and len([col for col in combined_features.columns if col.startswith('PC')]) > 0:
            # Use PCA columns
            feature_columns = [col for col in combined_features.columns if col.startswith('PC')]
        else:
            # Use original RMS columns
            feature_columns = [col for col in combined_features.columns if col.endswith('_rms')]
        
        select_cols = [
            "session_id::BIGINT as session_id",
            "label::INTEGER as label", 
            "t::INTEGER as t"
        ]
        select_cols.extend([f"{col}::DOUBLE as {col}" for col in feature_columns])
        
        conn.execute(f"""
            COPY (
                SELECT {', '.join(select_cols)}
                FROM features_temp
                ORDER BY session_id, t
            ) TO '{output_parquet.as_posix()}' (FORMAT PARQUET)
        """)
        
        conn.close()
        
        print(f"Dilated RMS Feature Extraction Complete!")
        print(f"Processed sessions: {processed_sessions}")
        print(f"Total feature vectors: {len(combined_features)}")
        print(f"Final feature columns: {len(feature_columns)} ({feature_columns if len(feature_columns) <= 10 else feature_columns[:10] + ['...']})")
        print(f"Results saved: {output_parquet}")
        print(f"File size: {output_parquet.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_parquet
    else:
        print("No features were generated!")
        conn.close()
        return None


if __name__ == "__main__":
    # Multi-step analysis configuration
    window_sizes = [200, 300, 400, 500, 600]
    dilations = [50, 100, 150, 200, 250]
    step_sizes = [100, 200, 300, 400]  # Multiple step sizes for comprehensive analysis
    
    print("Starting comprehensive multi-step analysis...")
    print(f"Window sizes: {window_sizes}")
    print(f"Dilations: {dilations}")
    print(f"Step sizes: {step_sizes}")
    
    total_combinations = len(window_sizes) * len(dilations) * len(step_sizes)
    print(f"Total combinations: {total_combinations}")
    
    completed = 0
    
    for step_size in step_sizes:
        print(f"\n{'='*80}")
        print(f"PROCESSING STEP SIZE: {step_size}")
        print(f"{'='*80}")
        
        step_completed = 0
        step_total = len(window_sizes) * len(dilations)
        
        for i, win_size in enumerate(window_sizes):
            for j, dilation in enumerate(dilations):
                completed += 1
                step_completed += 1
                
                print(f"\n[STEP {step_size}] [{step_completed}/{step_total}] [OVERALL {completed}/{total_combinations}]")
                print(f"Processing: window_size={win_size}, dilation={dilation}, step_size={step_size}")
                
                try:
                    output_file = process_all_sessions(window_size=win_size, step_size=step_size, dilation=dilation)
                    if output_file:
                        print(f"SUCCESS: Completed: {output_file.name}")
                    else:
                        print(f"FAILED: Failed for parameters: win={win_size}, dil={dilation}, step={step_size}")
                except Exception as e:
                    print(f"ERROR: Error with parameters win={win_size}, dil={dilation}, step={step_size}: {e}")
                
                print("-" * 60)
        
        print(f"Step size {step_size} complete: {step_completed} configurations processed")
    
    print(f"\n{'='*80}")
    print(f"MULTI-STEP ANALYSIS COMPLETE!")
    print(f"Total configurations processed: {completed}")
    print(f"Step sizes analyzed: {step_sizes}")
    print(f"{'='*80}")
