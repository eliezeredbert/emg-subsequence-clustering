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


def calculate_midpoint_positions(
    session_length, center_window_samples, step_size, dilation
):
    """Calculate valid midpoint positions for dilated window processing.

    Args:
        session_length: Number of samples in the session
        center_window_samples: Size of center window in samples
        step_size: Step between windows in samples
        dilation: Dilation gap in samples

    Returns:
        List of valid midpoint positions, or None if signal is too short
    """
    # Calculate required buffer for dilated windows
    half_window_samples = center_window_samples // 2
    buffer_samples = dilation + half_window_samples

    # Calculate valid midpoint range
    midpoint_start = buffer_samples
    midpoint_end = session_length - buffer_samples

    if midpoint_start >= midpoint_end:
        return None  # Signal too short for dilated windows

    # Get valid midpoint positions
    return list(range(midpoint_start, midpoint_end + 1, step_size))


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
    session_id = session_data["id"].iloc[0]
    label = session_data["label"].iloc[0]
    session_length = len(session_data)

    # Extract channel columns (these should be CH1_voltage, CH2_voltage, etc. from processed data)
    emg_channels_header_name = [f"CH{i}_voltage" for i in range(1, 9)]

    # Check which channels are available
    channels_in_session_data = [
        col for col in emg_channels_header_name if col in session_data.columns
    ]

    if not channels_in_session_data:
        return None

    # Sort by time_index to ensure proper order
    session_data = session_data.sort_values("time_index").reset_index(drop=True)

    # Calculate valid midpoint positions for dilated windows
    midpoint_positions = calculate_midpoint_positions(
        session_length, window_size, step_size, dilation
    )

    if midpoint_positions is None:
        return None

    # Calculate dilated RMS for all channels at each valid midpoint position
    session_df = generate_dilated_rms_dataframe(
        session_data,
        window_size,
        dilation,
        session_id,
        label,
        channels_in_session_data,
        midpoint_positions,
    )

    # Apply z-score normalization to RMS columns within this session
    return apply_zscore_normalization(session_df)


def apply_zscore_normalization(session_df):
    rms_header_in_session_data = [
        col for col in session_df.columns if col.endswith("_rms")
    ]
    if rms_header_in_session_data:
        for col in rms_header_in_session_data:
            mean_val = session_df[col].mean()
            std_val = session_df[col].std()
            if std_val > 0:
                session_df[col] = (session_df[col] - mean_val) / std_val
            else:
                session_df[col] = 0  # Handle case where std is 0

    return session_df


def generate_dilated_rms_dataframe(
    session_data,
    window_size,
    dilation,
    session_id,
    label,
    channels_in_session_data,
    midpoint_positions,
):
    session_data_in_rms = []

    for midpoint in midpoint_positions:
        # Calculate window positions around midpoint (fixed calculations)
        before_start = midpoint - dilation - window_size // 2
        before_end = midpoint - dilation

        center_start = midpoint - window_size // 2
        center_end = midpoint + window_size // 2

        after_start = midpoint + dilation
        after_end = midpoint + dilation + window_size // 2

        # Calculate dilated RMS for each channel
        rms_values = {}

        for channel in channels_in_session_data:
            emg_signal_in_one_channel = session_data[channel].values

            # Extract and concatenate the three windows
            window_before = emg_signal_in_one_channel[before_start:before_end]
            window_center = emg_signal_in_one_channel[center_start:center_end]
            window_after = emg_signal_in_one_channel[after_start:after_end]

            # Concatenate and calculate single RMS
            combined_signal = np.concatenate(
                [window_before, window_center, window_after]
            )
            dilated_rms = np.sqrt(np.mean(combined_signal**2))

            # Store with simplified column name (CH1_rms instead of CH1_voltage_rms)
            channel_num = channel.split("_")[0]  # Extract CH1, CH2, etc.
            rms_values[f"{channel_num}_rms"] = dilated_rms

        # Store feature vector for this time window
        session_data_in_rms.append(
            {"session_id": session_id, "label": label, "t": midpoint, **rms_values}
        )

    session_df = pd.DataFrame(session_data_in_rms)
    return session_df


def process_all_sessions(
    window_size=400, step_size=100, dilation=100, apply_pca=True, n_components=None
):
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
    FEATURES_DIR.mkdir(exist_ok=True)

    output_filename = f"emg_len{window_size}_dil{dilation}_step{step_size}.parquet"
    output_parquet = FEATURES_DIR / output_filename

    if not INPUT_PARQUET.exists():
        print(f"Error: Input file {INPUT_PARQUET} not found!")
        print("Please run prep.py first to create the processed database.")
        return None

    df = get_dataframe_from_parquet()

    session_dfs = []
    unique_sessions = df["id"].unique()
    count_processed_sessions = 0

    for i, session_id in enumerate(unique_sessions, 1):
        session_data = df[df["id"] == session_id].copy()

        try:
            session_df = process_session_data(
                session_data, window_size, step_size, dilation
            )

            if session_df is not None:
                session_dfs.append(session_df)
                count_processed_sessions += 1

        except Exception as e:
            print(f"Error processing session {session_id}: {e}")
            continue

    # Combine all results
    if not session_dfs:
        print("No sessions were processed successfully.")
        return None

    features_df = pd.concat(session_dfs, ignore_index=True)

    # Apply PCA to RMS features if requested
    if apply_pca:
        features_df = apply_pca_transformation(n_components, features_df)

    # Save results to parquet file
    export_features_to_parquet(apply_pca, output_parquet, features_df)

    return output_parquet


def export_features_to_parquet(apply_pca, output_parquet, features_df):
    with duckdb.connect() as conn:
        # Register the DataFrame as a temporary table in DuckDB
        conn.register("features_temp", features_df)

        # Determine which columns to save (PCA components or RMS features)
        if apply_pca and any(col.startswith("PC") for col in features_df.columns):
            # Use PCA component columns
            data_columns = [col for col in features_df.columns if col.startswith("PC")]
        else:
            # Use original RMS feature columns
            data_columns = [col for col in features_df.columns if col.endswith("_rms")]

        # Build SQL column specifications with proper data types
        # DuckDB needs explicit type casting for parquet export
        sql_columns = [
            "session_id::BIGINT as session_id",  # Session ID as big integer
            "label::INTEGER as label",  # Label as integer
            "t::INTEGER as t",  # Time position as integer
        ]

        # Add feature columns as double precision floating point
        for col in data_columns:
            sql_columns.append(f"{col}::DOUBLE as {col}")

        # Execute SQL query to export data to parquet
        sql_query = f"""
            COPY (
                SELECT {', '.join(sql_columns)}
                FROM features_temp
                ORDER BY session_id, t
            ) TO '{output_parquet.as_posix()}' (FORMAT PARQUET)
        """

        conn.execute(sql_query)


def apply_pca_transformation(n_components, features_df):
    rms_columns_headers = [col for col in features_df.columns if col.endswith("_rms")]

    if len(rms_columns_headers) <= 0:
        return features_df

    rms_feature_array = features_df[rms_columns_headers].values
    scaler = StandardScaler()
    standardized_rms_array = scaler.fit_transform(rms_feature_array)

    # Auto-select components to retain 95% of variance
    if n_components is None:
        pca = PCA()
        pca.fit(standardized_rms_array)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= 0.95) + 1

    # Apply PCA with selected number of components
    pca = PCA(n_components=n_components)
    pca_transformed_features = pca.fit_transform(standardized_rms_array)

    # Replace RMS columns with PCA components
    features_df = features_df.drop(columns=rms_columns_headers)

    for i in range(n_components):
        features_df[f"PC{i+1}"] = pca_transformed_features[:, i]

    return features_df


def get_dataframe_from_parquet():
    with duckdb.connect() as conn:
        # Load the data
        df = conn.execute(
            f"""
            SELECT * FROM '{INPUT_PARQUET.as_posix()}'
            ORDER BY id, time_index
        """
        ).df()

    return df


def run_parameter_sweep():
    """Run parameter sweep for different window sizes, dilations, and step sizes"""
    window_sizes = [200, 300, 400, 500, 600]
    dilations = [50, 100, 150, 200, 250]
    step_sizes = [100, 200, 300, 400]

    total_combinations = len(window_sizes) * len(dilations) * len(step_sizes)
    completed = 0

    for step_size in step_sizes:
        step_completed = 0
        step_total = len(window_sizes) * len(dilations)

        for i, win_size in enumerate(window_sizes):
            for j, dilation in enumerate(dilations):
                completed += 1
                step_completed += 1

                print(
                    f"\n[STEP {step_size}] [{step_completed}/{step_total}] [OVERALL {completed}/{total_combinations}]"
                )
                print(
                    f"Processing: window_size={win_size}, dilation={dilation}, step_size={step_size}"
                )

                try:
                    output_file = process_all_sessions(
                        window_size=win_size, step_size=step_size, dilation=dilation
                    )
                    if output_file:
                        print(f"SUCCESS: Completed: {output_file.name}")
                    else:
                        print(
                            f"FAILED: Failed for parameters: win={win_size}, dil={dilation}, step={step_size}"
                        )
                except Exception as e:
                    print(
                        f"ERROR: Error with parameters win={win_size}, dil={dilation}, step={step_size}: {e}"
                    )

                print("-" * 60)

        print(
            f"Step size {step_size} complete: {step_completed} configurations processed"
        )

    print(f"\n{'='*80}")
    print(f"MULTI-STEP ANALYSIS COMPLETE!")
    print(f"Total configurations processed: {completed}")
    print(f"Step sizes analyzed: {step_sizes}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_parameter_sweep()
