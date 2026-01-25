import os
import pandas as pd
import numpy as np
from scipy import signal
import sys
import duckdb
from pathlib import Path

# Add parent folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define constants (extracted from config for this database version)
ADC_MAX = 4096
V_LOW = -5
V_HIGH = 5
FREQUENCY = 4000  # 4000Hz sampling rate as mentioned in db.py
BPF_LOW = 15
BPF_HIGH = 400

# Directory and file paths
DB_DIR = Path("../data/db")
PROCESSED_DB_DIR = Path("../data/processed")
INPUT_PARQUET = DB_DIR / "emg_data.parquet"
OUTPUT_PARQUET = PROCESSED_DB_DIR / "emg_processed.parquet"


def apply_filter(data, fs, low_freq, high_freq, notch_freq=60, notch_q=30):
    """
    Apply bandpass filter and PLI notch filter to data array
    Data array is 1D numpy array, representing EMG signal from one channel.
    """
    # Apply bandpass filter
    sos_bp = signal.butter(4, [low_freq, high_freq], btype='band', fs=fs, output='sos')
    filtered_data = signal.sosfiltfilt(sos_bp, data)
    
    # Apply PLI notch filter at 60Hz - convert to SOS format
    b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, fs)
    sos_notch = signal.tf2sos(b_notch, a_notch)
    filtered_data = signal.sosfiltfilt(sos_notch, filtered_data)
    
    return filtered_data


def process_emg_data():
    """
    Process EMG data from parquet database:
    1. Read from data/db/emg_data.parquet
    2. Convert digital values to voltage
    3. Apply filtering (bandpass + PLI notch)
    4. Apply DC offset correction
    5. Save to data/db/emg_processed.parquet
    """
    # Ensure directories exist
    DB_DIR.mkdir(exist_ok=True)
    PROCESSED_DB_DIR.mkdir(exist_ok=True)
    
    # Check if input file exists
    if not INPUT_PARQUET.exists():
        print(f"Error: Input file {INPUT_PARQUET} not found!")
        print("Run db.py first to create the database.")
        return
    
    # Read data from parquet using DuckDB
    print("\n1. Loading data from parquet...")
    conn = duckdb.connect()
    
    # Load the data - get all unique sessions for processing
    df = conn.execute(f"""
        SELECT * FROM '{INPUT_PARQUET.as_posix()}'
        ORDER BY id, time_index
    """).df()
    
    # Process each session separately to maintain data structure
    processed_sessions = []
    unique_sessions = df['id'].unique()
    
    for i, session_id in enumerate(unique_sessions, 1):
        session_data = df[df['id'] == session_id].copy()
        label = session_data['label'].iloc[0]
         
        print(f"\n2. Processing session {i}/{len(unique_sessions)}: ID {session_id}, Label {label}")
        
        # Extract channel columns
        channel_cols = [f'CH{i}' for i in range(1, 9)]  # CH1 through CH8
        
        # Step 1: Convert digital values to voltage
        voltage_data = session_data.copy()
        for col in channel_cols:
            if col in session_data.columns:
                digital_values = session_data[col].values
                voltage_values = (V_HIGH - V_LOW) * digital_values / ADC_MAX + V_LOW
                voltage_data[col] = voltage_values
        
        # Step 2: Apply filtering
        filtered_data = voltage_data.copy()
        for col in channel_cols:
            if col in voltage_data.columns:
                try:
                    data = voltage_data[col].values
                    filtered_values = apply_filter(data, FREQUENCY, BPF_LOW, BPF_HIGH)
                    filtered_data[col] = filtered_values
                    print(f"      Filtered {col}")
                except Exception as e:
                    print(f"      Error filtering {col}: {e}")
        
        # Step 3: Apply DC offset correction
        offset_corrected = filtered_data.copy()
        offset_info = {}
        
        for col in channel_cols:
            if col in filtered_data.columns:
                try:
                    data = filtered_data[col].values
                    dc_offset = np.mean(data)
                    offset_info[col] = dc_offset
                    offset_corrected[col] = data - dc_offset
                except Exception as e:
                    print(f"Error correcting DC offset of {session_id} channel {col}: {e}")
        
        processed_sessions.append(offset_corrected)
    
    # Combine all processed sessions
    print(f"\n3. Combining {len(processed_sessions)} processed sessions...")
    final_data = pd.concat(processed_sessions, ignore_index=True)
    
    # Save processed data to parquet
    print(f"\n4. Saving processed data")
    
    # Use DuckDB to save with proper data types
    conn.register('processed_data_temp', final_data)
    
    conn.execute(f"""
        COPY (
            SELECT 
                id::BIGINT as id,
                label::INTEGER as label,
                time_index::INTEGER as time_index,
                CH1::DOUBLE as CH1_voltage,
                CH2::DOUBLE as CH2_voltage,
                CH3::DOUBLE as CH3_voltage,
                CH4::DOUBLE as CH4_voltage,
                CH5::DOUBLE as CH5_voltage,
                CH6::DOUBLE as CH6_voltage,
                CH7::DOUBLE as CH7_voltage,
                CH8::DOUBLE as CH8_voltage
            FROM processed_data_temp
            ORDER BY id, time_index
        ) TO '{OUTPUT_PARQUET.as_posix()}' (FORMAT PARQUET)
    """)
    
    conn.close()
    
    print(f"Data preprocessing complete!")
    print(f"Output file: {OUTPUT_PARQUET}") 
    
    # Display summary statistics
    print("\n=== Processing Summary ===")
    conn = duckdb.connect()
    
    # Basic stats
    result = conn.execute(f"""
        SELECT 
            COUNT(DISTINCT id) as unique_sessions,
            COUNT(DISTINCT label) as unique_labels
        FROM '{OUTPUT_PARQUET.as_posix()}'
    """).fetchone()
    
    print(f"Unique sessions: {result[0]}")
    print(f"Unique labels: {result[1]}")
    
    # Voltage range statistics for each channel
    print(f"\nVoltage ranges per channel:")
    for ch in range(1, 9):
        col_name = f'CH{ch}_voltage'
        stats = conn.execute(f"""
            SELECT 
                MIN({col_name}) as min_v,
                MAX({col_name}) as max_v,
                AVG({col_name}) as avg_v,
                STDDEV({col_name}) as std_v
            FROM '{OUTPUT_PARQUET.as_posix()}'
        """).fetchone()
        
        print(f"  CH{ch}: {stats[0]:.4f}V to {stats[1]:.4f}V (avg: {stats[2]:.4f}V, std: {stats[3]:.4f}V)")
    
    conn.close()
    
    # Step 5: Resample data to ensure same length per label (using average length)
    print(f"\n5. Resampling data to uniform length per label (using average length)...")
    resample_data(OUTPUT_PARQUET)
    
    return OUTPUT_PARQUET


def resample_data(parquet_file):
    """
    Resample all sessions to have the same length within each label
    Uses average length per label instead of minimum length
    """
    from scipy import signal as scipy_signal
    
    conn = duckdb.connect()
    
    # Load processed data
    df = conn.execute(f"""
        SELECT * FROM '{parquet_file.as_posix()}'
        ORDER BY id, time_index
    """).df()
    
    # Find average length for each label
    print("   Finding average length per label...")
    label_lengths = {}
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        session_lengths = label_data.groupby('id').size()
        avg_length = int(round(session_lengths.mean()))  # Round to nearest integer
        label_lengths[label] = avg_length
        print(f"     Label {label}: avg length = {avg_length} samples (from {session_lengths.min()} to {session_lengths.max()})")
    
    # Resample each session to its label's minimum length
    resampled_sessions = []
    channel_cols = [f'CH{i}_voltage' for i in range(1, 9)]
    
    for session_id in df['id'].unique():
        session_data = df[df['id'] == session_id].copy()
        label = session_data['label'].iloc[0]
        target_length = label_lengths[label]
        current_length = len(session_data)
        
        if current_length != target_length:
            print(f"     Resampling session {session_id} (label {label}): {current_length} -> {target_length}")
            
            # Create new resampled session with correct structure
            resampled_session = session_data.iloc[:1].copy()  # Keep first row for structure
            resampled_session = pd.concat([resampled_session] * target_length, ignore_index=True)
            resampled_session['id'] = session_id
            resampled_session['label'] = label
            resampled_session['time_index'] = range(target_length)
            
            # Resample each channel
            for col in channel_cols:
                if col in session_data.columns:
                    original_signal = session_data[col].values
                    resampled_signal = scipy_signal.resample(original_signal, target_length)
                    resampled_session[col] = resampled_signal
            
            resampled_sessions.append(resampled_session)
        else:
            print(f"     Session {session_id} (label {label}): already correct length ({current_length})")
            resampled_sessions.append(session_data)
    
    # Combine resampled sessions
    resampled_df = pd.concat(resampled_sessions, ignore_index=True)
    
    # Save resampled data back to parquet
    conn.register('resampled_data_temp', resampled_df)
    
    conn.execute(f"""
        COPY (
            SELECT 
                id::BIGINT as id,
                label::INTEGER as label,
                time_index::INTEGER as time_index,
                CH1_voltage::DOUBLE as CH1_voltage,
                CH2_voltage::DOUBLE as CH2_voltage,
                CH3_voltage::DOUBLE as CH3_voltage,
                CH4_voltage::DOUBLE as CH4_voltage,
                CH5_voltage::DOUBLE as CH5_voltage,
                CH6_voltage::DOUBLE as CH6_voltage,
                CH7_voltage::DOUBLE as CH7_voltage,
                CH8_voltage::DOUBLE as CH8_voltage
            FROM resampled_data_temp
            ORDER BY id, time_index
        ) TO '{parquet_file.as_posix()}' (FORMAT PARQUET)
    """)
    
    conn.close()
    print("   Resampling complete!")
    
    # Show final statistics
    print("\n=== Final Resampled Data Summary ===")
    conn = duckdb.connect()
    
    for label in sorted(label_lengths.keys()):
        count = conn.execute(f"""
            SELECT COUNT(DISTINCT id) 
            FROM '{parquet_file.as_posix()}' 
            WHERE label = {label}
        """).fetchone()[0]
        length = label_lengths[label]
        print(f"Label {label}: {count} sessions, {length} samples each")
    
    conn.close()


if __name__ == "__main__":
    # Process EMG data from parquet database
    output_file = process_emg_data()
