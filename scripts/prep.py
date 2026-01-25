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


def process_emg_data():
    """
    Process EMG data from parquet database:
    1. Read from data/db/emg_data.parquet
    2. Convert digital values to voltage
    3. Apply filtering (bandpass + PLI notch)
    4. Apply DC offset correction
    5. Resample to uniform length per label
    6. Save to data/processed/emg_processed.parquet
    """
    DB_DIR.mkdir(exist_ok=True)
    PROCESSED_DB_DIR.mkdir(exist_ok=True)

    if not INPUT_PARQUET.exists():
        print(f"Error: Input file {INPUT_PARQUET} not found!")
        print("Run db.py first to create the database.")
        return

    df = create_dataframe_from_parquet(INPUT_PARQUET)
    target_resampling_length = get_average_length_per_label(df)
    emg_channels_column_headers = [f"CH{i}" for i in range(1, 9)]  # CH1 through CH8
    processed_sessions = []
    unique_sessions = df["id"].unique()

    for i, session_id in enumerate(unique_sessions, 1):
        session_data = df[df["id"] == session_id].copy()
        label = session_data["label"].iloc[0]

        print(
            f"\n3. Processing session {i}/{len(unique_sessions)}: ID {session_id}, Label {label}"
        )

        target_length = target_resampling_length[label]
        current_length = len(session_data)

        voltage_data = convert_digital_to_voltage(
            session_data, emg_channels_column_headers
        )
        filtered_data = filter_voltage_data(emg_channels_column_headers, voltage_data)
        offset_corrected = apply_dc_offset_correction(
            session_id, emg_channels_column_headers, filtered_data
        )

        if current_length != target_length:
            resampled_session = resample(
                session_id,
                label,
                emg_channels_column_headers,
                offset_corrected,
                target_length,
            )
            processed_sessions.append(resampled_session)
        else:
            processed_sessions.append(offset_corrected)

    final_data = pd.concat(processed_sessions, ignore_index=True)
    export_emg_data_to_parquet(final_data)
    print(f"Output file: {OUTPUT_PARQUET}")

    return OUTPUT_PARQUET


def apply_filter(data, fs, low_freq, high_freq, notch_freq=60, notch_q=30):
    """
    Apply bandpass filter and PLI notch filter to data array
    Data array is 1D numpy array, representing EMG signal from one channel.
    """
    # Apply bandpass filter
    sos_bp = signal.butter(4, [low_freq, high_freq], btype="band", fs=fs, output="sos")
    filtered_data = signal.sosfiltfilt(sos_bp, data)

    # Apply PLI notch filter at 60Hz - convert to SOS format
    b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, fs)
    sos_notch = signal.tf2sos(b_notch, a_notch)
    filtered_data = signal.sosfiltfilt(sos_notch, filtered_data)

    return filtered_data


def create_dataframe_from_parquet(parquet_path):
    """
    Create a pandas DataFrame from a parquet file using DuckDB for efficient reading.
    """
    with duckdb.connect() as conn:
        df = conn.execute(f"SELECT * FROM '{parquet_path.as_posix()}'").fetchdf()
    return df


def export_emg_data_to_parquet(final_data):
    with duckdb.connect() as conn:
        conn.register("processed_data_temp", final_data)

        conn.execute(
            f"""
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
        """
        )


def summarize_processed_data():
    print("\n=== Processing Summary ===")
    with duckdb.connect() as conn:
        # Basic stats
        result = conn.execute(
            f"""
            SELECT 
                COUNT(DISTINCT id) as unique_sessions,
                COUNT(DISTINCT label) as unique_labels
            FROM '{OUTPUT_PARQUET.as_posix()}'
        """
        ).fetchone()

        print(f"Unique sessions: {result[0]}")
        print(f"Unique labels: {result[1]}")

        # Voltage range statistics for each channel
        print(f"\nVoltage ranges per channel:")
        for ch in range(1, 9):
            col_name = f"CH{ch}_voltage"
            stats = conn.execute(
                f"""
                SELECT 
                    MIN({col_name}) as min_v,
                    MAX({col_name}) as max_v,
                    AVG({col_name}) as avg_v,
                    STDDEV({col_name}) as std_v
                FROM '{OUTPUT_PARQUET.as_posix()}'
            """
            ).fetchone()

            print(
                f"  CH{ch}: {stats[0]:.4f}V to {stats[1]:.4f}V (avg: {stats[2]:.4f}V, std: {stats[3]:.4f}V)"
            )


def resample(
    session_id, label, emg_channels_column_headers, offset_corrected, target_length
):
    resampled_session = offset_corrected.iloc[:1].copy()  # Keep first row for structure
    resampled_session = pd.concat(
        [resampled_session] * target_length, ignore_index=True
    )
    resampled_session["id"] = session_id
    resampled_session["label"] = label
    resampled_session["time_index"] = range(target_length)

    # Resample each channel
    for channel in emg_channels_column_headers:
        if channel in offset_corrected.columns:
            original_signal = offset_corrected[channel].values
            resampled_signal = signal.resample(original_signal, target_length)
            resampled_session[channel] = resampled_signal
    return resampled_session


def apply_dc_offset_correction(session_id, emg_channels_column_headers, filtered_data):
    offset_corrected = filtered_data.copy()
    offset_info = {}

    for channel in emg_channels_column_headers:
        if channel in filtered_data.columns:
            try:
                data = filtered_data[channel].values
                dc_offset = np.mean(data)
                offset_info[channel] = dc_offset
                offset_corrected[channel] = data - dc_offset
            except Exception as e:
                print(
                    f"Error correcting DC offset of {session_id} channel {channel}: {e}"
                )

    return offset_corrected


def filter_voltage_data(emg_channels_column_headers, voltage_data):
    filtered_data = voltage_data.copy()
    for channel in emg_channels_column_headers:
        if channel in voltage_data.columns:
            try:
                data = voltage_data[channel].values
                filtered_values = apply_filter(data, FREQUENCY, BPF_LOW, BPF_HIGH)
                filtered_data[channel] = filtered_values
            except Exception as e:
                print(f"      Error filtering {channel}: {e}")
    return filtered_data


def convert_digital_to_voltage(session_data, emg_channels_column_headers):
    voltage_data = session_data.copy()
    for channel in emg_channels_column_headers:
        if channel in session_data.columns:
            digital_values = session_data[channel].values
            voltage_values = (V_HIGH - V_LOW) * digital_values / ADC_MAX + V_LOW
            voltage_data[channel] = voltage_values
    return voltage_data


def get_average_length_per_label(df):
    target_resampling_length = {}
    for label in df["label"].unique():
        label_data = df[df["label"] == label]
        session_lengths = label_data.groupby("id").size()
        avg_length = int(round(session_lengths.mean()))  # Round to nearest integer
        target_resampling_length[label] = avg_length
        print(
            f"   Target length for label {label}: {avg_length} samples (from {session_lengths.min()} to {session_lengths.max()})"
        )

    return target_resampling_length


if __name__ == "__main__":
    # Process EMG data from parquet database
    output_file = process_emg_data()
    summarize_processed_data()
