"""
Converts DAQ output CSV files to DuckDB SQL table saved as parquet format.

Schema:
- id: timestamp extracted from filename (format: YYYYMMDDHHMMSSMMM)
- label: label digit extracted from filename (e.g., 'num6' -> 6)
- time_index: row index representing time (sampling rate = 4096Hz)
- CH1-CH8: EMG data
"""

import os
import re
import logging
import duckdb
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory configuration
RAW_DATA_DIR = Path("../data/raw")
DB_DIR = Path("../data/db")
DB_FILE = "emg_data.parquet"



def create_database_from_csv_files():
    # Create output directory
    DB_DIR.mkdir(exist_ok=True)
    
    # Output parquet file path
    output_parquet_path = DB_DIR / DB_FILE
    
    # Check if CSV files exist
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")
    
    logger.info(f"Processing all CSV files directly with DuckDB...")
    logger.info(f"Saving to: {output_parquet_path}")
    
    try:
        with duckdb.connect() as conn:
            # Process all CSV files in a single DuckDB operation
            conn.execute(f"""
                COPY (
                    SELECT 
                        regexp_extract(filename, '(\\d+)_num(\\d+)\\.csv', 1)::BIGINT as id,
                        regexp_extract(filename, '(\\d+)_num(\\d+)\\.csv', 2)::INTEGER as label,
                        (row_number() OVER (PARTITION BY filename ORDER BY (SELECT 1)) - 1)::INTEGER as time_index,
                        CH1::INTEGER as CH1,
                        CH2::INTEGER as CH2,
                        CH3::INTEGER as CH3,
                        CH4::INTEGER as CH4,
                        CH5::INTEGER as CH5,
                        CH6::INTEGER as CH6,
                        CH7::INTEGER as CH7,
                        CH8::INTEGER as CH8
                    FROM read_csv_auto('{RAW_DATA_DIR.as_posix()}/*.csv', filename=true)
                    WHERE filename ~ '.*_num\\d+\\.csv$'  -- Filter valid filenames only
                    ORDER BY id, time_index
                ) TO '{output_parquet_path.as_posix()}' (FORMAT PARQUET)
            """)
            
        logger.info(f"Successfully processed and saved EMG data to: {output_parquet_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to process CSV files with DuckDB: {e}") from e


def print_database_stats():
    parquet_file = DB_DIR / DB_FILE
    
    print("=== Database Stats ===")
    
    try:
        with duckdb.connect() as conn:
            # Query 1: Basic statistics
            print("\n1. Dataset Overview:")
            result = conn.execute(f"""
                SELECT 
                    COUNT(DISTINCT id) as unique_sessions,
                    COUNT(DISTINCT label) as unique_labels,
                FROM '{parquet_file.as_posix()}'
            """).fetchone()
            
            print(f"   Unique sessions: {result[0]}")
            print(f"   Unique labels: {result[1]}")
            
            # Query 2: label distribution
            print("\n2. Label Distribution:")
            results = conn.execute(f"""
                SELECT 
                    label,
                    COUNT(DISTINCT id) as session_count
                FROM '{parquet_file.as_posix()}'
                GROUP BY label
                ORDER BY label
            """).fetchall()
            
            for label, session_count in results:
                print(f"   Label {label}: {session_count} sessions")
    except Exception as e:
        raise RuntimeError(f"Failed to query database statistics: {e}") from e


if __name__ == "__main__":
    # Convert raw data to database
    create_database_from_csv_files()
    
    # Check if database file exists before printing stats
    parquet_file = DB_DIR / DB_FILE
    if not parquet_file.exists():
        raise FileNotFoundError(f"Database file not found: {parquet_file}")
    
    # Print database statistics
    print_database_stats()
