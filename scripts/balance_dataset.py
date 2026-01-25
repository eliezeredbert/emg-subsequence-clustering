"""
Balance EMG dataset by copying N oldest files per digit label from raw_source to raw folder
"""
import os
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

# Directory constants
RAW_SOURCE_DIR = Path('../data/raw_source')
RAW_DIR = Path('../data/raw')

def balance_dataset(num_files_to_keep=50):
    """Copy N oldest files per digit label from raw_source to raw directory"""
    
    # Ensure directories exist
    if not RAW_SOURCE_DIR.exists():
        print(f'Error: Source directory does not exist: {RAW_SOURCE_DIR}')
        return
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear existing raw directory first
    if RAW_DIR.exists():
        existing_files = list(RAW_DIR.glob('*.csv'))
        if existing_files:
            for existing_file in existing_files:
                existing_file.unlink()
            print(f'Cleared {len(existing_files)} existing files from raw directory')
    
    # Get all CSV files in raw_source directory
    source_files = list(RAW_SOURCE_DIR.glob('*.csv'))
    
    print(f'Found {len(source_files)} total files in source directory')
    print(f'Target: copying {num_files_to_keep} oldest files per digit to raw directory')
    
    # Group files by digit label
    files_by_digit = defaultdict(list)
    for file in source_files:
        # Extract digit from filename like '20250618174456980_num2.csv'
        if '_num' in file.name:
            try:
                digit = int(file.name.split('_num')[1].split('.')[0])
                timestamp = file.name.split('_num')[0]
                files_by_digit[digit].append((timestamp, file))
            except (ValueError, IndexError):
                print(f"Skipping invalid filename: {file.name}")
                continue
    
    print('\nFiles per digit (available in source):') 
    for digit in sorted(files_by_digit.keys()):
        print(f'  Digit {digit}: {len(files_by_digit[digit])} files')
    
    # Select oldest files per digit to copy
    files_to_copy = []
    for digit in sorted(files_by_digit.keys()):
        files_list = files_by_digit[digit]
        # Sort by timestamp (oldest first)
        files_list.sort(key=lambda x: x[0])
        # Take first N (oldest) files
        to_copy = files_list[:min(num_files_to_keep, len(files_list))]
        files_to_copy.extend([f[1] for f in to_copy])
        print(f'\nDigit {digit}: copying {len(to_copy)} oldest files (of {len(files_list)} available)')
        print('  Copying:')
        for i, (ts, f) in enumerate(to_copy):
            if i < 5:  # Show first 5 for brevity
                print(f'    {i+1}. {f.name}')
            elif i == 5:
                print(f'    ... and {len(to_copy)-5} more')
    
    print(f'\n=== SUMMARY ===')
    print(f'Total files to copy: {len(files_to_copy)}')
    
    if files_to_copy:
        print(f'\nCopying {len(files_to_copy)} files to raw directory...')
        copied_count = 0
        for file in files_to_copy:
            try:
                dest_file = RAW_DIR / file.name
                shutil.copy2(file, dest_file)
                print(f'  Copied: {file.name}')
                copied_count += 1
            except Exception as e:
                print(f'  Error copying {file.name}: {e}')
        
        print(f'\nSuccessfully copied {copied_count} files')
        
        # Show final distribution
        final_files = list(RAW_DIR.glob('*.csv'))
        final_by_digit = defaultdict(int)
        for file in final_files:
            if '_num' in file.name:
                try:
                    digit = int(file.name.split('_num')[1].split('.')[0])
                    final_by_digit[digit] += 1
                except (ValueError, IndexError):
                    continue
        
        print('\nFiles per digit (in raw directory):') 
        for digit in sorted(final_by_digit.keys()):
            print(f'  Digit {digit}: {final_by_digit[digit]} files')
    else:
        print('No files to copy - no source files found')

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Balance EMG dataset by copying N oldest files per digit from raw_source to raw')
    parser.add_argument('-n', '--num-files', type=int, default=50, 
                       help='Number of oldest files to copy per digit (default: 50)')
    
    args = parser.parse_args()
    balance_dataset(args.num_files)

if __name__ == "__main__":
    main()