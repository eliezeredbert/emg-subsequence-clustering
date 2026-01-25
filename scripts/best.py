"""
Analyze phonics results to find best parameters based on:
1. No duplicate phonics between labels
2. Lowest average phonics character length
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
import sys
import io

# Directory constants
DICT_RESULTS_DIR = Path("../result/dict")

# Force UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_parameters_from_filename(filename):
    """Extract win, dil, step, k from phonics filename"""
    # Expected format: step200_win400_dil150_k5_phonics.csv
    pattern = r'step(\d+)_win(\d+)_dil(\d+)_k(\d+)_phonics\.csv'
    match = re.match(pattern, filename)
    
    if match:
        step = int(match.group(1))
        win = int(match.group(2))
        dil = int(match.group(3))
        k = int(match.group(4))
        return step, win, dil, k
    else:
        return None, None, None, None

def analyze_phonics_file(file_path):
    """Analyze a single phonics file for duplicates and average length"""
    try:
        df = pd.read_csv(file_path)
        
        # Calculate average phonics length
        if 'phonics' in df.columns:
            phonics_lengths = df['phonics'].str.len()
            avg_length = phonics_lengths.mean()
            
            # Check for duplicates
            phonics_counts = Counter(df['phonics'])
            has_duplicates = any(count > 1 for count in phonics_counts.values())
            
            # Count how many labels share the same phonics
            duplicate_count = sum(1 for count in phonics_counts.values() if count > 1)
            
            return {
                'avg_length': avg_length,
                'has_duplicates': has_duplicates,
                'duplicate_count': duplicate_count,
                'total_labels': len(df),
                'unique_phonics': len(phonics_counts)
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error analyzing {file_path.name}: {e}")
        return None

def create_phonics_analysis_table():
    """Create analysis table for all phonics files"""
    
    print("PHONICS PARAMETER ANALYSIS")
    print("=" * 60)
    print(f"Scanning directory: {DICT_RESULTS_DIR.absolute()}")
    
    # Find all phonics files
    phonics_files = list(DICT_RESULTS_DIR.glob("step*_phonics.csv"))
    
    if not phonics_files:
        print("No phonics files found!")
        return
    
    print(f"Found {len(phonics_files)} phonics files")
    print()
    
    results = []
    
    for file_path in phonics_files:
        # Extract parameters from filename
        step, win, dil, k = extract_parameters_from_filename(file_path.name)
        
        if step is None:
            print(f"Warning: Could not parse filename {file_path.name}")
            continue
        
        # Analyze the phonics file
        analysis = analyze_phonics_file(file_path)
        
        if analysis is None:
            print(f"Warning: Could not analyze {file_path.name}")
            continue
        
        results.append({
            'step': step,
            'win': win,
            'dil': dil,
            'k': k,
            'avg_length': analysis['avg_length'],
            'has_duplicates': analysis['has_duplicates'],
            'duplicate_count': analysis['duplicate_count'],
            'total_labels': analysis['total_labels'],
            'unique_phonics': analysis['unique_phonics'],
            'filename': file_path.name
        })
        
        print(f"Processed: {file_path.name} - Avg Length: {analysis['avg_length']:.2f}, Duplicates: {'Yes' if analysis['has_duplicates'] else 'No'}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No valid results to analyze")
        return
    
    # Sort by: no duplicates first, then by lowest average length
    results_df = results_df.sort_values(['has_duplicates', 'avg_length'], ascending=[True, True])
    
    # Add rank column
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # Reorder columns to put rank first
    cols = ['rank'] + [col for col in results_df.columns if col != 'rank']
    results_df = results_df[cols]
    
    # Display results table
    print("\n" + "=" * 120)
    print("PHONICS ANALYSIS RESULTS (Best Parameters at Top)")
    print("=" * 120)
    print("Sorted by: No Duplicates First → Lowest Average Length")
    print()
    
    # Format table for display
    print(f"{'Rank':<4} {'Step':<4} {'Win':<3} {'Dil':<3} {'K':<2} {'Avg Len':<7} {'Duplicates':<10} {'Dup Count':<9} {'Total Labels':<12} {'Unique Phonics':<14} {'Filename'}")
    print("-" * 120)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        duplicates_str = "No" if not row['has_duplicates'] else "Yes"
        print(f"{i:<4} {row['step']:<4} {row['win']:<3} {row['dil']:<3} {row['k']:<2} "
              f"{row['avg_length']:<7.2f} {duplicates_str:<10} {row['duplicate_count']:<9} "
              f"{row['total_labels']:<12} {row['unique_phonics']:<14} {row['filename']}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    no_duplicate_files = results_df[~results_df['has_duplicates']]
    duplicate_files = results_df[results_df['has_duplicates']]
    
    print(f"Total configurations analyzed: {len(results_df)}")
    print(f"Configurations with no duplicates: {len(no_duplicate_files)}")
    print(f"Configurations with duplicates: {len(duplicate_files)}")
    
    if len(no_duplicate_files) > 0:
        best_no_dup = no_duplicate_files.iloc[0]
        print(f"\nBest configuration (no duplicates):")
        print(f"  Step: {best_no_dup['step']}, Win: {best_no_dup['win']}, Dil: {best_no_dup['dil']}, K: {best_no_dup['k']}")
        print(f"  Average phonics length: {best_no_dup['avg_length']:.2f}")
        print(f"  File: {best_no_dup['filename']}")
    
    if len(results_df) > 0:
        overall_best = results_df.iloc[0]
        print(f"\nOverall best configuration:")
        print(f"  Step: {overall_best['step']}, Win: {overall_best['win']}, Dil: {overall_best['dil']}, K: {overall_best['k']}")
        print(f"  Average phonics length: {overall_best['avg_length']:.2f}")
        print(f"  Has duplicates: {'Yes' if overall_best['has_duplicates'] else 'No'}")
        print(f"  File: {overall_best['filename']}")
    
    # Save results to CSV
    output_file = DICT_RESULTS_DIR / "rank.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return results_df

if __name__ == "__main__":
    create_phonics_analysis_table()
