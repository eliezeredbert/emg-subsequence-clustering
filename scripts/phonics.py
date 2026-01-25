import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Directory and file paths
DICT_RESULTS_DIR = Path("../result/dict")


def group_consecutive_repeats(sequence):
    """Group consecutive repeating numbers

    Example: [0,1,1,1,2,2,2,3,3,1,1] -> [0,1,2,3,1]
    """
    if len(sequence) == 0:
        return []

    grouped = [sequence[0]]

    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            grouped.append(sequence[i])

    return grouped


def numbers_to_alphabet(sequence):
    """Convert numbers to alphabet letters

    0->a, 1->b, 2->c, 3->d, 4->e, etc.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return [alphabet[num] if num < len(alphabet) else f"#{num}" for num in sequence]


def process_dictionary_file(file_path, output_dir=None):
    """Process a single dictionary CSV file"""

    print(f"Processing: {file_path.name}")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Get time columns (all columns except 'label')
    time_columns = [col for col in df.columns if col != "label"]

    results = []

    for _, row in df.iterrows():
        label = int(row["label"])  # Convert to integer

        # Extract sequence values, excluding NaN values and empty strings
        sequence_values = []
        for col in time_columns:
            val = row[col]
            if pd.notna(val) and val != "":
                try:
                    sequence_values.append(int(val))
                except (ValueError, TypeError):
                    # Skip values that can't be converted to int
                    continue

        if len(sequence_values) == 0:
            print(f"  Warning: No valid data for label {label}")
            continue

        # Group consecutive repeats
        grouped_sequence = group_consecutive_repeats(sequence_values)

        # Convert to alphabet
        alphabet_sequence = numbers_to_alphabet(grouped_sequence)

        # Join into string
        alphabet_string = "".join(alphabet_sequence)

        results.append({"label": label, "phonics": alphabet_string})

        print(
            f"  Label {label}: {sequence_values[:10]}... -> {grouped_sequence} -> {alphabet_string}"
        )

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Generate output filename and directory
    input_stem = file_path.stem  # e.g., "step200_win400_dil150_k5_dictionary"
    output_filename = input_stem.replace("_dictionary", "_phonics") + ".csv"

    # Use custom output directory or default to same folder as input
    if output_dir is None:
        output_dir = file_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / output_filename

    # Save results
    results_df.to_csv(output_file, index=False)

    print(f"  Saved: {output_file.name}")
    print(f"  Processed {len(results)} labels")
    print("-" * 60)

    return output_file


def process_all_dictionary_files():
    """Process all dictionary CSV files in the default directory"""

    print("PHONICS PROCESSING - Converting Cluster Sequences to Alphabet Patterns")
    print("=" * 80)

    # Use default input folder
    dictionary_dir = DICT_RESULTS_DIR
    print(f"Using default input folder: {dictionary_dir.absolute()}")

    # Find all dictionary files
    dictionary_files = list(dictionary_dir.glob("step*_dictionary.csv"))

    if not dictionary_files:
        print(f"No dictionary files found in {dictionary_dir}!")
        return

    print(f"Found {len(dictionary_files)} dictionary files to process")

    # Output to same folder as input files
    print("Output: Same folder as input files")

    print()

    processed_files = []

    for file_path in sorted(dictionary_files):
        try:
            output_file = process_dictionary_file(file_path, None)
            processed_files.append(output_file)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            print("-" * 60)

    print("=" * 80)
    print("PHONICS PROCESSING COMPLETE")
    print(f"Processed {len(processed_files)} files successfully")
    print()
    print("Generated files:")
    for file_path in processed_files:
        print(f"  - {file_path}")
    print("=" * 80)


def main():
    """Main function"""
    process_all_dictionary_files()


if __name__ == "__main__":
    main()
