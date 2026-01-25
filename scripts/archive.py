"""
Archive EMG analysis results by moving directories into archive folder
Usage: python archive.py <archive_name>

This script moves:
- data/processed -> archive/<archive_name>/processed
- data/features -> archive/<archive_name>/features
- result/ -> archive/<archive_name>/result

This allows you to preserve analysis results before running new experiments.
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
import errno

def archive_results(archive_name):
    """Archive analysis results by renaming directories"""
    
    # Define paths (relative to parent directory)
    base_dir = Path("..").resolve()
    archive_dir = base_dir / "archive" / archive_name
    processed_source = base_dir / "data" / "processed"
    processed_target = archive_dir / "processed"
    features_source = base_dir / "data" / "features"
    features_target = archive_dir / "features"
    result_source = base_dir / "result"
    result_target = archive_dir / "result"
    
    print(f"\nARCHIVE EMG ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"Archive name: {archive_name}")
    print(f"Archive directory: {archive_dir}")
    print()
    
    archived_items = []
    errors = []
    
    # Create archive directory if it doesn't exist
    if archive_dir.exists():
        print(f"❌ Archive directory already exists: {archive_dir}")
        print("Choose a different archive name or remove the existing archive.")
        return False
    
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created archive directory: {archive_dir}")
    except Exception as e:
        print(f"❌ Error creating archive directory: {e}")
        return False
    
    # Archive data/processed directory
    if processed_source.exists():
        try:
            shutil.move(str(processed_source), str(processed_target))
            print(f"✅ Archived: data/processed -> archive/{archive_name}/processed")
            archived_items.append(f"archive/{archive_name}/processed")
        except Exception as e:
            if isinstance(e, OSError) and e.errno == errno.EACCES:
                print(f"❌ Archive process canceled: data/processed is being used by another process")
                return False
            else:
                print(f"❌ Error archiving data/processed: {e}")
                errors.append(f"Failed to archive data/processed: {e}")
    else:
        print(f"⚠️  Source not found: {processed_source}")
    
    # Archive data/features directory
    if features_source.exists():
        try:
            shutil.move(str(features_source), str(features_target))
            print(f"✅ Archived: data/features -> archive/{archive_name}/features")
            archived_items.append(f"archive/{archive_name}/features")
        except Exception as e:
            if isinstance(e, OSError) and e.errno == errno.EACCES:
                print(f"❌ Archive process canceled: data/features is being used by another process")
                return False
            else:
                print(f"❌ Error archiving data/features: {e}")
                errors.append(f"Failed to archive data/features: {e}")
    else:
        print(f"⚠️  Source not found: {features_source}")
    
    # Archive result directory  
    if result_source.exists():
        try:
            shutil.move(str(result_source), str(result_target))
            print(f"✅ Archived: result/ -> archive/{archive_name}/result")
            archived_items.append(f"archive/{archive_name}/result")
        except Exception as e:
            if isinstance(e, OSError) and e.errno == errno.EACCES:
                print(f"❌ Archive process canceled: result/ is being used by another process")
                return False
            else:
                print(f"❌ Error archiving result/: {e}")
                errors.append(f"Failed to archive result/: {e}")
    else:
        print(f"⚠️  Source not found: {result_source}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"ARCHIVE SUMMARY")
    print(f"{'='*50}")
    
    if archived_items:
        print(f"✅ Successfully archived {len(archived_items)} directories:")
        for item in archived_items:
            print(f"   - {item}")
    
    if errors:
        print(f"\n❌ Errors encountered:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if not archived_items:
        print("⚠️  No directories were archived (none found or all targets exist)")
        return False
    
    print(f"\n🎉 Archive '{archive_name}' completed successfully!")
    print("You can now run new analyses without overwriting previous results.")
    
    return True

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Archive EMG analysis results by renaming directories',
        epilog="""
Examples:
  python archive.py balanced_20    # Archive as archive/balanced_20/processed, archive/balanced_20/features, and archive/balanced_20/result
  python archive.py avg_resample   # Archive as archive/avg_resample/processed, archive/avg_resample/features, and archive/avg_resample/result
  python archive.py test_run_1     # Archive as archive/test_run_1/processed, archive/test_run_1/features, and archive/test_run_1/result
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('archive_name', 
                       help='Name suffix for archived directories (e.g., "balanced_20")')
    
    args = parser.parse_args()
    
    # Validate archive name
    archive_name = args.archive_name.strip()
    if not archive_name:
        print("Error: Archive name cannot be empty")
        return 1
    
    # Check for invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in archive_name for char in invalid_chars):
        print(f"Error: Archive name contains invalid characters: {invalid_chars}")
        return 1
    
    # Perform archiving
    success = archive_results(archive_name)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())