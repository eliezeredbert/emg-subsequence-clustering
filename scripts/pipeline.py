import subprocess
import sys
from pathlib import Path
import time
import argparse

class PipelineError(Exception):
    """Custom exception for pipeline failures"""
    pass

def run_script(script_name, description, script_args=None):
    """Run a script and handle errors - raises PipelineError on failure"""
    print(f"Running: {description}")
    
    start_time = time.time()
    
    try:
        # Build command with arguments
        cmd = [sys.executable, script_name]
        if script_args:
            cmd.extend(script_args)
        
        # For scripts in the scripts folder, run them from their directory
        if script_name.startswith("scripts/"):
            script_path = Path(script_name)
            cmd[1] = script_path.name  # Use just the script name
            result = subprocess.run(cmd, cwd=script_path.parent, capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"  ✓ SUCCESS ({elapsed:.1f}s)")
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"  ✗ FAILED ({elapsed:.1f}s)")
        print(f"  Return code: {e.returncode}")
        if e.stderr:
            print(f"  Error: {e.stderr.strip()}")
        if e.stdout:
            print(f"  Output: {e.stdout.strip()}")
        raise PipelineError(f"{description} failed with return code {e.returncode}")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ EXCEPTION ({elapsed:.1f}s): {e}")
        raise PipelineError(f"{description} failed with exception: {e}")

def main(start_step=1, num_files=50):
    """Run the complete EMG clustering pipeline"""
    
    print("EMG CLUSTERING PIPELINE")
    if start_step > 1:
        print(f"Starting from step {start_step}")
    print(f"Dataset balance: {num_files} files per digit")
    print("="*50)
    
    start_pipeline = time.time()
    
    try:
        # Step 1: Dataset balancing
        if start_step <= 1:
            run_script("balance_dataset.py", "Balancing dataset...", ["-n", str(num_files)])
        
        # Step 2: Database preparation
        if start_step <= 2:
            run_script("db.py", "Preparing database...")

        # Step 3: Data preprocessing  
        if start_step <= 3:
            run_script("prep.py", "Preprocessing data...")
        # Step 4: Feature extraction
        if start_step <= 4:
            run_script("features.py", "Extracting features...")
        
        # Step 5: Cluster analysis (window/dilation/step optimization)
        if start_step <= 5:
            run_script("cluster_win_dil_step.py", "Looking for Best Window and Dilation in each Step Size...")
        
        # Step 6: Detailed clustering (k optimization)
        if start_step <= 6:
            run_script("cluster_k.py", "Clustering with best parameter for each time step...")
        
        # Step 7: Generate visualizations
        if start_step <= 7:
            run_script("plot_clusters.py", "Generating plots...")
        
        # Step 8: Generate phonics patterns from cluster dictionaries
        if start_step <= 8:
            run_script("phonics.py", "Generating phonics patterns...")
        # Step 9: Analyze phonics results and rank parameters
        if start_step <= 9:
            run_script("best.py", "Analyzing phonics results and ranking parameters...")
        
        # Success!
        total_time = time.time() - start_pipeline
        print(f"\n{'='*50}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print("="*50)
        
    except PipelineError as e:
        total_time = time.time() - start_pipeline
        print(f"\n{'='*50}")
        print("PIPELINE FAILED!")
        print(f"Error: {e}")
        print(f"Time elapsed: {total_time/60:.1f} minutes")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run EMG clustering pipeline')
    parser.add_argument('--step', '-s', type=int, default=1, choices=range(1, 10),
                        help='Start pipeline from specific step (1-9). Default: 1 (start from beginning)')
    parser.add_argument('-n', '--num-files', type=int, default=50,
                        help='Number of files per digit for dataset balancing (only used if starting from step 1)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.step > 1 and args.num_files != 50:
        print(f"Warning: -n argument ignored when starting from step {args.step} (dataset balancing skipped)")
    
    main(args.step, args.num_files)