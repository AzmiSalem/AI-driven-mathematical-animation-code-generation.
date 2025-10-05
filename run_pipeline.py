#!/usr/bin/env python3
"""
Complete Code2Prompt pipeline runner.

This script runs the entire pipeline from data extraction to training.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def main():
    """Run the complete pipeline."""
    print("Code2Prompt Complete Pipeline Runner")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("01_data_extraction").exists():
        print("ERROR: Please run this script from the code2prompt-clean directory")
        sys.exit(1)
    
    # Step 1: Extract Edoh dataset
    if not run_command(
        "cd 01_data_extraction && python extract_edoh_dataset.py",
        "Extracting Edoh dataset from Hugging Face"
    ):
        print("Failed to extract Edoh dataset")
        sys.exit(1)
    
    # Step 2: Process Manim docs (requires OpenAI API key)
    print("\n" + "="*60)
    print("IMPORTANT: Manim docs processing requires OpenAI API key")
    print("Please ensure you have:")
    print("1. OpenAI API key file (e.g., openai_api_key.txt)")
    print("2. Manim docs snippets file (manim_docs_snippets.jsonl)")
    print("="*60)
    
    # Check if API key file exists
    api_key_file = "openai_api_key.txt"
    if not Path(api_key_file).exists():
        print(f"WARNING: {api_key_file} not found. Skipping Manim docs processing.")
        print("To process Manim docs, create the API key file and run:")
        print("cd 02_data_processing && python format_snippets_with_openai.py --api_key_file ../openai_api_key.txt")
    else:
        if not run_command(
            f"cd 02_data_processing && python format_snippets_with_openai.py --api_key_file ../{api_key_file}",
            "Processing Manim docs with OpenAI"
        ):
            print("Failed to process Manim docs")
    
    # Step 3: Combine datasets
    if not run_command(
        "cd 02_data_processing && python combine_full_datasets.py",
        "Combining Edoh and Manim datasets"
    ):
        print("Failed to combine datasets")
        sys.exit(1)
    
    # Step 4: Training (optional)
    print("\n" + "="*60)
    print("TRAINING STEP")
    print("="*60)
    print("Training requires GPU and can take 2-4 hours.")
    print("To run training:")
    print("cd 03_training && python colab_training_complete.py")
    print("Or use the Jupyter notebook:")
    print("jupyter notebook Code2Prompt_Colab_Training.ipynb")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)
    print("Files created:")
    print("- 04_datasets/edoh_all_original.jsonl (Edoh dataset)")
    print("- 04_datasets/manim_docs_html_pairs.all.jsonl (Manim docs)")
    print("- 04_datasets/train_full_dataset.json (Combined dataset)")
    print("\nNext steps:")
    print("1. Review the datasets in 04_datasets/")
    print("2. Run training: cd 03_training && python colab_training_complete.py")
    print("3. Check results in 05_results/")

if __name__ == "__main__":
    main()
