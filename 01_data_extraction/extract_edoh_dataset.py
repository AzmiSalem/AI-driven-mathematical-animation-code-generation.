#!/usr/bin/env python3
"""
Extract Edoh dataset from Hugging Face for Code2Prompt training.

This script downloads the Edoh dataset from Hugging Face and saves it
in the format needed for the training pipeline.
"""

import json
from datasets import load_dataset
from pathlib import Path

def extract_edoh_dataset():
    """Extract and save the Edoh dataset."""
    print("Loading Edoh dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("Edoh/code2prompt")
    
    # Extract train and test splits
    train_data = dataset['train']
    test_data = dataset['test']
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(test_data)} test examples")
    
    # Combine all data
    all_data = []
    
    # Add training data
    for example in train_data:
        all_data.append({
            "instruction": example["instruction"],
            "output": example["output"]
        })
    
    # Add test data
    for example in test_data:
        all_data.append({
            "instruction": example["instruction"],
            "output": example["output"]
        })
    
    print(f"Total examples: {len(all_data)}")
    
    # Save combined dataset
    output_file = Path("../04_datasets/edoh_all_original.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Saved Edoh dataset to: {output_file}")
    
    # Also save train and test separately
    train_file = Path("../04_datasets/edoh_train_original.jsonl")
    test_file = Path("../04_datasets/edoh_test_original.jsonl")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_data:
            f.write(json.dumps({
                "instruction": example["instruction"],
                "output": example["output"]
            }, ensure_ascii=False) + '\n')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for example in test_data:
            f.write(json.dumps({
                "instruction": example["instruction"],
                "output": example["output"]
            }, ensure_ascii=False) + '\n')
    
    print(f"Saved training data to: {train_file}")
    print(f"Saved test data to: {test_file}")
    
    # Display sample
    print("\nSample from Edoh dataset:")
    sample = all_data[0]
    print(f"Instruction: {sample['instruction'][:100]}...")
    print(f"Output: {sample['output'][:100]}...")
    
    return len(all_data)

if __name__ == "__main__":
    extract_edoh_dataset()
