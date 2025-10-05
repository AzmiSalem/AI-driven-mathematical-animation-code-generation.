#!/usr/bin/env python3
"""
Script to combine the full datasets:
- manim_docs_html_pairs.all.jsonl (990 examples)
- edoh_all_original.jsonl (651 examples)

This will create a much larger training dataset.
"""

import json
import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Any
import random

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def contains_no_visible_scene(entry: Dict[str, Any]) -> bool:
    """Check if entry contains 'No visible scene' in any text field."""
    text_fields = ['instruction', 'output', 'code', 'enhanced_prompt', 'original_instruction', 'original_code']
    
    for field in text_fields:
        if field in entry and isinstance(entry[field], str):
            if 'No visible scene' in entry[field]:
                return True
    return False

def clean_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove entries containing 'No visible scene'."""
    cleaned_data = []
    removed_count = 0
    
    for entry in data:
        if contains_no_visible_scene(entry):
            removed_count += 1
            print(f"Removing entry: {entry.get('instruction', 'No instruction')[:50]}...")
        else:
            cleaned_data.append(entry)
    
    print(f"Removed {removed_count} entries containing 'No visible scene'")
    print(f"Remaining entries: {len(cleaned_data)}")
    return cleaned_data

def standardize_format(entry: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Standardize data format for instruction-following training."""
    standardized = {
        'instruction': '',
        'output': '',
        'source': source
    }
    
    if source == 'edoh':
        # From edoh_all_original.jsonl
        standardized['instruction'] = entry.get('instruction', '')
        standardized['output'] = entry.get('output', '')
    elif source == 'manim_docs':
        # From manim_docs_html_pairs.all.jsonl
        standardized['instruction'] = entry.get('instruction', '')
        standardized['output'] = entry.get('code', '')
    
    return standardized

def combine_full_datasets():
    """Main function to combine the full datasets."""
    print("Starting FULL dataset combination process...")
    print("=" * 60)
    
    # Load full datasets
    print("Loading Edoh full dataset...")
    edoh_data = load_jsonl('edoh_all_original.jsonl')
    print(f"Loaded {len(edoh_data)} entries from Edoh full dataset")
    
    print("Loading Manim documentation full dataset...")
    manim_docs_data = load_jsonl('manim_docs_html_pairs.all.jsonl')
    print(f"Loaded {len(manim_docs_data)} entries from Manim docs full dataset")
    
    # Clean datasets
    print("\nCleaning Edoh dataset...")
    cleaned_edoh = clean_dataset(edoh_data)
    
    print("\nCleaning Manim docs dataset...")
    cleaned_manim = clean_dataset(manim_docs_data)
    
    # Standardize formats
    print("\nStandardizing data formats...")
    standardized_edoh = [standardize_format(entry, 'edoh') for entry in cleaned_edoh]
    standardized_manim = [standardize_format(entry, 'manim_docs') for entry in cleaned_manim]
    
    # Combine datasets
    print("\nCombining datasets...")
    combined_data = standardized_edoh + standardized_manim
    
    # Shuffle the combined dataset
    random.shuffle(combined_data)
    
    print(f"Combined dataset size: {len(combined_data)} entries")
    print(f"  - Edoh: {len(standardized_edoh)} entries")
    print(f"  - Manim Docs: {len(standardized_manim)} entries")
    
    # Save combined dataset
    output_file = 'combined_full_manim_dataset.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCombined dataset saved to: {output_file}")
    
    # Create train/test split (80/20)
    split_idx = int(len(combined_data) * 0.8)
    train_data = combined_data[:split_idx]
    test_data = combined_data[split_idx:]
    
    train_file = 'train_full_dataset.json'
    test_file = 'test_full_dataset.json'
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Train dataset saved to: {train_file} ({len(train_data)} entries)")
    print(f"Test dataset saved to: {test_file} ({len(test_data)} entries)")
    
    # Display sample entries
    print("\nSample entries from combined dataset:")
    for i, entry in enumerate(combined_data[:3]):
        print(f"\nEntry {i+1} (Source: {entry['source']}):")
        print(f"Instruction: {entry['instruction'][:100]}...")
        print(f"Output: {entry['output'][:100]}...")
    
    # Dataset statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS:")
    print(f"Total examples: {len(combined_data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Edoh examples: {len(standardized_edoh)}")
    print(f"Manim docs examples: {len(standardized_manim)}")
    
    # Average instruction and output lengths
    avg_instruction_len = sum(len(entry['instruction']) for entry in combined_data) / len(combined_data)
    avg_output_len = sum(len(entry['output']) for entry in combined_data) / len(combined_data)
    
    print(f"Average instruction length: {avg_instruction_len:.1f} characters")
    print(f"Average output length: {avg_output_len:.1f} characters")
    
    print("\n" + "=" * 60)
    print("✅ Full dataset combination completed successfully!")
    print("Ready for LoRA fine-tuning with much more data!")

if __name__ == "__main__":
    combine_full_datasets()
