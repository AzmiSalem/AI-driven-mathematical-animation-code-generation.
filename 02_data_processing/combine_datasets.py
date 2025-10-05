#!/usr/bin/env python3
"""
Script to combine and clean datasets for LoRA fine-tuning.
This script combines the enhanced Edoh dataset with Manim documentation data.
"""

import json
import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    
    if source == 'enhanced_edoh':
        # From enhanced_prompts_edoh_results.json
        standardized['instruction'] = entry.get('enhanced_prompt', entry.get('original_instruction', ''))
        standardized['output'] = entry.get('original_code', '')
    elif source == 'manim_docs':
        # From manim_docs_html_pairs files
        standardized['instruction'] = entry.get('instruction', '')
        standardized['output'] = entry.get('code', '')
    
    return standardized

def combine_datasets():
    """Main function to combine and clean datasets."""
    print("Starting dataset combination process...")
    
    # Load datasets
    print("Loading enhanced Edoh dataset...")
    enhanced_edoh_data = load_json('enhanced_prompts_edoh_results.json')
    print(f"Loaded {len(enhanced_edoh_data)} entries from enhanced Edoh dataset")
    
    print("Loading Manim documentation dataset...")
    manim_docs_data = load_jsonl('manim_docs_html_pairs.sample.jsonl')
    print(f"Loaded {len(manim_docs_data)} entries from Manim docs dataset")
    
    # Clean datasets
    print("\nCleaning enhanced Edoh dataset...")
    cleaned_edoh = clean_dataset(enhanced_edoh_data)
    
    print("\nCleaning Manim docs dataset...")
    cleaned_manim = clean_dataset(manim_docs_data)
    
    # Standardize formats
    print("\nStandardizing data formats...")
    standardized_edoh = [standardize_format(entry, 'enhanced_edoh') for entry in cleaned_edoh]
    standardized_manim = [standardize_format(entry, 'manim_docs') for entry in cleaned_manim]
    
    # Combine datasets
    print("\nCombining datasets...")
    combined_data = standardized_edoh + standardized_manim
    
    # Shuffle the combined dataset
    import random
    random.shuffle(combined_data)
    
    print(f"Combined dataset size: {len(combined_data)} entries")
    print(f"  - Enhanced Edoh: {len(standardized_edoh)} entries")
    print(f"  - Manim Docs: {len(standardized_manim)} entries")
    
    # Save combined dataset
    output_file = 'combined_manim_dataset.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCombined dataset saved to: {output_file}")
    
    # Create train/test split (80/20)
    split_idx = int(len(combined_data) * 0.8)
    train_data = combined_data[:split_idx]
    test_data = combined_data[split_idx:]
    
    train_file = 'train_dataset.json'
    test_file = 'test_dataset.json'
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Train dataset saved to: {train_file} ({len(train_data)} entries)")
    print(f"Test dataset saved to: {test_file} ({len(test_data)} entries)")
    
    # Display sample entries
    print("\nSample entries from combined dataset:")
    for i, entry in enumerate(combined_data[:3]):
        print(f"\nEntry {i+1}:")
        print(f"Instruction: {entry['instruction'][:100]}...")
        print(f"Output: {entry['output'][:100]}...")
        print(f"Source: {entry['source']}")

if __name__ == "__main__":
    combine_datasets()
