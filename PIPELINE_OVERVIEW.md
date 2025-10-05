# Pipeline Overview

This document provides a detailed overview of the complete  fine-tuning pipeline.

## Pipeline Architecture

```
Raw Data Sources → Data Extraction → Data Processing → Training → Results
     ↓                    ↓                ↓            ↓         ↓
Hugging Face     →  Edoh Dataset    →  Combined    →  LoRA    →  Trained
Manim Docs       →  Manim Snippets  →  Dataset     →  Training →  Model
```

## Step 1: Data Extraction (01_data_extraction/)

### Edoh Dataset Extraction
- **Source**: Hugging Face dataset repository
- **Script**: `extract_edoh_dataset.py`
- **Output**: `edoh_all_original.jsonl` (651 pairs)
- **Format**: `{"instruction": "...", "output": "..."}`
- **Note**: Uses original Edoh dataset without enhancement

### Manim Documentation Extraction
- **Source**: Manim documentation website
- **Script**: `extract_manim_docs_snippets.py`
- **Output**: `manim_docs_snippets.jsonl` (raw code snippets)
- **Format**: `{"code": "...", "before_context": "...", "after_context": "...", "url": "..."}`

## Step 2: Data Processing (02_data_processing/)

### HTML + LLM Enhancement
- **Script**: `format_snippets_with_openai.py`
- **Process**:
  1. Extract code snippets from Manim documentation
  2. Use BeautifulSoup for HTML parsing
  3. Send to GPT-4o-mini with 200-word context
  4. Generate concise instructions (≤60 words)
- **Output**: `manim_docs_html_pairs.all.jsonl` (989 pairs)
- **Format**: `{"instruction": "...", "code": "...", "url": "...", "depth": 0}`

### Dataset Combination
- **Script**: `combine_full_datasets.py`
- **Process**:
  1. Load original Edoh dataset (651 pairs, no enhancement)
  2. Load Manim docs dataset (989 pairs)
  3. Standardize formats
  4. Remove "No visible scene" entries
  5. Shuffle and split (80/20 train/test)
- **Output**: `train_full_dataset.json` (1,640 pairs)

## Step 3: Training (03_training/)

### LoRA Configuration
```python
CONFIG = {
    "model_name": "LFM2.5-1.5B",
    "max_length": 1024,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1
}
```

### Training Process
1. **Model Loading**: Load LFM2.5-1.5B with LoRA adapters
2. **Dataset Preparation**: Tokenize instruction→code pairs
3. **Training**: Fine-tune with LoRA on combined dataset
4. **Evaluation**: Test on held-out data
5. **Model Saving**: Save LoRA adapters and merged model

### Training Scripts
- `colab_training_complete.py` - Complete training implementation
- `Code2Prompt_Colab_Training.ipynb` - Jupyter notebook version

## Step 4: Datasets (04_datasets/)

### Raw Datasets
- `edoh_all_original.jsonl` - Original Edoh dataset (651 pairs, no enhancement)
- `manim_docs_snippets.jsonl` - Raw Manim code snippets

### Processed Datasets
- `manim_docs_html_pairs.all.jsonl` - Manim docs with LLM-generated instructions (989 pairs)
- `train_full_dataset.json` - Combined training dataset (1,640 pairs)

### Sample Datasets
- `manim_docs_html_pairs.sample.jsonl` - Sample of Manim docs for testing

## Step 5: Results (05_results/)

### Training Results
- `TRAINING_RESULTS.md` - Detailed training analysis
- Model performance metrics
- Before/after comparison
- Deployment instructions

## Data Flow Details

### Instruction Generation Process
1. **Input**: Raw code snippet + context
2. **LLM Prompt**: "Describe the visual result in ≤60 words"
3. **Output**: Concise instruction describing the animation

### Training Format
```
Instruction: Create a blue circle that rotates clockwise
Code:
from manim import *

class RotatingCircle(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(Create(circle))
        self.play(Rotate(circle, angle=2*PI), run_time=2)
        self.wait()
```

### Quality Control
- Remove entries with "No visible scene"
- Validate instruction length (≤60 words)
- Ensure code compiles and runs
- Shuffle datasets to prevent overfitting

## Performance Metrics

### Dataset Statistics
- **Total Examples**: 1,640 pairs
- **Training Split**: ~1,500 pairs (80%)
- **Test Split**: ~140 pairs (20%)
- **Average Instruction Length**: 50-100 words
- **Average Code Length**: 100-300 lines

### Training Metrics
- **Training Time**: 2-4 hours (Google Colab)
- **Memory Usage**: ~8GB VRAM
- **Model Size**: ~1.7GB (base + LoRA adapters)
- **Convergence**: 3 epochs

## Deployment Options

1. **Google Colab**: Ready-to-run notebook
2. **Local GPU**: Direct Python execution
3. **Cloud**: AWS/GCP with GPU instances
4. **Docker**: Containerized deployment

## Key Innovations

1. **Dual Dataset Approach**: Combines original Edoh dataset with custom Manim docs
2. **HTML + LLM Processing**: Automated instruction generation from code snippets
3. **LoRA Efficiency**: Memory-efficient fine-tuning for consumer hardware
4. **Context-Aware Processing**: Uses 200-word context for better instruction quality

## Future Enhancements

- Additional dataset sources
- Multi-language support
- Advanced prompt engineering
- Model architecture improvements
- Real-time inference optimization
