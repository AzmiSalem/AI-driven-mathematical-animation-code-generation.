#  Code2Prompt Complete LLM Fine-Tuning Pipeline

> **Complete pipeline for fine-tuning a language model to generate Manim animation code from natural language descriptions**

## Overview

This project demonstrates a complete pipeline for fine-tuning a small language model (LFM2.5-1.5B) to generate mathematical animation code using the Manim library. The approach combines two datasets and uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

## Pipeline Structure

```
Code2Prompt/
├── 01_data_extraction/     # Extract training data from Hugging Face
├── 02_data_processing/     # Process Manim docs with HTML + LLM
├── 03_training/           # LoRA fine-tuning implementation
├── 04_datasets/           # Final processed datasets
└── 05_results/            # Training results and analysis
```

## Quick Start

### 1. Data Extraction
Extract the Edoh dataset from Hugging Face:
```bash
cd 01_data_extraction
python extract_edoh_dataset.py
```

### 2. Data Processing
Process Manim documentation snippets with HTML extraction and LLM enhancement:
```bash
cd 02_data_processing
python format_snippets_with_openai.py \
    --input ../04_datasets/manim_docs_snippets.jsonl \
    --output ../04_datasets/manim_docs_html_pairs.all.jsonl \
    --api_key_file your_openai_api_key.txt
```

### 3. Dataset Combination
Combine Edoh and Manim datasets:
```bash
python combine_full_datasets.py
```

### 4. Training
Fine-tune the model using LoRA:
```bash
cd 03_training
python colab_training_complete.py
```

## Dataset Summary

| Dataset | Source | Pairs | Description |
|---------|--------|-------|-------------|
| **Edoh** | Hugging Face | 651 | Original instruction→code pairs |
| **Manim Docs** | Documentation extraction | 989 | Code snippets from Manim docs |
| **Combined** | Merged datasets | 1,640 | Final training dataset |

## Key Features

- **Dual Dataset Approach**: Combines original Edoh dataset with custom-extracted Manim documentation
- **HTML + LLM Processing**: Uses BeautifulSoup for HTML extraction and GPT-4o-mini for instruction generation
- **LoRA Fine-tuning**: Efficient training with Low-Rank Adaptation
- **Google Colab Ready**: Complete implementation tested on Google Colab
- **Production Ready**: Trained model ready for deployment

## Training Configuration

- **Base Model**: LFM2.5-1.5B
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Batch Size**: 1 (gradient accumulation: 4)

## Results

The model was successfully trained and shows significant improvement in:
- Manim code generation quality
- Mathematical animation understanding
- Instruction following accuracy

See `05_results/TRAINING_RESULTS.md` for detailed analysis.

## File Descriptions

### 01_data_extraction/
- `extract_edoh_dataset.py` - Downloads Edoh dataset from Hugging Face

### 02_data_processing/
- `format_snippets_with_openai.py` - Converts Manim snippets to instruction→code pairs using GPT-4o-mini
- `combine_full_datasets.py` - Combines Edoh and Manim datasets

### 03_training/
- `colab_training_complete.py` - Complete LoRA fine-tuning implementation
- `Colab_Training.ipynb` - Jupyter notebook version
- `lora_requirements.txt` - Required Python packages

### 04_datasets/
- `edoh_all_original.jsonl` - Original Edoh dataset (651 pairs, no enhancement)
- `manim_docs_html_pairs.all.jsonl` - Processed Manim docs (989 pairs)
- `train_full_dataset.json` - Combined training dataset (1,640 pairs)

### 05_results/
- `TRAINING_RESULTS.md` - Detailed training analysis and results

## Usage Example

**Input**: "Create a blue circle that rotates clockwise"

**Output**:
```python
from manim import *

class RotatingCircle(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(Create(circle))
        self.play(Rotate(circle, angle=2*PI), run_time=2)
        self.wait()
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenAI API key (for data processing)
- 8GB+ RAM
- 10GB+ disk space

## Installation

```bash
# Install training dependencies
pip install -r 03_training/lora_requirements.txt

# For data processing
pip install openai beautifulsoup4 requests
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **Edoh Dataset**: Original research dataset for code generation
- **Manim Community**: Mathematical animation library and documentation
- **Hugging Face**: Model hosting and training infrastructure
- **LoRA Paper**: Efficient fine-tuning methodology
