#!/usr/bin/env python3
"""
Complete Google Colab Training Script for Code2Prompt
Adapted from Liquid4All/leap-finetune for Code2Prompt dataset

This script contains the exact code used in the Google Colab notebook
for fine-tuning Qwen2.5-1.5B with LoRA on the Code2Prompt dataset.
"""

import torch
import json
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def install_dependencies():
    """Install required dependencies for Colab."""
    import subprocess
    import sys
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "transformers accelerate peft datasets bitsandbytes",
        "wandb tensorboard",
        "scipy scikit-learn matplotlib seaborn"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + package.split())

def setup_device():
    """Setup device and print GPU information."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device

def load_code2prompt_dataset(dataset_path=None):
    """Load the Code2Prompt dataset."""
    if dataset_path:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    else:
        # Sample dataset for demonstration
        dataset = [
            {
                "instruction": "Create a blue circle that rotates clockwise",
                "code": """from manim import *

class RotatingCircle(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(Create(circle))
        self.play(Rotate(circle, angle=2*PI), run_time=2)
        self.wait()"""
            },
            {
                "instruction": "Create a square that transforms into a triangle",
                "code": """from manim import *

class SquareToTriangle(Scene):
    def construct(self):
        square = Square(color=RED)
        triangle = Triangle(color=BLUE)
        self.play(Create(square))
        self.play(Transform(square, triangle), run_time=2)
        self.wait()"""
            },
            {
                "instruction": "Create a moving dot that follows a sine wave",
                "code": """from manim import *
import numpy as np

class SineWaveDot(Scene):
    def construct(self):
        dot = Dot(color=YELLOW)
        
        def sine_func(x):
            return np.sin(x) * 2
        
        sine_path = ParametricFunction(
            lambda t: np.array([t, sine_func(t), 0]),
            t_range=[-3*PI, 3*PI],
            color=WHITE
        )
        
        self.play(Create(sine_path))
        self.play(MoveAlongPath(dot, sine_path), run_time=3)
        self.wait()"""
            }
        ]
    
    print(f"Loaded {len(dataset)} examples")
    return dataset

def format_instruction_code(example):
    """Format instruction and code into training prompt."""
    instruction = example["instruction"]
    code = example["code"]
    
    prompt = f"Instruction: {instruction}\nCode:\n{code}"
    return {"text": prompt}

def prepare_dataset(dataset, tokenizer, max_length=1024):
    """Prepare dataset for training."""
    # Format dataset
    formatted_data = [format_instruction_code(ex) for ex in dataset]
    
    # Split into train/eval
    split_idx = int(0.9 * len(formatted_data))
    train_data = formatted_data[:split_idx]
    eval_data = formatted_data[split_idx:]
    
    print(f"Training examples: {len(train_data)}")
    print(f"Evaluation examples: {len(eval_data)}")
    
    # Convert to Hugging Face dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    # Tokenize dataset
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    print("Dataset tokenized successfully")
    return train_dataset, eval_dataset

def load_model_and_tokenizer(model_name, lora_config):
    """Load model and apply LoRA configuration."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Model loaded and LoRA configured")
    return model, tokenizer

def create_lora_config(lora_rank=16, lora_alpha=32, lora_dropout=0.1):
    """Create LoRA configuration."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )

def setup_training_args(config):
    """Setup training arguments."""
    return TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        report_to=None,  # Disable wandb for Colab
        remove_unused_columns=False
    )

def train_model(model, tokenizer, train_dataset, eval_dataset, training_args):
    """Train the model."""
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("Trainer initialized")
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("Training completed and model saved")
    return trainer

def evaluate_model(trainer):
    """Evaluate the trained model."""
    eval_results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    return eval_results

def generate_code(model, tokenizer, instruction, device, max_length=500, temperature=0.7):
    """Generate code from instruction."""
    prompt = f"Instruction: {instruction}\nCode:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    code = generated_text.split("Code:")[-1].strip()
    return code

def test_inference(model, tokenizer, device):
    """Test model inference with sample instructions."""
    test_instructions = [
        "Create a red square that moves to the right",
        "Draw a blue circle and make it rotate",
        "Create a green triangle that changes to a square"
    ]
    
    print("Testing model inference:")
    for instruction in test_instructions:
        print(f"\nInstruction: {instruction}")
        print("Generated code:")
        generated_code = generate_code(model, tokenizer, instruction, device)
        print(generated_code)
        print("-" * 50)

def save_models(model, tokenizer, output_dir="./outputs"):
    """Save models for deployment."""
    model_path = f"{output_dir}/final_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Also save as merged model (optional)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{output_dir}/merged_model")
    tokenizer.save_pretrained(f"{output_dir}/merged_model")
    
    print("Models saved for deployment:")
    print(f"  LoRA model: {model_path}")
    print(f"  Merged model: {output_dir}/merged_model")

def main():
    """Main training function."""
    # Configuration
    CONFIG = {
        "model_name": "LFM2.5-1.5B",
        "max_length": 1024,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "save_steps": 500,
        "eval_steps": 500,
        "logging_steps": 10,
        "output_dir": "./outputs",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1
    }
    
    print("Code2Prompt Training Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Setup
    device = setup_device()
    
    # Load dataset
    dataset = load_code2prompt_dataset()  # Use your dataset path here
    
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(dataset, tokenizer, CONFIG["max_length"])
    
    # Create LoRA config
    lora_config = create_lora_config(
        CONFIG["lora_rank"], 
        CONFIG["lora_alpha"], 
        CONFIG["lora_dropout"]
    )
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(CONFIG["model_name"], lora_config)
    
    # Setup training
    training_args = setup_training_args(CONFIG)
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, training_args)
    
    # Evaluate
    eval_results = evaluate_model(trainer)
    
    # Test inference
    test_inference(model, tokenizer, device)
    
    # Save models
    save_models(model, tokenizer, CONFIG["output_dir"])
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
