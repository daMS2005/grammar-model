#!/usr/bin/env python3
"""
T5 Fine-tuning script optimized for Azure Machine Learning
"""
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
import argparse
import os
from tqdm import tqdm
import mlflow
import mlflow.pytorch

class PunctuationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_input_length: int = 512, max_target_length: int = 512):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        
        print(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Add task prefix to input
        input_text = f"restore punctuation: {item['input']}"
        target_text = item['output']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For T5, we need to set labels (target tokens) and replace padding token id with -100
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

def load_model_and_tokenizer(model_name: str):
    """Load T5 model and tokenizer"""
    print(f"Loading model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5 for punctuation restoration on Azure ML")
    parser.add_argument("--model-name", default="google-t5/t5-base", help="T5 model to use")
    parser.add_argument("--train-data", default="./data/train_20k_v2.jsonl", help="Training data path")
    parser.add_argument("--eval-data", default="./data/eval_20k_v2.jsonl", help="Evaluation data path")
    parser.add_argument("--output-dir", default="./outputs/t5-punct-model", help="Output directory for model")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--max-input-length", type=int, default=512, help="Max input sequence length")
    parser.add_argument("--max-target-length", type=int, default=512, help="Max target sequence length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Check for available device (CUDA preferred for Azure)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Start MLflow tracking
    mlflow.start_run()
    
    # Log parameters
    mlflow.log_params({
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length
    })
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Create datasets
    train_dataset = PunctuationDataset(
        args.train_data, tokenizer, args.max_input_length, args.max_target_length
    )
    eval_dataset = PunctuationDataset(
        args.eval_data, tokenizer, args.max_input_length, args.max_target_length
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,  # We're using MLflow instead
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),  # Enable mixed precision on GPU
        dataloader_num_workers=4,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    train_result = trainer.train()
    
    # Log training metrics
    mlflow.log_metrics({
        "train_loss": train_result.training_loss,
        "train_steps": train_result.global_step,
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0)
    })
    
    # Save the final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    
    # Log evaluation metrics
    mlflow.log_metrics({
        "eval_loss": eval_results["eval_loss"],
        "eval_runtime": eval_results["eval_runtime"],
        "eval_samples_per_second": eval_results["eval_samples_per_second"]
    })
    
    # Log the model
    mlflow.pytorch.log_model(model, "t5-punct-model")
    
    # End MLflow run
    mlflow.end_run()
    
    print("Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 