#!/usr/bin/env python3
# Satoshi AI - Bitcoin Expert Fine-Tuning
# This script fine-tunes the Llama-3.2-1B model to become a Bitcoin expert with a Satoshi Nakamoto persona
# Uses QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning

import os
import json
import logging
import glob
import random
from typing import List, Dict, Any, Optional, Union

import torch
import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure CUDA memory allocation to avoid fragmentation
# This setting helps with the "CUDA out of memory" error
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./satoshi-ai-model"
TRAINING_DATA_DIR = "./training-data/output"
PEFT_CONFIG_PATH = os.path.join(TRAINING_DATA_DIR, "persona_peft_config.json")
SEED = 42
MAX_SEQ_LENGTH = 1024  # Reduced from 2048 to save memory

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
else:
    logger.warning("CUDA is not available. Training will be very slow on CPU!")

def load_peft_config() -> Dict[str, Any]:
    """Load PEFT configuration from the config file."""
    try:
        with open(PEFT_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading PEFT config: {e}")
        raise

def load_datasets() -> Dataset:
    """Load and process datasets from the training data directories."""
    datasets = []
    
    # Define directories to load data from
    data_dirs = [
        os.path.join(TRAINING_DATA_DIR, "bitcoinbook"),
        os.path.join(TRAINING_DATA_DIR, "bips"),
        os.path.join(TRAINING_DATA_DIR, "lnbook"),
        os.path.join(TRAINING_DATA_DIR, "emails"),
        os.path.join(TRAINING_DATA_DIR, "posts"),
    ]
    
    for data_dir in data_dirs:
        try:
            # Load the text dataset
            text_dataset_path = os.path.join(data_dir, "hf_text_dataset.json")
            if os.path.exists(text_dataset_path):
                with open(text_dataset_path, "r") as f:
                    dataset_data = json.load(f)
                
                # Create a HuggingFace dataset from the loaded data
                if "data" in dataset_data and len(dataset_data["data"]) > 0:
                    dataset = Dataset.from_dict({
                        "text": [item["text"] for item in dataset_data["data"]],
                        "source": [item["source"] for item in dataset_data["data"]],
                    })
                    datasets.append(dataset)
                    logger.info(f"Loaded {len(dataset_data['data'])} examples from {data_dir}")
        except Exception as e:
            logger.warning(f"Error loading dataset from {data_dir}: {e}")
    
    if not datasets:
        raise ValueError("No datasets were loaded. Check the data paths and formats.")
    
    # Combine all datasets
    combined_dataset = concatenate_datasets(datasets)
    logger.info(f"Combined dataset contains {len(combined_dataset)} examples")
    
    return combined_dataset

def prepare_training_data(dataset: Dataset, tokenizer) -> Dataset:
    """Prepare and tokenize the dataset for training."""
    # Add a special instruction and Satoshi persona to each example
    def add_satoshi_persona(examples):
        satoshi_prefix = "You are Satoshi Nakamoto, the creator of Bitcoin. Please respond with deep technical knowledge and the writing style of Satoshi Nakamoto: "
        
        # Format examples with the Satoshi persona instruction
        texts = []
        for text in examples["text"]:
            # Format as an instruction-response pair
            if random.random() < 0.3:  # 30% of examples become Q&A format
                # Extract a question from the text (simplistic approach)
                segments = text.split("?")
                if len(segments) > 1:
                    question = segments[0] + "?"
                    answer = text
                    formatted_text = f"### Instruction:\n{question}\n\n### Response:\n{answer}"
                else:
                    formatted_text = f"### Instruction:\nExplain the following Bitcoin concept: {text[:50]}...\n\n### Response:\n{text}"
            else:
                # Direct Satoshi-style response
                formatted_text = f"### Instruction:\n{satoshi_prefix}\n\n### Response:\n{text}"
            
            texts.append(formatted_text)
        
        return {"text": texts}
    
    # Apply the persona transformation
    persona_dataset = dataset.map(add_satoshi_persona, batched=True)
    
    # Tokenize the dataset with reduced sequence length
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,  # Using reduced sequence length to save memory
            padding="max_length",
        )
    
    tokenized_dataset = persona_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["source"],
        num_proc=1,  # Using a single process to avoid OOM errors
    )
    
    logger.info(f"Prepared and tokenized {len(tokenized_dataset)} examples for training")
    return tokenized_dataset

def main():
    """Main function to execute the fine-tuning process."""
    logger.info("Starting Satoshi AI fine-tuning process")
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training will be very slow on CPU!")
    
    # Load PEFT config
    logger.info("Loading PEFT configuration")
    peft_config = load_peft_config()
    
    # Configure quantization
    logger.info("Setting up quantization config")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    logger.info(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 to reduce memory usage
        offload_folder="offload_folder",  # Enable CPU offloading for memory management
    )
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training
    logger.info("Preparing model for k-bit training")
    model = prepare_model_for_kbit_training(model)
    
    # Get LoRA configuration from loaded PEFT config
    logger.info("Setting up LoRA configuration")
    lora_config = LoraConfig(
        r=peft_config["lora_config"]["r"],
        lora_alpha=peft_config["lora_config"]["lora_alpha"],
        lora_dropout=peft_config["lora_config"]["lora_dropout"],
        bias=peft_config["lora_config"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=peft_config["lora_config"]["target_modules"],
    )
    
    # Apply LoRA adapters to the model
    logger.info("Applying LoRA adapters to the model")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Log trainable parameters
    
    # Free up memory
    torch.cuda.empty_cache()
    
    # Load and prepare datasets
    logger.info("Loading datasets")
    raw_dataset = load_datasets()
    
    logger.info("Preparing training data")
    tokenized_dataset = prepare_training_data(raw_dataset, tokenizer)
    
    # Split dataset into train and eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=SEED)
    
    # Set up training arguments from PEFT config
    training_config = peft_config["training_args"]
    
    # Override batch size with smaller values to prevent OOM errors
    batch_size = min(2, training_config["per_device_train_batch_size"])
    grad_accumulation = max(8, training_config["gradient_accumulation_steps"])
    
    logger.info(f"Using batch size of {batch_size} with gradient accumulation steps of {grad_accumulation}")
    
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=batch_size,  # Reduced batch size
        per_device_eval_batch_size=batch_size,   # Reduced eval batch size
        gradient_accumulation_steps=grad_accumulation,  # Increased gradient accumulation
        gradient_checkpointing=True,  # Always use gradient checkpointing
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        fp16=training_config["fp16"],
        max_grad_norm=training_config["max_grad_norm"],
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",  # Updated from evaluation_strategy
        load_best_model_at_end=True,
        save_total_limit=2,  # Reduced to save disk space
        report_to="tensorboard",
        # Additional memory-saving settings
        optim="adamw_torch_fused",  # Use fused optimizer
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,  # Reduce parallel workers
        remove_unused_columns=True,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )
    
    # Create Trainer
    logger.info("Creating trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )
    
    # Free up memory before training
    torch.cuda.empty_cache()
    
    # Train model
    logger.info("Starting training")
    try:
        # Start training from scratch instead of resuming
        logger.info("Starting fresh training (not resuming from checkpoint)")
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        # Try to save partial progress if possible
        try:
            logger.info("Attempting to save partial progress...")
            trainer.save_model(os.path.join(OUTPUT_DIR, "partial_checkpoint"))
        except:
            logger.error("Could not save partial progress")
        raise
    
    # Save model
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    # Save adapter
    logger.info("Saving PEFT adapter")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 