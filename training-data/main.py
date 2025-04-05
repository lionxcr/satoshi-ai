import os
import json
import re
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from bs4 import BeautifulSoup
import markdown

# Configuration settings
CONFIG = {
    "text_extensions": [".mediawiki", ".proto", ".adoc", ".asciidoc", ".md", ".html", ".json"],
    "image_extensions": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"],
    "base_dir": os.path.dirname(__file__),  # Current directory is already the training-data directory
    "output_dir": "/output",  # Use absolute path to output directory
    "max_workers": os.cpu_count() or 4,
    "chunk_size": 512,  # Token chunk size for text processing
    "image_size": 224,  # Size for image processing (common for vision models)
    "overlap": 64,  # Overlap between chunks
    "persona_format": True,  # Enable persona generation for JSON files
}

# Ensure output directory exists
os.makedirs(CONFIG["output_dir"], exist_ok=True)


def clean_markdown_text(text, file_ext):
    """Clean and normalize text from various markup formats."""
    if file_ext in [".mediawiki"]:
        # Remove mediawiki syntax
        text = re.sub(r"'{2,}(.*?)'{2,}", r"\1", text)  # Remove bold/italic markers
        text = re.sub(r"==+\s*(.*?)\s*==+", r"\n\1\n", text)  # Convert headers to plain text with newlines
        text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)  # Remove wiki links
        text = re.sub(r"\[https?://[^\s\]]+\s+(.*?)\]", r"\1", text)  # Remove external links, keep text
        
    elif file_ext in [".md"]:
        # Convert markdown to plain text
        text = markdown.markdown(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        
    elif file_ext in [".adoc", ".asciidoc"]:
        # Remove asciidoc syntax
        text = re.sub(r"=+\s*(.*?)\s*", r"\n\1\n", text)  # Convert headers to plain text
        text = re.sub(r"::(.*?)::", r"\1", text)  # Remove term definitions
        text = re.sub(r"link:(.*?)\[(.*?)\]", r"\2", text)  # Extract link text
        
    elif file_ext in [".html"]:
        # Convert HTML to plain text
        text = BeautifulSoup(text, "html.parser").get_text()
    
    # General cleanup for all formats
    text = re.sub(r"\n{3,}", "\n\n", text)  # Normalize multiple newlines
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = text.strip()
    
    return text


def format_json_for_persona(json_data):
    """Format JSON content specifically for persona generation."""
    try:
        # Parse the JSON data
        data = json.loads(json_data)
        
        # Check if it's a list of messages or conversation
        if isinstance(data, list):
            # Format as conversation or message collection
            formatted_text = ""
            for item in data:
                if isinstance(item, dict):
                    # Extract the most important personality elements
                    if "role" in item and "content" in item:
                        # Format like ChatML
                        formatted_text += f"{item['role']}:\n{item['content']}\n\n"
                    elif "author" in item and "text" in item:
                        formatted_text += f"{item['author']}:\n{item['text']}\n\n"
                    elif "name" in item and "message" in item:
                        formatted_text += f"{item['name']}:\n{item['message']}\n\n"
                    else:
                        # Just add all items as key-value pairs
                        for key, value in item.items():
                            if isinstance(value, str):
                                formatted_text += f"{key}: {value}\n"
                        formatted_text += "\n"
            
            return formatted_text
                    
        # Check if it's a persona or character object
        elif isinstance(data, dict):
            formatted_text = "PERSONA DESCRIPTION:\n"
            
            # Handle common persona fields
            priority_fields = ["name", "personality", "description", "background", "traits", 
                             "speaking_style", "voice", "mannerisms", "character", "bio"]
            
            # First process priority fields
            for field in priority_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, str):
                        formatted_text += f"{field}: {value}\n\n"
                    elif isinstance(value, list):
                        formatted_text += f"{field}: {', '.join(value)}\n\n"
                    elif isinstance(value, dict):
                        formatted_text += f"{field}:\n"
                        for k, v in value.items():
                            if isinstance(v, str):
                                formatted_text += f"  {k}: {v}\n"
                        formatted_text += "\n"
            
            # Then process any remaining fields
            for key, value in data.items():
                if key not in priority_fields:
                    if isinstance(value, str):
                        formatted_text += f"{key}: {value}\n"
                    elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                        formatted_text += f"{key}: {', '.join(value)}\n"
            
            # If there's a sample dialog or examples section, add it at the end
            if "examples" in data and isinstance(data["examples"], list):
                formatted_text += "\nEXAMPLE DIALOGUES:\n"
                for example in data["examples"]:
                    if isinstance(example, dict) and "input" in example and "output" in example:
                        formatted_text += f"User: {example['input']}\n"
                        formatted_text += f"Character: {example['output']}\n\n"
            
            return formatted_text
        
        # Fallback: just stringify the JSON with indentation
        return json.dumps(data, indent=2)
    
    except json.JSONDecodeError:
        # If it's not valid JSON, return the original text
        return json_data


def process_json_file(file_path):
    """Process a JSON file optimized for persona generation."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None
    
    # Format the JSON content for persona generation
    formatted_text = format_json_for_persona(content)
    
    # Split into overlapping chunks
    chunks = chunk_text(formatted_text)
    
    result = {
        "file_path": file_path,
        "file_type": ".json",
        "chunks": chunks,
        "is_persona": True,
        "metadata": {
            "original_size": len(content),
            "processed_size": len(formatted_text),
            "num_chunks": len(chunks)
        }
    }
    
    return result


def chunk_text(text, chunk_size=CONFIG["chunk_size"], overlap=CONFIG["overlap"]):
    """Split text into overlapping chunks of approximately equal size."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Try to find a good breaking point (newline or space)
        if end < len(text) - overlap:
            # Look for newline in the overlap region
            newline_pos = text.rfind("\n", end - overlap, end)
            if newline_pos != -1:
                end = newline_pos + 1
            else:
                # Look for space in the overlap region
                space_pos = text.rfind(" ", end - overlap, end)
                if space_pos != -1:
                    end = space_pos + 1
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def process_text_file(file_path):
    """Process a text file and return chunks in a format suitable for model training."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Special handling for JSON files if they should be treated as personas
    if file_ext == ".json" and CONFIG["persona_format"]:
        return process_json_file(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    # Clean and normalize text based on file type
    clean_text = clean_markdown_text(content, file_ext)
    
    # Split into overlapping chunks
    chunks = chunk_text(clean_text)
    
    result = {
        "file_path": file_path,
        "file_type": file_ext,
        "chunks": chunks,
        "metadata": {
            "original_size": len(content),
            "processed_size": len(clean_text),
            "num_chunks": len(chunks)
        }
    }
    
    return result


def setup_image_processor():
    """Set up the CLIP model for processing images."""
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        return processor, model
    except Exception as e:
        print(f"Error setting up image processor: {e}")
        print("Falling back to basic image processing")
        return None, None


def process_image_file(file_path, processor=None, model=None):
    """Process an image file and generate tensors for model training."""
    try:
        # Load and resize image
        image = Image.open(file_path).convert('RGB')
        
        # Basic processing (resize and normalize)
        image_resized = image.resize((CONFIG["image_size"], CONFIG["image_size"]))
        image_array = np.array(image_resized) / 255.0  # Normalize to [0,1]
        
        # If CLIP processor is available, use it for better feature extraction
        if processor and model:
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                
            # Convert to numpy for storage
            image_embedding = image_features.cpu().numpy()
            
            result = {
                "file_path": file_path,
                "image_size": [image.width, image.height],
                "tensor_shape": image_embedding.shape,
                "embedding": image_embedding.tolist(),  # Store as list for JSON
                "processing": "clip_embedding"
            }
        else:
            # Basic tensor representation if CLIP is not available
            tensor = torch.tensor(image_array.transpose(2, 0, 1), dtype=torch.float32)  # Convert to CHW format
            
            result = {
                "file_path": file_path,
                "image_size": [image.width, image.height],
                "tensor_shape": tensor.shape,
                "normalized_array": image_array.tolist(),  # Store as list for JSON
                "processing": "basic_normalization"
            }
            
        return result
    
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return {
            "file_path": file_path,
            "error": str(e),
            "processing": "failed"
        }


def process_file(file_path, clip_processor=None, clip_model=None):
    """Process a single file based on its extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext in CONFIG["text_extensions"]:
        return process_text_file(file_path)
    elif ext in CONFIG["image_extensions"]:
        return process_image_file(file_path, clip_processor, clip_model)
    else:
        # Skip files with unsupported extensions
        return None


def crawl_and_process():
    """Crawl through directories and process files."""
    # Get all subdirectories in the base directory
    directories = [d for d in os.listdir(CONFIG["base_dir"]) 
                  if os.path.isdir(os.path.join(CONFIG["base_dir"], d))]
    
    # Set up image processor if possible
    clip_processor, clip_model = setup_image_processor()
    
    # Process each directory
    for directory in directories:
        dir_path = os.path.join(CONFIG["base_dir"], directory)
        output_dir = os.path.join(CONFIG["output_dir"], directory)
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all files to process
        files_to_process = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                ext = ext.lower()
                
                if ext in CONFIG["text_extensions"] or ext in CONFIG["image_extensions"]:
                    files_to_process.append(file_path)
        
        print(f"Found {len(files_to_process)} files to process in {directory}")
        
        # Process files in parallel
        results = []
        with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            # Create a partial function with the image processor
            future_to_file = {
                executor.submit(process_file, file_path, clip_processor, clip_model): file_path 
                for file_path in files_to_process
            }
            
            for future in tqdm(future_to_file, desc=f"Processing {directory}"):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    file_path = future_to_file[future]
                    print(f"Error processing {file_path}: {e}")
        
        # Group results by type
        text_results = [r for r in results if r.get("chunks", None)]
        persona_results = [r for r in text_results if r.get("is_persona", False)]
        regular_text_results = [r for r in text_results if not r.get("is_persona", False)]
        image_results = [r for r in results if r.get("tensor_shape", None) or r.get("normalized_array", None)]
        
        # Save regular text results
        if regular_text_results:
            text_output_path = os.path.join(output_dir, "text_data.json")
            with open(text_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "format_version": "1.0",
                    "model_target": "llama-3.2-1B",
                    "training_type": "PEFT_LoRA",
                    "data": regular_text_results
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Saved {len(regular_text_results)} text files to {text_output_path}")
        
        # Save persona results separately
        if persona_results:
            persona_output_path = os.path.join(output_dir, "persona_data.json")
            with open(persona_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "format_version": "1.0",
                    "model_target": "llama-3.2-1B",
                    "training_type": "PEFT_LoRA_persona",
                    "data": persona_results
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Saved {len(persona_results)} persona files to {persona_output_path}")
            
        # Save image results
        if image_results:
            image_output_path = os.path.join(output_dir, "image_data.json")
            with open(image_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "format_version": "1.0",
                    "model_target": "llama-3.2-1B",
                    "training_type": "PEFT_LoRA_vision",
                    "data": image_results
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Saved {len(image_results)} image files to {image_output_path}")


def prepare_hf_dataset(output_dir, format_type="hf_transformers"):
    """Convert processed data into formats specifically for Hugging Face training."""
    
    # For each directory in the output
    for directory in os.listdir(output_dir):
        dir_path = os.path.join(output_dir, directory)
        
        if not os.path.isdir(dir_path):
            continue
            
        # Check for text data
        text_path = os.path.join(dir_path, "text_data.json")
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
                
            # Create HF-ready dataset format
            hf_dataset = []
            for item in text_data["data"]:
                file_path = item["file_path"]
                for i, chunk in enumerate(item["chunks"]):
                    hf_dataset.append({
                        "text": chunk,
                        "source": os.path.basename(file_path),
                        "chunk_id": i,
                        "metadata": {
                            "file_type": item["file_type"],
                            "original_path": file_path
                        }
                    })
                    
            # Save in HF-ready format
            hf_output_path = os.path.join(dir_path, "hf_text_dataset.json")
            with open(hf_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": "text",
                    "format": "hf_transformers",
                    "data": hf_dataset
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Created Hugging Face text dataset at {hf_output_path}")
        
        # Check for persona data
        persona_path = os.path.join(dir_path, "persona_data.json")
        if os.path.exists(persona_path):
            with open(persona_path, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
                
            # Create HF-ready dataset format for personas
            hf_persona_dataset = []
            for item in persona_data["data"]:
                file_path = item["file_path"]
                for i, chunk in enumerate(item["chunks"]):
                    hf_persona_dataset.append({
                        "text": chunk,
                        "source": os.path.basename(file_path),
                        "chunk_id": i,
                        "is_persona": True,
                        "metadata": {
                            "file_type": item["file_type"],
                            "original_path": file_path
                        }
                    })
                    
            # Save in HF-ready format
            hf_persona_output_path = os.path.join(dir_path, "hf_persona_dataset.json")
            with open(hf_persona_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": "persona",
                    "format": "hf_transformers",
                    "data": hf_persona_dataset
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Created Hugging Face persona dataset at {hf_persona_output_path}")
                
        # Check for image data
        image_path = os.path.join(dir_path, "image_data.json")
        if os.path.exists(image_path):
            with open(image_path, 'r', encoding='utf-8') as f:
                image_data = json.load(f)
                
            # Create HF-ready dataset format for vision
            hf_image_dataset = []
            for item in image_data["data"]:
                if "error" not in item:
                    hf_image_dataset.append({
                        "file_path": item["file_path"],
                        "image_size": item["image_size"],
                        "tensor_shape": item["tensor_shape"],
                        "processing": item["processing"],
                        # Include either embedding or normalized array based on what's available
                        "features": item.get("embedding", item.get("normalized_array", []))
                    })
                    
            # Save in HF-ready format
            hf_image_output_path = os.path.join(dir_path, "hf_image_dataset.json")
            with open(hf_image_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": "image",
                    "format": "hf_transformers",
                    "data": hf_image_dataset
                }, f, ensure_ascii=False, indent=2)
                
            print(f"Created Hugging Face image dataset at {hf_image_output_path}")


def create_peft_config(output_dir):
    """Create default PEFT configuration for LoRA training."""
    peft_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "model_id": "meta-llama/Llama-3.2-1B",
        "lora_config": {
            "r": 16,  # Rank
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "training_args": {
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "fp16": True,
            "max_grad_norm": 0.3,
        }
    }
    
    # Create persona-specific PEFT configuration
    persona_peft_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "model_id": "meta-llama/Llama-3.2-1B",
        "lora_config": {
            "r": 16,  # Rank
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "training_args": {
            "learning_rate": 3e-4,  # Higher learning rate for persona adaptation
            "num_train_epochs": 5,   # More epochs for better style adaptation
            "per_device_train_batch_size": 4,  # Smaller batch for more updates
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": True,
            "warmup_ratio": 0.05,
            "weight_decay": 0.005,
            "fp16": True,
            "max_grad_norm": 0.3,
        },
        "persona_config": {
            "enable_style_adaptation": True,
            "use_prefix_tuning": True,
            "prefix_length": 8,
            "preserve_writing_style": True
        }
    }
    
    # Save the PEFT configuration
    config_path = os.path.join(output_dir, "peft_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(peft_config, f, indent=2)
    
    # Save the persona PEFT configuration
    persona_config_path = os.path.join(output_dir, "persona_peft_config.json")
    with open(persona_config_path, 'w', encoding='utf-8') as f:
        json.dump(persona_peft_config, f, indent=2)
    
    print(f"Created PEFT configurations at {output_dir}")


def main():
    """Main function to execute the data processing pipeline."""
    print(f"Satoshi AI Training Data Processor")
    print("=" * 40)
    print(f"Base directory: {CONFIG['base_dir']}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print("=" * 40)
    
    # Check if required directories exist
    required_dirs = ["bips", "bitcoinbook", "lnbook"]
    missing_dirs = [d for d in required_dirs if not os.path.isdir(os.path.join(CONFIG['base_dir'], d))]
    
    if missing_dirs:
        print("WARNING: The following required repositories are missing:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("\nPlease clone the repositories to continue:")
        print("  cd training-data")
        for d in missing_dirs:
            print(f"  git clone https://github.com/bitcoin/{d}.git {d}")
        print("\nOr run with existing directories only? (y/n)")
        response = input("> ").strip().lower()
        if response != 'y':
            print("Exiting. Please clone the repositories and try again.")
            return
    
    # Ensure the output directory exists
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Crawl and process files
    crawl_and_process()
    
    # Convert to HF-ready format
    prepare_hf_dataset(CONFIG["output_dir"])
    
    # Create PEFT config
    create_peft_config(CONFIG["output_dir"])
    
    print("\nProcessing complete!")
    print(f"Processed datasets are available in: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
