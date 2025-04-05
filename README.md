# Bitcoin & Lightning Network Training Data Processor

This tool processes various text and image files from Bitcoin and Lightning Network related repositories to create datasets optimized for fine-tuning Llama-3.2-1B models using Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA and QLoRA.

## Features

- Processes multiple file formats including `.mediawiki`, `.proto`, `.adoc`, `.asciidoc`, `.md`, `.html`, and `.json`
- Special handling for JSON files to optimize for persona generation and writing style adaptation
- Handles images and converts them to tensor representations suitable for vision models
- Creates Hugging Face-compatible datasets
- Processes files in parallel for better performance
- Generates optimal PEFT configuration for Llama-3.2-1B

## Data Sources

This project uses the following open-source repositories as training data sources:

| Repository | Description | License |
|------------|-------------|---------|
| [Bitcoin Improvement Proposals (BIPs)](https://github.com/bitcoin/bips) | Technical documentation for Bitcoin protocol enhancements | MIT License |
| [Mastering Bitcoin](https://github.com/bitcoinbook/bitcoinbook) | Comprehensive guide to Bitcoin by Andreas M. Antonopoulos | CC BY-SA 4.0 |
| [Mastering Lightning Network](https://github.com/lnbook/lnbook) | Guide to Lightning Network by Andreas M. Antonopoulos et al. | CC BY-SA 4.0 |
| [Bitcoin Core](https://github.com/bitcoin/bitcoin) | Reference implementation of Bitcoin | MIT License |
| [Lightning Network Daemon (LND)](https://github.com/lightningnetwork/lnd) | Implementation of the Lightning Network | MIT License |

## Directory Structure

```
training-data/                # This directory
├── main.py                   # Main processing script
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation file
├── peft_config.json          # Standard PEFT configuration
├── persona_peft_config.json  # Persona-optimized PEFT config
├── __init__.py               # Empty init file for Python module
├── bips/                     # Bitcoin Improvement Proposals repository (cloned)
├── bitcoinbook/              # Mastering Bitcoin book repository (cloned)
├── lnbook/                   # Lightning Network book repository (cloned)
├── emails/                   # Email correspondence/messages for training
└── posts/                    # Additional blog posts and articles
```

## Preparing Training Data

### 1. Clone Required Repositories

To gather the source data, clone the following repositories into the training-data directory:

```bash
# Clone Bitcoin Improvement Proposals (BIPs)
git clone https://github.com/bitcoin/bips.git bips

# Clone Mastering Bitcoin book
git clone https://github.com/bitcoinbook/bitcoinbook.git bitcoinbook

# Clone Lightning Network book
git clone https://github.com/lnbook/lnbook.git lnbook
```

### 2. Install Dependencies

Make sure all required Python packages are installed:

```bash
pip install -r requirements.txt
```

### 3. Process the Data

Run the main script to process all the data:

```bash
python main.py
```

The processed data will be saved to the `/output` directory.

## Output Folder Structure

After processing the data, the `/output` directory will have the following structure:

```
/output/
├── bips/                      # Processed Bitcoin Improvement Proposals
│   ├── text_data.json         # Raw processed text data
│   ├── hf_text_dataset.json   # HuggingFace-ready text dataset
│   ├── image_data.json        # Raw processed image data (if any)
│   └── hf_image_dataset.json  # HuggingFace-ready image dataset (if any)
│
├── bitcoinbook/               # Processed Mastering Bitcoin content
│   ├── text_data.json         # Raw processed text data
│   ├── hf_text_dataset.json   # HuggingFace-ready text dataset
│   ├── image_data.json        # Raw processed image data
│   └── hf_image_dataset.json  # HuggingFace-ready image dataset
│
├── lnbook/                    # Processed Lightning Network book content
│   ├── text_data.json         # Raw processed text data
│   ├── hf_text_dataset.json   # HuggingFace-ready text dataset
│   ├── image_data.json        # Raw processed image data
│   └── hf_image_dataset.json  # HuggingFace-ready image dataset
│
├── emails/                    # Processed email correspondence (if included)
│   ├── text_data.json         # Raw processed text data
│   ├── hf_text_dataset.json   # HuggingFace-ready text dataset
│   ├── persona_data.json      # Raw persona data (if any JSON files)
│   └── hf_persona_dataset.json # HuggingFace-ready persona dataset
│
├── posts/                     # Processed blog posts and articles
│   ├── text_data.json         # Raw processed text data
│   ├── hf_text_dataset.json   # HuggingFace-ready text dataset
│   ├── persona_data.json      # Raw persona data (if any JSON files)  
│   └── hf_persona_dataset.json # HuggingFace-ready persona dataset
│
├── peft_config.json           # Standard PEFT configuration for training
└── persona_peft_config.json   # Persona-optimized PEFT configuration
```

### Dataset File Formats

Each type of dataset file contains specific information:

1. **Raw processed data files** (`text_data.json`, `persona_data.json`, `image_data.json`):
   - Contains the processed chunks from source files
   - Includes metadata about original file sizes and processing
   - Used as intermediate storage before conversion to HuggingFace format

2. **HuggingFace-ready datasets** (`hf_text_dataset.json`, `hf_persona_dataset.json`, `hf_image_dataset.json`):
   - Formatted specifically for use with HuggingFace's training libraries
   - Text datasets include chunked content with source information
   - Persona datasets include specially formatted content for personality adaptation
   - Image datasets include tensor representations of images using CLIP embeddings

3. **PEFT Configuration files**:
   - `peft_config.json`: Standard configuration for LoRA fine-tuning of Llama-3.2-1B
   - `persona_peft_config.json`: Specialized configuration for personality/writing style adaptation

## Data Processing Details

### Format-specific Processing

Each file format undergoes specialized cleaning and normalization:

1. **Format-specific cleaning**:
   - **Mediawiki**: Removal of wiki syntax while preserving content structure
   - **Markdown**: Conversion to plain text while maintaining semantic structure
   - **AsciiDoc**: Preservation of content hierarchy while removing formatting codes
   - **HTML**: Extraction of clean text from HTML documents
   - **JSON**: Special handling for persona data and conversation examples

2. **Chunking with overlaps**: Text is divided into chunks of approximately 512 tokens with 64-token overlaps between chunks to maintain context between segments.

3. **Image processing**: Images are processed using CLIP embeddings for semantic representation, with resizing and normalization for consistent tensor shapes.

## Output Data Formats

### Text Data

```json
{
  "type": "text",
  "format": "hf_transformers",
  "data": [
    {
      "text": "content of the chunk",
      "source": "original_filename",
      "chunk_id": 0,
      "metadata": {
        "file_type": ".mediawiki",
        "original_path": "path/to/original/file"
      }
    }
  ]
}
```

### Persona Data (JSON files)

```json
{
  "type": "persona",
  "format": "hf_transformers",
  "data": [
    {
      "text": "PERSONA DESCRIPTION:\nname: Satoshi Nakamoto\npersonality: Private, intelligent, focused...",
      "source": "satoshi_persona.json",
      "chunk_id": 0,
      "is_persona": true,
      "metadata": {
        "file_type": ".json",
        "original_path": "path/to/original/file"
      }
    }
  ]
}
```

### Image Data

```json
{
  "type": "image",
  "format": "hf_transformers",
  "data": [
    {
      "file_path": "path/to/image.png",
      "image_size": [width, height],
      "tensor_shape": [dimensions],
      "processing": "clip_embedding",
      "features": [array_of_features]
    }
  ]
}
```

### PEFT Configurations

Two PEFT configuration files are generated:

1. **Regular PEFT Configuration** (`peft_config.json`): For general knowledge acquisition
2. **Persona-optimized Configuration** (`persona_peft_config.json`): For personality and writing style adaptation 