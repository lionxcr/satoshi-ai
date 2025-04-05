# Satoshi AI - Fine-Tuning

This project fine-tunes a Llama-3.2-1B model to become a Bitcoin expert with a Satoshi Nakamoto persona, using QLoRA (Quantized Low-Rank Adaptation).

## Project Structure

- `training_model.py`: The main script that implements QLoRA fine-tuning
- `requirements.txt`: Required Python packages
- `training-data/`: Directory containing the training data
- `Llama-3.2-1B/`: Directory containing the base model

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Ensure you have access to the Llama-3.2-1B model, which should be in the `Llama-3.2-1B` directory.

3. Verify the training data is properly set up in the `training-data/output` directory.

## Running the Training

To start the fine-tuning process:

```bash
python training_model.py
```

The script will:

1. Load the Llama-3.2-1B model in 4-bit precision using QLoRA
2. Configure the LoRA adapters according to the settings in `training-data/output/persona_peft_config.json`
3. Process training data from the output directories (bitcoinbook, bips, lnbook, emails, posts)
4. Fine-tune the model and save the results to the `satoshi-ai-model` directory

## Training on Google Colab

For those without access to a high-end GPU, you can use Google Colab to train the model. Colab provides free access to GPUs (typically NVIDIA T4, P100, or V100) which are suitable for this fine-tuning task.

### Setting Up Google Colab

1. Create a new Colab notebook by visiting [Google Colab](https://colab.research.google.com/)

2. Mount your Google Drive to store the training data and model outputs:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Create a project directory in your Google Drive:
   ```python
   !mkdir -p /content/drive/MyDrive/satoshi-ai
   %cd /content/drive/MyDrive/satoshi-ai
   ```

4. Clone the project repository (if using Git) or upload the required files:
   ```python
   # Option 1: Clone from a Git repository
   !git clone https://your-repository-url.git .
   
   # Option 2: Upload files manually
   # Use the Colab file browser to upload training_model.py, requirements.txt, and other necessary files
   ```

5. Install the required dependencies:
   ```python
   !pip install -r requirements.txt
   ```

6. Set up the data directory structure and upload or prepare your training data:
   ```python
   !mkdir -p training-data/output/bitcoinbook
   !mkdir -p training-data/output/bips
   !mkdir -p training-data/output/lnbook
   !mkdir -p training-data/output/emails
   !mkdir -p training-data/output/posts
   
   # Upload your training data to these directories
   # You can use Google Drive UI or commands like:
   # !cp /path/to/your/data/* training-data/output/bitcoinbook/
   ```

7. Upload or create the PEFT configuration file:
   ```python
   # Example: Creating a basic PEFT config if you don't have one
   !cat > training-data/output/persona_peft_config.json << 'EOL'
   {
     "lora_config": {
       "r": 16,
       "lora_alpha": 32,
       "lora_dropout": 0.05,
       "bias": "none",
       "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
     },
     "training_args": {
       "learning_rate": 2e-4,
       "num_train_epochs": 3,
       "per_device_train_batch_size": 2,
       "gradient_accumulation_steps": 8,
       "warmup_ratio": 0.03,
       "weight_decay": 0.01,
       "fp16": true,
       "max_grad_norm": 0.3
     }
   }
   EOL
   ```

### Running the Training on Colab

1. Run the training script:
   ```python
   !python training_model.py
   ```

2. Monitor the training process:
   - Colab will display training logs in the output cell
   - You can use TensorBoard for more detailed monitoring:
     ```python
     %load_ext tensorboard
     %tensorboard --logdir satoshi-ai-model
     ```

3. After training, compress and download the model:
   ```python
   !zip -r satoshi-ai-model.zip satoshi-ai-model/
   ```
   Then use the Colab file browser to download the ZIP file to your local machine.

### Colab-Specific Considerations

1. **Runtime Limits**: Colab has runtime limits (usually 12 hours for free tier). For longer training:
   - Save checkpoints frequently
   - Use a smaller dataset or fewer epochs
   - Resume training from checkpoints if disconnected

2. **GPU Allocation**: Colab doesn't guarantee which GPU you'll get. To check your assigned GPU:
   ```python
   !nvidia-smi
   ```

3. **Memory Management**: 
   - Colab GPUs typically have 12-16GB VRAM
   - Use the memory optimization techniques mentioned below
   - Close other browser tabs using Colab to free up resources

4. **Persistent Storage**: 
   - Always mount Google Drive to save your outputs
   - Download important files after training
   - Remember that Colab runtime storage is temporary and will be lost when the session ends

### Memory Optimization

The script includes several memory optimization techniques:

- 4-bit quantization with NF4 data type
- Reduced sequence length (1024 tokens instead of 2048)
- Small batch size with increased gradient accumulation
- Gradient checkpointing
- Fused AdamW optimizer
- CPU offloading for parts of the model
- PyTorch memory allocation optimizations

If you encounter CUDA out-of-memory errors, you can try:

1. Further reduce the batch size by editing the `batch_size` variable in the script
2. Reduce the sequence length by changing `MAX_SEQ_LENGTH`
3. Run on a GPU with more VRAM
4. Enable CPU offloading by modifying the model loading parameters

## Technical Details

### QLoRA Implementation

The script uses QLoRA, which quantizes the base model to 4-bit precision and then applies LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. This allows the model to be fine-tuned with minimal memory requirements while maintaining performance.

### Data Processing

The script:
- Loads text data from multiple sources
- Formats it with a Satoshi Nakamoto persona prefix
- Converts some examples to Q&A format
- Tokenizes the data with appropriate padding and truncation

### Hyperparameters

Hyperparameters are loaded from the `persona_peft_config.json` file, which includes:
- LoRA configuration (rank, alpha, dropout, target modules)
- Training arguments (learning rate, batch size, epochs, etc.)

## Hardware Requirements

- CUDA-capable GPU with at least 8GB of VRAM (16GB+ recommended)
- At least 16GB of system RAM
- Several GB of disk space for the model and training data

The script is optimized to work on consumer GPUs, but larger batch sizes or sequence lengths will require more VRAM.

## Output

After training, you will find:
- The full fine-tuned model in the `satoshi-ai-model` directory
- The LoRA adapter weights in the `satoshi-ai-model/adapter` directory 
- Training logs and evaluation metrics in the `satoshi-ai-model` directory

## Using the Fine-Tuned Model

After training, you can load the model with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Load the adapter config
config = PeftConfig.from_pretrained("./satoshi-ai-model/adapter")

# Load the base model with 4-bit quantization (for inference)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Load the fine-tuned model with the adapter
model = PeftModel.from_pretrained(base_model, "./satoshi-ai-model/adapter")

# Generate text with the fine-tuned model
inputs = tokenizer("What is the purpose of Bitcoin mining?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

### CUDA Out of Memory Errors

If you encounter CUDA out-of-memory errors during training:

1. Make sure no other processes are using the GPU
2. Try setting the following environment variable before running:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
3. Reduce batch size by editing the script
4. If persistently out of memory, reduce the model's quantization settings or try a smaller model 