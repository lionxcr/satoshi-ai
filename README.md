# Satoshi AI - Building a Bitcoin Expert with a hybrid LLM approach

This project creates a specialized Bitcoin AI assistant with an authentic Satoshi Nakamoto persona through a two-stage approach:

1. Fine-tuning a Llama-3.2-1B model to become a Bitcoin and blockchain expert using QLoRA (Quantized Low-Rank Adaptation)
2. Using OpenAI's GPT-4o-mini to enhance the user experience while preserving technical accuracy
3. Generating high-quality visual explanations using DALL-E 3 for educational diagrams
4. Providing proactive learning recommendations to guide users toward deeper Bitcoin understanding

## Project Overview

### Why This Hybrid Approach?

Smaller open-source models like Llama-3.2-1B have significant limitations when generating user-friendly responses:
- Limited context window
- Less nuanced understanding of complex topics
- Inconsistent response quality and formatting
- Difficulty maintaining character voice and style

Our solution leverages the best of both worlds:
1. A **domain-specific expert model** (fine-tuned Llama-3.2-1B) that deeply understands Bitcoin, blockchain concepts, and Satoshi's writings
2. A **UX enhancement layer** (GPT-4o-mini) that improves readability, coherence, and presentation while preserving technical accuracy
3. A **visual learning component** (DALL-E 3) that creates text-free educational diagrams to explain complex Bitcoin concepts
4. A **learning path generator** that recommends related topics for further exploration

### How It Works

1. The user asks a Bitcoin or blockchain-related question
2. Our fine-tuned Llama-3.2-1B model generates an initial technical response with Bitcoin expertise
3. This response is fed to GPT-4o-mini along with Satoshi's persona characteristics
4. GPT-4o-mini reformats and enhances the response while preserving technical accuracy
5. The system generates 3 personalized recommendations for further learning on related topics
6. For image requests, the system creates a text-free, high-quality diagram using DALL-E 3
7. The user receives either a high-quality answer with learning recommendations or an educational diagram

### API Example

```python
# Client-side example
import requests
import json

url = "http://localhost:8000/generate"
headers = {"Content-Type": "application/json"}

# Example 1: Text response with recommendations
data = {
    "messages": [
        {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
        {"role": "user", "content": "Why did you choose Proof of Work for Bitcoin?"}
    ],
    "output_type": "text",
    "temperature": 0.2,
    "max_tokens": 500
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()
print(result["text"])
print("\nRecommended topics to explore:")
for rec in result["recommendations"]:
    print(f"- {rec}")

# Example 2: Image generation
image_data = {
    "messages": [
        {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
        {"role": "user", "content": "Show me how Bitcoin mining works"}
    ],
    "output_type": "image",
    "temperature": 0.2
}

image_response = requests.post(url, headers=headers, data=json.dumps(image_data))
# Handle image response (streaming response with image data)
```

### Example Outputs

**User Question:** Why did you choose Proof of Work for Bitcoin?

**Initial Llama-3.2-1B Model Output:**
```
I chose proof of work for Bitcoin because it solves the double-spend problem in a distributed network without requiring trusted third parties. The system works by making miners expend computational resources to add blocks to the blockchain. This creates a cost to mining that prevents Sybil attacks where someone might try to control the network by creating many nodes. Additionally, PoW creates a clear chain of history that everyone can agree on, since the longest chain represents the majority of computational work. This prevents attackers from rewriting history unless they control majority of network hashpower. While PoW requires energy consumption, this cost is what secures the network and ensures trustlessness.
```

**Final Enhanced Output via GPT-4o-mini:**
```
I chose Proof of Work for Bitcoin because it elegantly solves several critical challenges in creating a trustless digital currency.

First, PoW establishes an objective, measurable cost to participation. Miners must demonstrate computational work, creating a resource-backed security model that can't be circumvented through virtual identities or network manipulation. This prevents Sybil attacks where an attacker might create numerous virtual nodes.

Second, PoW creates a verifiable chronological record that cannot be altered without redoing the work. The blockchain becomes an immutable history precisely because changing any block requires redoing all subsequent proof-of-work - an insurmountable task unless an attacker controls more computational power than the honest network.

Third, PoW enables decentralized consensus without trusted authorities. The longest chain rule provides a clear, objective metric for determining the valid transaction history.

The energy expenditure is not wasteful - it's the fundamental mechanism that secures the network. It transforms electrical energy into immutable trust, providing the backbone of Bitcoin's security model.

Alternatives like Proof of Stake existed when I designed Bitcoin, but they introduce complex game-theoretical vulnerabilities and centralization pressures that I deliberately chose to avoid.
```

**Recommended Learning Topics:**
```
- "Study the Byzantine Generals Problem and how PoW solves distributed consensus"
- "Explore the economic incentives that secure the Bitcoin network"
- "Analyze the energy economics of Bitcoin mining and its sustainability impacts"
```

**Image Generation Example:**

When requesting an image, the system:
1. Generates a detailed description using the fine-tuned Satoshi model
2. Converts this into a visual prompt for DALL-E 3
3. Produces a high-quality, text-free educational diagram that visually explains the concept
4. Returns this diagram as an image response

The diagrams follow a consistent design language:
- Clean, minimalist Apple/Google-style aesthetic
- Bitcoin orange (#F7931A) as the primary accent color
- Visual flow that explains complex concepts through icons and symbols
- No text labels (to avoid readability issues and "alien language" problems)
- Professional infographic quality suitable for educational purposes

## Project Structure

- `training_model.py`: The main script that implements QLoRA fine-tuning
- `api.py`: FastAPI server that implements the hybrid model approach
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

4. Configure your API keys in a `.env` file:
```
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

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

## Running the API Server

After training the model, run the API server:

```bash
python api.py
```

This will start a FastAPI server on port 8000 that:
1. Loads the fine-tuned Llama-3.2-1B model with the LoRA adapter
2. Exposes an endpoint for generating responses using the hybrid approach
3. Connects to OpenAI's API for response enhancement and image generation

## API Endpoints

- `POST /generate`: Generate a response using the hybrid LLM approach
  - Request body: 
    ```json
    {
      "messages": [...], 
      "output_type": "text" or "image",
      "temperature": 0.2, 
      "max_tokens": 500
    }
    ```
  - Returns for text: 
    ```json
    {
      "type": "text", 
      "text": "response text",
      "recommendations": ["topic 1", "topic 2", "topic 3"]
    }
    ```
  - Returns for image: Streaming response with image data and headers containing metadata

- `GET /admin/view_persona`: View the cached Satoshi persona description
- `POST /admin/refresh_persona`: Refresh the cached Satoshi persona description

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

4. Clone the project repository:
   ```python
   !git clone https://github.com/lionxcr/satoshi-ai.git .
   ```

5. Install the required dependencies:
   ```python
   !pip install -r requirements.txt
   ```

6. Set up the Llama model access:
   ```python
   # Install Git LFS if not already installed
   !apt-get install git-lfs
   
   # Clone the Llama-3.2-1B model repository (requires access credentials)
   # Note: You'll need to authenticate with your HuggingFace credentials
   !git lfs install
   !git clone https://huggingface.co/meta-llama/Llama-3.2-1B
   
   # Alternative: Use the Hugging Face hub directly with an access token
   !pip install huggingface_hub
   !huggingface-cli login
   # Then enter your token when prompted
   ```

7. Verify or create the training data directories:
   ```python
   !mkdir -p training-data/output/bitcoinbook
   !mkdir -p training-data/output/bips
   !mkdir -p training-data/output/lnbook
   !mkdir -p training-data/output/emails
   !mkdir -p training-data/output/posts
   
   # If your training data is in a separate Git repository:
   !git clone https://github.com/lionxcr/satoshi-training-data.git temp-data
   !cp -r temp-data/* training-data/output/
   !rm -rf temp-data
   ```

8. Verify the PEFT configuration file exists or create it:
   ```python
   # Check if config exists
   !ls -la training-data/output/persona_peft_config.json || echo "Config file not found!"
   
   # Create config if needed
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

3. After training, push the results to your Git repository (optional):
   ```python
   # Configure Git user information
   !git config --global user.email "your.email@example.com"
   !git config --global user.name "Your Name"
   
   # Add and commit the adapter files (not the full model to save space)
   !git add satoshi-ai-model/adapter/
   !git commit -m "Add trained model adapter"
   !git push origin main
   ```

4. Alternatively, compress and download the model:
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

### Text-Free Image Generation

The image generation process uses a multi-step approach to create high-quality educational diagrams:

1. **Image Description Generation**: The fine-tuned Llama-3.2-1B model generates a detailed description of how to visually represent the Bitcoin concept
2. **Prompt Engineering**: The system reformats this description into a specialized prompt optimized for DALL-E 3, with strict instructions to:
   - Create text-free diagrams that communicate entirely through visual elements
   - Use a consistent design language with Bitcoin branding
   - Focus on clear visual flow and educational value
3. **HD Image Creation**: The system uses DALL-E 3's "hd" quality setting to generate crisp, detailed 1024x1024 diagrams
4. **Fallback Mechanism**: If DALL-E fails, the system reverts to rendering text directly on a simple image

This approach avoids the "alien language" problem that can occur with AI-generated text in images, resulting in cleaner, more professional educational visuals.

### Learning Recommendations

The recommendation system uses a two-phase approach:

1. **Generation Phase**:
   - The fine-tuned Llama-3.2-1B model creates initial topic recommendations based on the user's query
   - These are directly related to Bitcoin concepts that would deepen the user's understanding

2. **Refinement Phase**:
   - GPT-4o-mini processes these raw recommendations into concise, insightful learning suggestions
   - The output is formatted as exactly three specific, actionable recommendations
   - Each recommendation maintains Satoshi's perspective and voice

3. **Robust Parsing**:
   - The system includes multiple fallback mechanisms for JSON parsing
   - Line-by-line extraction in case of formatting issues
   - Default recommendations if all else fails

This feature adds educational value by proactively guiding users toward related Bitcoin concepts, creating a more comprehensive learning experience.

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

## Using the Fine-Tuned Model Directly

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

## Using the Hybrid Approach (Recommended)

For best results, use the API server which implements the hybrid approach:

```python
import requests
import json

url = "http://localhost:8000/generate"
headers = {"Content-Type": "application/json"}

# Get a text response with recommendations
data = {
    "messages": [
        {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
        {"role": "user", "content": "What are your thoughts on the Lightning Network?"}
    ],
    "output_type": "text",
    "temperature": 0.2,
    "max_tokens": 800
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

# Display the main response
print(result["text"])

# Display the learning recommendations
if "recommendations" in result:
    print("\nTo learn more, explore these topics:")
    for i, rec in enumerate(result["recommendations"], 1):
        print(f"{i}. {rec}")

# Generate an educational diagram about Bitcoin mining
image_data = {
    "messages": [
        {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
        {"role": "user", "content": "Create a diagram showing how Bitcoin transactions are verified in a block"}
    ],
    "output_type": "image",
    "temperature": 0.3
}

# For image responses, we get a streaming response with the image data
image_response = requests.post(url, headers=headers, data=json.dumps(image_data), stream=True)

# Save the image
if image_response.status_code == 200:
    # Extract metadata from headers if needed
    metadata = json.loads(image_response.headers.get("X-Response", "{}"))
    
    # Save the image to a file
    with open("bitcoin_diagram.png", "wb") as f:
        for chunk in image_response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Image saved to bitcoin_diagram.png")
    if "url" in metadata:
        print(f"Original image URL: {metadata['url']}")
```

You can also build a simple web interface to interact with the API, displaying both text responses with recommendations and educational diagrams.

## Advanced Usage: Creating Educational Content

The system can be used to generate comprehensive educational content about Bitcoin:

```python
import requests
import json
import os

url = "http://localhost:8000/generate"
headers = {"Content-Type": "application/json"}

# Define a learning path
topics = [
    "What is Bitcoin and how does it work?",
    "Explain Bitcoin's Proof of Work consensus mechanism",
    "How do Bitcoin transactions work?",
    "What is Bitcoin mining and why is it important?",
    "How does Bitcoin's blockchain prevent double-spending?",
    "What are Bitcoin wallets and private keys?",
    "Explain the concept of Bitcoin's 21 million coin limit"
]

# Create a directory for our educational content
os.makedirs("bitcoin_course", exist_ok=True)

# Generate text explanations and visual diagrams for each topic
for i, topic in enumerate(topics, 1):
    print(f"Generating content for topic {i}: {topic}")
    
    # Generate text explanation with recommendations
    text_response = requests.post(url, headers=headers, json={
        "messages": [
            {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
            {"role": "user", "content": topic}
        ],
        "output_type": "text",
        "max_tokens": 1000
    })
    
    text_data = text_response.json()
    
    # Generate companion educational diagram
    image_response = requests.post(url, headers=headers, json={
        "messages": [
            {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
            {"role": "user", "content": f"Create an educational diagram about: {topic}"}
        ],
        "output_type": "image"
    }, stream=True)
    
    # Save content to files
    with open(f"bitcoin_course/{i:02d}_{topic.replace('?', '').replace(' ', '_')[:30]}.md", "w") as f:
        f.write(f"# {topic}\n\n")
        f.write(text_data["text"])
        f.write("\n\n## Further Learning\n\n")
        for rec in text_data.get("recommendations", []):
            f.write(f"- {rec}\n")
    
    # Save image
    if image_response.status_code == 200:
        with open(f"bitcoin_course/{i:02d}_{topic.replace('?', '').replace(' ', '_')[:30]}.png", "wb") as f:
            for chunk in image_response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"  ✓ Content saved for topic {i}")

print("Educational content generation complete!")
```

This script generates a complete educational course on Bitcoin with text explanations, learning recommendations, and visual diagrams for each topic.

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

### API Connection Issues

If the API server fails to connect to OpenAI:

1. Verify your OpenAI API key in the `.env` file
2. Check internet connectivity
3. Set a higher timeout value in the API code
4. The system will fall back to using only the fine-tuned model if OpenAI is unavailable 