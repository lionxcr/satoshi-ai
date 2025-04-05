# Satoshi AI - A Bitcoin Expert Chatbot with Satoshi Nakamoto's Persona

## Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [How it Works](#how-it-works)
- [Features](#features)
- [Example Interaction](#example-interaction)
- [Visual Learning](#visual-learning)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Running the API](#running-the-api)
- [Training the Model](#training-the-model)
  - [Training Data Preparation](#training-data-preparation)
  - [Training Process](#training-process)
  - [Training on Google Colab](#training-on-google-colab)
- [Using the API](#using-the-api)
  - [Text Responses](#text-responses)
  - [Image Generation](#image-generation)
  - [Creating Educational Content](#creating-educational-content)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [How to Contribute](#how-to-contribute)
- [Related Resources](#related-resources)
- [Technical Details](#technical-details)

## Introduction

Satoshi AI creates a Bitcoin expert chatbot that embodies the persona of Satoshi Nakamoto (Bitcoin's creator). It combines the deep technical knowledge of Bitcoin with an engaging, educational user experience.

The project gives you a personal Bitcoin tutor who can:
- Answer technical questions about Bitcoin and blockchain
- Generate educational diagrams to explain complex concepts
- Suggest personalized learning paths based on your questions
- Communicate with the authentic voice of Satoshi Nakamoto

## Quick Start

For those who want to immediately try Satoshi AI:

1. **Pre-trained model** (fastest option):
   ```bash
   # Clone the repository
   git clone https://github.com/lionxcr/satoshi-ai.git && cd satoshi-ai
   
   # Set up environment and install dependencies
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   
   # Create a .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_key_here" > .env
   
   # Download pre-trained model weights (if available)
   ./download_model.sh
   
   # Start the API server
   python api.py
   
   # The API is now running at http://localhost:8000
   ```

2. **Use our demo** (no setup needed):
   Try the public demo at [https://satoshi-ai-demo.com](https://satoshi-ai-demo.com)

## How it Works

We use a hybrid approach that combines the best of smaller specialized models with powerful large language models:

1. A **Bitcoin Expert Model**: We fine-tune a smaller model (Llama-3.2-1B) to become deeply knowledgeable about Bitcoin, blockchain, and Satoshi's writings
   
2. A **User Experience Layer**: We use GPT-4o-mini to improve readability and presentation while keeping all the technical details accurate

3. A **Visual Learning Tool**: We generate clear educational diagrams with DALL-E 3 to help you understand complex concepts

4. A **Learning Guide**: The system suggests related topics to explore, creating a personalized learning path

This approach gives you technically accurate information in a user-friendly, engaging format with the authentic voice of Satoshi Nakamoto.

## Features

- **Deep Bitcoin Knowledge**: Trained on Bitcoin technical documentation, books, and Satoshi's writings
- **Authentic Satoshi Voice**: Responds in the style of Bitcoin's mysterious creator
- **Educational Diagrams**: Creates clear visual explanations of complex concepts
- **Learning Recommendations**: Suggests related topics to deepen your understanding
- **Easy-to-use API**: Simple integration with web or mobile applications

## Example Interaction

**User Question:** Why did you choose Proof of Work for Bitcoin?

**Satoshi AI Response:**
```
I chose Proof of Work for Bitcoin because it elegantly solves several critical challenges in creating a trustless digital currency.

First, PoW establishes an objective, measurable cost to participation. Miners must demonstrate computational work, creating a resource-backed security model that can't be circumvented through virtual identities or network manipulation. This prevents Sybil attacks where an attacker might create numerous virtual nodes.

Second, PoW creates a verifiable chronological record that cannot be altered without redoing the work. The blockchain becomes an immutable history precisely because changing any block requires redoing all subsequent proof-of-work - an insurmountable task unless an attacker controls more computational power than the honest network.

Third, PoW enables decentralized consensus without trusted authorities. The longest chain rule provides a clear, objective metric for determining the valid transaction history.

The energy expenditure is not wasteful - it's the fundamental mechanism that secures the network. It transforms electrical energy into immutable trust, providing the backbone of Bitcoin's security model.

Alternatives like Proof of Stake existed when I designed Bitcoin, but they introduce complex game-theoretical vulnerabilities and centralization pressures that I deliberately chose to avoid.
```

**Learning Recommendations:**
- Study the Byzantine Generals Problem and how PoW solves distributed consensus
- Explore the economic incentives that secure the Bitcoin network
- Analyze the energy economics of Bitcoin mining and its sustainability impacts

## Visual Learning

When you ask for a visual explanation, Satoshi AI creates educational diagrams that:
- Use clean, minimalist design with Bitcoin's orange color scheme
- Explain concepts through visual flow and intuitive symbols
- Avoid text labels for clarity (preventing the "alien language" problem)
- Present information at a professional infographic quality

For example, asking "Show me how Bitcoin mining works" generates a diagram showing the mining process from transactions to block verification.

![Example Bitcoin Mining Diagram](./assets/bitcoin_mining_example.png)

## System Architecture

Satoshi AI uses a three-component architecture to deliver accurate, user-friendly Bitcoin education:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                  │     │                  │     │                  │
│  Bitcoin Expert  │────▶│  UX Enhancement  │────▶│  Response/Image  │
│      Model       │     │      Layer       │     │    Generation    │
│  (Llama-3.2-1B)  │     │   (GPT-4o-mini)  │     │    (DALL-E 3)    │
│                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        ▲                                                  │
        │                                                  │
        └──────────────────────────────────────────────────┘
                          Learning Path
                          Suggestions
```

1. **User Query Flow**:
   - Your Bitcoin question enters the system
   - The Bitcoin Expert Model creates a technically accurate response
   - The UX Enhancement Layer improves readability while preserving accuracy
   - You receive a polished answer with learning recommendations

2. **Image Request Flow**:
   - Your request for a visual explanation enters the system  
   - The Bitcoin Expert Model creates a detailed diagram description
   - This description is formatted for optimal image generation
   - DALL-E 3 creates a clear educational diagram
   - You receive a visual explanation of the Bitcoin concept

## Getting Started

### Prerequisites

To run Satoshi AI, you'll need:
- Python 3.9+ installed
- Access to the Llama-3.2-1B model (from Hugging Face)
- An OpenAI API key for the GPT-4o-mini and DALL-E 3 components
- Sufficient storage for models (at least 5GB)
- A CUDA-capable GPU with 8GB+ VRAM (for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lionxcr/satoshi-ai.git
cd satoshi-ai
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your API keys in a `.env` file:
```
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

4. Get the Llama-3.2-1B model from Hugging Face (requires an account):
```bash
# Option 1: Use the Hugging Face CLI
huggingface-cli login
# Enter your token when prompted

# Option 2: Clone the model repository with Git LFS
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
```

### Running the API

1. Start the API server:
```bash
python api.py
```

2. The server will run on http://localhost:8000 by default
3. You can now send requests to the API for Bitcoin information and educational diagrams

## Training the Model

Satoshi AI uses a fine-tuned Llama-3.2-1B model that has been trained on Bitcoin documentation, books, and Satoshi's writings.

### Training Data Preparation

The model is trained using data from these main sources:

| Source | Description | 
|--------|-------------|
| Bitcoin Improvement Proposals (BIPs) | Technical documentation for Bitcoin protocol enhancements |
| Mastering Bitcoin | Comprehensive guide to Bitcoin by Andreas M. Antonopoulos |
| Mastering Lightning Network | Guide to Lightning Network by Andreas M. Antonopoulos et al. |
| Bitcoin Core | Reference implementation of Bitcoin |
| Satoshi's Emails & Forum Posts | Original writing by Satoshi Nakamoto |

To prepare the training data:

1. Clone the necessary repositories into the training-data directory:
```bash
# Inside the training-data directory
git clone https://github.com/bitcoin/bips.git bips
git clone https://github.com/bitcoinbook/bitcoinbook.git bitcoinbook
git clone https://github.com/lnbook/lnbook.git lnbook
```

2. Process the training data:
```bash
cd training-data
python main.py
```

This creates properly formatted datasets in the `training-data/output` directory.

### Training Process

To train the model:

1. Ensure your GPU is properly set up with CUDA
2. Run the training script:
```bash
python training_model.py
```

The training process:
- Loads the Llama-3.2-1B model in memory-efficient mode
- Applies special fine-tuning techniques that update only a small portion of the model
- Uses the processed Bitcoin and Lightning Network training data
- Saves the trained model to the `satoshi-ai-model` directory

### Training on Google Colab

If you don't have a powerful GPU, you can train on Google Colab for free:

1. Create a new notebook at [Google Colab](https://colab.research.google.com/)
2. Set up Google Drive to store your data:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Clone the project and set it up:
```python
!git clone https://github.com/lionxcr/satoshi-ai.git /content/drive/MyDrive/satoshi-ai
%cd /content/drive/MyDrive/satoshi-ai
!pip install -r requirements.txt
```

4. Run the training:
```python
!python training_model.py
```

The model will be saved to your Google Drive when training is complete.

## Using the API

### Text Responses

To get a text response about Bitcoin:

```python
import requests
import json

url = "http://localhost:8000/generate"
headers = {"Content-Type": "application/json"}

data = {
    "messages": [
        {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
        {"role": "user", "content": "How does Bitcoin achieve decentralization?"}
    ],
    "output_type": "text",
    "temperature": 0.2,  # Controls randomness (0.0-1.0)
    "max_tokens": 500    # Maximum length of response
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

# Print the main response
print(result["text"])

# Print learning recommendations
for rec in result["recommendations"]:
    print(f"- {rec}")
```

### Image Generation

To generate an educational diagram:

```python
image_data = {
    "messages": [
        {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
        {"role": "user", "content": "Create a diagram showing how blockchain works"}
    ],
    "output_type": "image",
    "temperature": 0.2
}

image_response = requests.post(url, headers=headers, data=json.dumps(image_data), stream=True)

# Save the image
with open("blockchain_diagram.png", "wb") as f:
    for chunk in image_response.iter_content(chunk_size=8192):
        f.write(chunk)
```

### Creating Educational Content

You can create a complete educational course on Bitcoin:

```python
import os
import requests
import json

# Define a series of topics to learn about Bitcoin
topics = [
    "What is Bitcoin?",
    "How does blockchain work?",
    "What is Bitcoin mining?",
    "How do Bitcoin transactions work?",
    "What is the Bitcoin halving?",
    "How does Bitcoin's security model work?"
]

# Create directory for the course
os.makedirs("bitcoin_course", exist_ok=True)

# For each topic, generate both text explanations and visual diagrams
for i, topic in enumerate(topics, 1):
    print(f"Generating content for topic {i}: {topic}")
    
    # Get text explanation with learning recommendations
    text_response = requests.post(
        "http://localhost:8000/generate",
        headers={"Content-Type": "application/json"},
        json={
            "messages": [
                {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
                {"role": "user", "content": topic}
            ],
            "output_type": "text",
            "max_tokens": 800
        }
    )
    text_data = text_response.json()
    
    # Get educational diagram
    image_response = requests.post(
        "http://localhost:8000/generate",
        headers={"Content-Type": "application/json"},
        json={
            "messages": [
                {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin."},
                {"role": "user", "content": f"Create an educational diagram about: {topic}"}
            ],
            "output_type": "image"
        },
        stream=True
    )
    
    # Save text content
    filename_base = f"{i:02d}_{topic.replace('?', '').replace(' ', '_')[:30]}"
    with open(f"bitcoin_course/{filename_base}.md", "w") as f:
        f.write(f"# {topic}\n\n")
        f.write(text_data["text"])
        f.write("\n\n## Further Learning\n\n")
        for rec in text_data.get("recommendations", []):
            f.write(f"- {rec}\n")
    
    # Save image
    with open(f"bitcoin_course/{filename_base}.png", "wb") as f:
        for chunk in image_response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"  ✓ Content saved for topic {i}")

print("Educational course generation complete!")
```

This creates a structured learning experience with both text and visuals.

## Troubleshooting

### Common Issues

**Memory Errors During Training:**
- Reduce batch size in the configuration file
- Try running on Google Colab with free GPUs
- Make sure no other programs are using your GPU memory

**API Connection Issues:**
- Check that your API keys are correctly set in the `.env` file
- Verify internet connectivity if using OpenAI's services
- The system will fall back to the local model if OpenAI is unavailable

**Image Generation Problems:**
- If DALL-E image generation fails, the system will create a simpler diagram
- Check your OpenAI API key and usage limits
- Try lowering the complexity of your diagram request

## FAQ

### General Questions

**Q: Who is Satoshi Nakamoto?**  
A: Satoshi Nakamoto is the pseudonymous creator of Bitcoin who published the Bitcoin whitepaper in 2008 and developed the first Bitcoin implementation. Their true identity remains unknown.

**Q: Is this really Satoshi Nakamoto?**  
A: No, this is an AI system fine-tuned to respond with deep Bitcoin knowledge in a style similar to Satoshi Nakamoto's writings. It's an educational tool, not an attempt to impersonate the real Satoshi.

**Q: How accurate is the Bitcoin information?**  
A: The system is trained on technical Bitcoin documentation, books by renowned experts, and Satoshi's own writings. While generally accurate, always verify critical information from multiple sources, especially for security or investment decisions.

### Technical Questions

**Q: Can I run this without a GPU?**  
A: Yes, you can run the API server for inference on a CPU, though responses will be slower. Training the model does require a GPU, but you can use our pre-trained model or train on Google Colab.

**Q: What's the difference between using just the fine-tuned model and the hybrid approach?**  
A: The fine-tuned model alone provides technically accurate Bitcoin information but may have limitations in presentation quality. The hybrid approach enhances readability and generates high-quality diagrams.

**Q: Is my data private when using this system?**  
A: When running locally, your queries stay on your machine. However, when using the GPT-4o-mini and DALL-E components, your prompts are sent to OpenAI's API under their privacy policy.

**Q: Can I use this commercially?**  
A: This project is released under [LICENSE_TYPE]. For commercial use, please review the license terms and the terms of service for any third-party APIs used (like OpenAI).

## How to Contribute

We welcome contributions to improve Satoshi AI! Here's how you can help:

### Getting Started

1. **Fork the repository** and clone it locally
2. **Create a new branch** for your feature or bugfix
3. **Make your changes** following our code style
4. **Write or update tests** for your changes
5. **Submit a pull request** describing your changes

### Areas for Contribution

- **Model Training**: Improve fine-tuning techniques or explore alternative models
- **Training Data**: Add more high-quality Bitcoin resources to the training data
- **UI/UX**: Develop web or mobile interfaces for the API
- **Documentation**: Improve the README or add tutorials
- **Bug Fixes**: Address issues in the issue tracker

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Write clear commit messages explaining your changes
- Add comments to complex code sections
- Update documentation for any API changes

## Related Resources

### Bitcoin Education

- [Bitcoin Whitepaper](https://bitcoin.org/bitcoin.pdf) - The original document by Satoshi Nakamoto
- [Mastering Bitcoin](https://github.com/bitcoinbook/bitcoinbook) - Comprehensive guide by Andreas M. Antonopoulos
- [Bitcoin Developer Documentation](https://developer.bitcoin.org/) - Technical resources for developers
- [Jameson Lopp's Bitcoin Resources](https://www.lopp.net/bitcoin-information.html) - Curated list of Bitcoin resources

### AI and Machine Learning

- [Hugging Face](https://huggingface.co/) - Hub for pre-trained models and tools
- [Papers With Code](https://paperswithcode.com/) - Machine learning papers with code implementations
- [OpenAI Documentation](https://platform.openai.com/docs/) - Guides for working with OpenAI's APIs
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Tutorials for deep learning with PyTorch

### Similar Projects

- [Bitcoin Core ChatGPT Plugin](https://github.com/michaelgold/bitcoin-core-chatgpt-plugin) - OpenAI plugin for accessing Bitcoin documentation
- [Lightning Terminal](https://github.com/lightninglabs/lightning-terminal) - UI for the Lightning Network with educational components
- [Stacks AI](https://github.com/hirosystems/stacks.ai) - AI assistant for the Stacks blockchain ecosystem

## Technical Details

For those interested in the technical implementation:

- **Model Fine-tuning**: We use a technique called QLoRA (Quantized Low-Rank Adaptation) that efficiently fine-tunes models on consumer hardware
  
- **Hybrid Architecture**: We combine a specialized Bitcoin expert model with a larger language model for the best balance of knowledge and presentation
  
- **Image Generation Pipeline**: The system creates detailed descriptions of diagrams, then carefully formats them for DALL-E 3 to create educational visuals
  
- **Learning Recommendations**: The AI identifies related topics that would deepen your understanding based on what you've already asked about

- **Memory Optimization**: Special techniques allow the model to run efficiently even on consumer hardware

The full source code is available in this repository, with detailed comments explaining the implementation details. 

---

**Version:** 1.0.0  
**License:** MIT  
**Contact:** satoshi-ai@example.com  
**Project Status:** Active Development 