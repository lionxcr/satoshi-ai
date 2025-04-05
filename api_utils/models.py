"""
Model loading and management utilities for Satoshi AI API
"""
import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI

# Get logger for this module
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(hf_token=None):
    """
    Load the base model and LoRA adapter, along with the tokenizer.
    
    Args:
        hf_token: Hugging Face token for authentication
        
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Use the original Llama 3.2 1B model as our LoRA adapter is compatible with it
        base_model_id = "meta-llama/Llama-3.2-1B-Instruct"  # Reverting to 1B model for compatibility
        adapter_path = "satoshi-ai-model"  # This folder should contain the adapter weights and tokenizer files

        # Check if HF_TOKEN is set, use the one provided or from env
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token or hf_token == "YOUR_HUGGING_FACE_TOKEN_HERE":
                logger.warning(
                    "HF_TOKEN not set or contains placeholder. "
                    "You need a valid Hugging Face token to download Llama models."
                )
        
        logger.info(f"Loading models and tokenizers...")
        
        # Check if CUDA is available, otherwise use CPU
        if torch.cuda.is_available():
            logger.info("CUDA is available. Using GPU.")
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            logger.info("CUDA is not available. Using CPU.")
            device_map = "cpu"
            torch_dtype = torch.float32
        
        # Load the tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, 
            token=hf_token,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load the base model with specific configuration to avoid offloading issues
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Check if adapter exists locally
        if os.path.exists(adapter_path):
            # Load the adapter with PEFT
            try:
                model = PeftModel.from_pretrained(
                    base_model, 
                    adapter_path,
                    device_map=device_map
                )
                logger.info("LoRA adapter loaded successfully")
            except ValueError as ve:
                # If we still get offloading error, use the base model instead
                if "offload" in str(ve):
                    logger.warning(f"Could not load adapter due to offloading error: {ve}")
                    logger.warning("Using base model without adapter")
                    model = base_model
                else:
                    raise
        else:
            logger.warning(f"Adapter path {adapter_path} not found. Using base model only.")
            model = base_model
            
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error during model loading: {e}", exc_info=True)
        raise e


def initialize_openai_client(api_key=None):
    """
    Initialize the OpenAI client with the provided API key.
    
    Args:
        api_key: OpenAI API key, if None will use environment variable
        
    Returns:
        OpenAI client instance or None if initialization failed
    """
    try:
        # Use provided API key or get from environment
        openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not set. You need a valid OpenAI API key.")
            logger.warning(
                "Please set a valid token in your .env file or as an environment variable."
            )
            return None
            
        # Clean initialization without additional parameters
        openai_client = OpenAI(
            api_key=openai_api_key,
        )
        logger.info("OpenAI client initialized successfully")
        return openai_client
        
    except TypeError as e:
        # Alternative initialization method for different versions
        logger.error(f"Error with standard initialization: {e}")
        try:
            from openai import api_key as openai_api_key_setter
            openai_api_key_setter = openai_api_key
            openai_client = OpenAI()
            logger.info("OpenAI client initialized with alternative method")
            return openai_client
        except Exception as alt_e:
            logger.error(f"Could not initialize OpenAI client: {alt_e}")
            return None 