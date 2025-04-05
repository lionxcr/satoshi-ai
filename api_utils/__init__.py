"""
Utility modules for Satoshi AI API

This package contains helper functions and utilities used by the main API.
"""

# Import key functions to make them available directly from the package
from .cost import calculate_token_cost
from .validation import evaluate_response_quality, validate_response_start
from .persona import generate_satoshi_persona, get_satoshi_persona, get_satoshi_system_prompt
from .models import load_model_and_tokenizer, initialize_openai_client
from .prompts import (
    get_quality_evaluation_prompt,
    get_response_regeneration_prompt,
    get_openai_enhanced_response_prompt,
    get_recommendations_system_prompt,
    get_llm_prompt_for_image,
    get_image_prompt,
    get_bitcoin_recommendations_prompt
)
from .openai_helpers import (
    call_openai,
    call_openai_with_fallback,
    extract_json_from_text
)
from .fallback import (
    get_default_recommendations,
    generate_fallback_recommendations,
    handle_response_truncation,
    process_fallback_response
)
from .image import (
    create_text_image,
    download_image,
    extract_core_description
)

__all__ = [
    # Cost calculation
    'calculate_token_cost',
    
    # Validation
    'evaluate_response_quality',
    'validate_response_start',
    
    # Persona management
    'generate_satoshi_persona',
    'get_satoshi_persona',
    'get_satoshi_system_prompt',
    
    # Model loading
    'load_model_and_tokenizer',
    'initialize_openai_client',
    
    # Prompts
    'get_quality_evaluation_prompt',
    'get_response_regeneration_prompt',
    'get_openai_enhanced_response_prompt',
    'get_recommendations_system_prompt',
    'get_llm_prompt_for_image',
    'get_image_prompt',
    'get_bitcoin_recommendations_prompt',
    
    # OpenAI helpers
    'call_openai',
    'call_openai_with_fallback',
    'extract_json_from_text',
    
    # Fallback handling
    'get_default_recommendations',
    'generate_fallback_recommendations',
    'handle_response_truncation',
    'process_fallback_response',
    
    # Image utilities
    'create_text_image',
    'download_image',
    'extract_core_description',
] 