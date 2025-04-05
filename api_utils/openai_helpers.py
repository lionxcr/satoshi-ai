"""
Helper functions for interacting with OpenAI API
"""
import logging
import json
import re
from typing import Dict, Any, Optional, List, Union

# Get logger for this module
logger = logging.getLogger(__name__)


def call_openai(
    client, 
    model: str, 
    messages: List[Dict[str, str]],
    max_tokens: int = 800,
    temperature: float = 0.5,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Call OpenAI API with the provided parameters.
    
    Args:
        client: OpenAI client
        model: Model name (e.g., "gpt-4o-mini")
        messages: List of message dictionaries
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        presence_penalty: Presence penalty
        frequency_penalty: Frequency penalty
        top_p: Top-p sampling
        stop: Stop sequences
        
    Returns:
        Response dictionary with 'content', 'usage', and 'finish_reason'
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            stop=stop
        )
        
        # Extract content, token usage, and finish reason
        content = response.choices[0].message.content.strip()
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                "completion_tokens": response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
            }
            
        finish_reason = None
        if hasattr(response.choices[0], 'finish_reason'):
            finish_reason = response.choices[0].finish_reason
            
        return {
            "content": content,
            "usage": usage,
            "finish_reason": finish_reason
        }
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return {
            "content": None,
            "error": str(e),
            "usage": None,
            "finish_reason": None
        }


def call_openai_with_fallback(
    client,
    model: str,
    fallback_model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 800,
    temperature: float = 0.5,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0
) -> Dict[str, Any]:
    """
    Call OpenAI API with a fallback model if the primary model fails.
    
    Args:
        client: OpenAI client
        model: Primary model name
        fallback_model: Fallback model name
        messages: List of message dictionaries
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        presence_penalty: Presence penalty
        frequency_penalty: Frequency penalty
        
    Returns:
        Response dictionary with 'content', 'usage', 'finish_reason', and 'model_used'
    """
    # Try with the primary model
    response = call_openai(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    )
    
    # If successful, return with the primary model name
    if response["content"] is not None:
        response["model_used"] = model
        return response
        
    # If failed, try with the fallback model
    logger.warning(f"Primary model {model} failed, falling back to {fallback_model}")
    fallback_response = call_openai(
        client=client,
        model=fallback_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    )
    
    # Add the model used
    fallback_response["model_used"] = fallback_model if fallback_response["content"] is not None else None
    return fallback_response


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from a text that might contain additional content.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Parsed JSON object or empty dict if parsing fails
    """
    try:
        # Try to extract JSON if it's embedded in text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            return json.loads(json_text)
            
        # Try to extract array if it's embedded in text
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            array_text = array_match.group(0)
            return json.loads(array_text)
            
        # If no JSON object or array found, try parsing the whole text
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return {} 