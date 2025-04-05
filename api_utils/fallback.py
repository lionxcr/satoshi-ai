"""
Fallback response handling utilities for Satoshi AI API
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

# Get logger for this module
logger = logging.getLogger(__name__)


def get_default_recommendations() -> List[str]:
    """
    Get default recommendations if recommendation generation fails.
    
    Returns:
        List of default recommendations
    """
    return [
        "Read the Bitcoin whitepaper to understand the original vision",
        "Study the Bitcoin Core codebase to understand implementation details",
        "Try implementing a simplified version of a Bitcoin transaction in code"
    ]


def generate_fallback_recommendations(rec_text: str) -> List[str]:
    """
    Generate recommendations from raw text when JSON parsing fails.
    
    Args:
        rec_text: Raw recommendation text
        
    Returns:
        List of recommendations
    """
    try:
        # Split by lines and clean up
        rec_lines = rec_text.split('\n')
        recommendations = [
            line.strip().strip('"\'-•').strip() 
            for line in rec_lines if line.strip()
        ][:3]
        
        # Ensure we have exactly 3 recommendations
        if len(recommendations) > 3:
            recommendations = recommendations[:3]
        while len(recommendations) < 3:
            recommendations.append(
                "Explore the Bitcoin whitepaper for foundational understanding"
            )
            
        return recommendations
    except Exception as e:
        logger.error(f"Error generating fallback recommendations: {e}")
        return get_default_recommendations()


def handle_response_truncation(response_text: str, finish_reason: Optional[str] = None) -> str:
    """
    Handle potentially truncated responses.
    
    Args:
        response_text: Generated response text
        finish_reason: Completion finish reason from OpenAI
        
    Returns:
        Cleaned response with proper ending if truncated
    """
    # If the response is empty or very short, return as is
    if not response_text or len(response_text) < 20:
        return response_text
        
    # If the response was truncated due to length, only trim from the end, never from the beginning
    if finish_reason == "length":
        logger.warning("Response was truncated by OpenAI due to length constraint")
        if response_text and len(response_text) > 3:
            # Try to find the last complete sentence
            last_period = max(
                response_text.rfind('.'), 
                response_text.rfind('!'), 
                response_text.rfind('?')
            )
            if last_period > len(response_text) * 0.7:  # Only truncate if we have most of the response
                response_text = response_text[:last_period+1]
                logger.info("Trimmed to last complete sentence (end only)")
    
    # Check for incomplete ending (ends without punctuation)
    if response_text and len(response_text) > 3:
        if not response_text[-1] in ['.', '!', '?', ':', ';', '"', "'", ')', ']', '}']:
            # Try to find the last complete sentence
            last_period = max(
                response_text.rfind('.'), 
                response_text.rfind('!'), 
                response_text.rfind('?')
            )
            if last_period > len(response_text) * 0.7:  # Only truncate if we have most of the response
                response_text = response_text[:last_period+1]
                logger.info("Truncated incomplete sentence at the end only")
    
    return response_text


def process_fallback_response(fine_tuned_response: str, max_tokens: int) -> str:
    """
    Process fine-tuned model response as a fallback when OpenAI fails.
    
    Args:
        fine_tuned_response: Response from fine-tuned model
        max_tokens: Maximum tokens requested
        
    Returns:
        Processed response
    """
    # Create a more polished version of the fine-tuned response without stripping from beginning
    response_text = fine_tuned_response
    
    # Handle truncation and cleanup
    response_text = handle_response_truncation(response_text)
    
    # Also limit the fallback response to a reasonable length, but only from the end
    if len(response_text) > max_tokens * 4:  # Using chars as approximation
        logger.warning(f"Response too long ({len(response_text)} chars), trimming from the end only")
        # Find a good place to cut from the end
        cutoff_point = max_tokens * 4
        # Try to find the last complete sentence before the cutoff
        content_to_search = response_text[:cutoff_point]
        last_period = max(
            content_to_search.rfind('.'), 
            content_to_search.rfind('!'), 
            content_to_search.rfind('?')
        )
        if last_period > len(content_to_search) * 0.7:  # Only truncate if we have most of the content
            response_text = response_text[:last_period+1]
        else:
            # Just truncate with an ellipsis at the end
            response_text = content_to_search + "..."
            
    return response_text 