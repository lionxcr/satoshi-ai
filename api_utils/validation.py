"""
Response validation and quality evaluation utilities for Satoshi AI API
"""
import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

# Import our utility for extracting JSON
from api_utils.openai_helpers import extract_json_from_text

# Get logger for this module
logger = logging.getLogger(__name__)


def evaluate_response_quality(response_text: str, user_query: str, model_client) -> Dict[str, Any]:
    """
    Evaluate the quality of a generated response, checking for completeness, clarity,
    and alignment with the user's query.
    
    Args:
        response_text: The generated response to evaluate
        user_query: The original user query
        model_client: OpenAI client instance
    
    Returns:
        Dict containing:
        - quality_score: float between 0-100
        - is_complete: bool indicating if response is complete
        - feedback: str with improvement suggestions
        - is_acceptable: bool indicating if quality meets threshold (88%)
    """
    logger.info("Evaluating response quality...")
    
    # First, perform a pre-check for truncated beginning
    # The key patterns that indicate response truncation at the beginning
    truncated_beginning_patterns = [
        r"^[a-z]",  # Starts with lowercase letter
        r"^\s*[,;:]",  # Starts with punctuation
        r"^(the|this|that|these|those|it|they|them|their|there|here|and|or|but|so|because|since|although|yet|therefore|thus|however|nevertheless)",  # Starts with conjunction or pronoun
        r"^[^\w#\*\-\>\`]",  # Doesn't start with word char, markdown, or code block
    ]
    
    # Check if the response might be truncated at the beginning
    might_be_truncated = False
    if response_text:
        first_words = response_text.strip().split()[:3] if response_text.strip() else []
        first_line = response_text.strip().split('\n')[0] if response_text.strip() else ""
        
        # Early truncation detection
        for pattern in truncated_beginning_patterns:
            if re.match(pattern, response_text.strip(), re.IGNORECASE):
                might_be_truncated = True
                break
                
        # Check if the first sentence is very short and doesn't make sense alone
        if first_line and len(first_line) < 30 and first_line[-1] not in ['.', '!', '?']:
            # Short first line without ending punctuation is suspicious
            might_be_truncated = True
    
    # If response is very short, it's likely incomplete
    if len(response_text) < 100:
        return {
            "quality_score": 50.0,
            "is_complete": False,
            "feedback": "Response is too short to be complete. Expand on the answer.",
            "is_acceptable": False
        }
    
    # Check for basic structural completeness
    has_intro = any(s in response_text[:200].lower() for s in ["bitcoin", "blockchain", "cryptocurrency", "system", "network", "protocol"])
    has_conclusion = any(s in response_text[-200:].lower() for s in ["summary", "conclusion", "finally", "in essence", "ultimately", "in conclusion", "to summarize", "in closing"])
    ends_properly = response_text[-1] in ['.', '!', '?', '"', "'", ')', ']', '}']
    
    # Quick-fail for obviously truncated responses
    if might_be_truncated:
        return {
            "quality_score": 50.0,
            "is_complete": False,
            "feedback": "Response appears to be truncated at the beginning. Please ensure it starts with a proper introduction and begins with one of these phrases: 'In Bitcoin...', 'To answer your question about...', 'The key to understanding...', or 'Let me explain...'.",
            "is_acceptable": False
        }
    
    # Enhanced check for proper beginning (not starting mid-sentence)
    starts_properly = False
    if len(response_text) > 10:
        # Response should start with a capital letter, or a markdown heading or list
        first_char = response_text[0]
        starts_properly = (first_char.isupper() or first_char in ['#', '-', '*', '1', '>', '`'])
        # Also check for markdown formatting indicators at start
        if not starts_properly:
            starts_properly = any(response_text.startswith(md) for md in ['# ', '## ', '### ', '- ', '* ', '1. ', '> '])
        
        # If still not marked as starting properly, perform deeper checks
        if not starts_properly:
            # Check first few words - might be lower case but should form a coherent phrase
            first_words = ' '.join(response_text.split()[:3]).lower()
            
            # Check if it starts with common pronouns or conjunctions that shouldn't start a response
            starts_with_bad_pronoun = any(first_words.startswith(p) for p in [
                'it ', 'they ', 'their ', 'these ', 'those ', 'this ', 'that ', 
                'and ', 'or ', 'but ', 'so ', 'because ', 'since ', 'although ',
                'which ', 'where ', 'when ', 'how ', 'why ', 'what ', 'who ',
                'additionally', 'furthermore', 'moreover', 'therefore', 'thus',
                'however', 'nevertheless', 'also', 'yet', 'then', 'now'
            ])
            
            # Response is cut off at beginning if it starts with a pronoun/conjunction without context
            if starts_with_bad_pronoun:
                starts_properly = False
            else:
                # If it doesn't start with pronoun, it might be okay even if lowercase
                starts_properly = True
                
            # Check if first sentence is grammatically complete
            first_sentence_end = response_text.find('.')
            if first_sentence_end > 0:
                first_sentence = response_text[:first_sentence_end].strip()
                # If first sentence is too short, it might be truncated
                if 0 < len(first_sentence) < 20:
                    # Very short first sentence is suspicious
                    starts_properly = False
    
    # Check for preferred beginning phrases
    has_preferred_beginning = False
    preferred_beginnings = [
        "in bitcoin", 
        "to answer your question about", 
        "the key to understanding", 
        "let me explain"
    ]
    
    for phrase in preferred_beginnings:
        if response_text.lower().startswith(phrase):
            has_preferred_beginning = True
            break
    
    # Boost scores for responses that follow our preferred beginning format
    beginning_bonus = 10.0 if has_preferred_beginning else 0.0
    
    # Check if response starts directly with a code block without introduction
    starts_with_code = False
    if len(response_text) > 10:
        starts_with_code = response_text.startswith('```')
    
    # Create evaluation prompt - simplified for gpt-4o-mini to ensure it completes properly
    evaluation_prompt = f"""Evaluate this Bitcoin response quality (1-100 scale):

Query: "{user_query}"

Response:
'''
{response_text}
'''

Score these criteria (1-100):
1. COMPLETENESS: Fully addresses query without cutting off?
2. CLARITY: Easy to understand for semi-technical people?
3. ACCURACY: Technically accurate for Bitcoin?
4. STRUCTURE: Has proper beginning and conclusion?
5. FORMATTING: Good markdown usage?
6. PROPER BEGINNING: Starts with complete intro (not cut off)?

For each, provide:
- Score (1-100)
- Short justification
- Brief suggestion if below 90

Return ONLY a JSON object with these exact keys:
{{
  "completeness": {{ "score": 0-100, "justification": "string", "suggestion": "string" }},
  "clarity": {{ "score": 0-100, "justification": "string", "suggestion": "string" }},
  "accuracy": {{ "score": 0-100, "justification": "string", "suggestion": "string" }},
  "structure": {{ "score": 0-100, "justification": "string", "suggestion": "string" }},
  "formatting": {{ "score": 0-100, "justification": "string", "suggestion": "string" }},
  "proper_beginning": {{ "score": 0-100, "justification": "string", "suggestion": "string" }},
  "overall_score": 0-100,
  "is_acceptable": true/false,
  "improvement_suggestions": ["string", "string", "string"]
}}
"""
    
    try:
        # Call OpenAI to evaluate the response - switched to gpt-4o-mini for faster, more reliable evaluation
        logger.info("Using gpt-4o-mini for response quality evaluation")
        evaluation_response = model_client.chat.completions.create(
            model="gpt-4o-mini",  # Changed from gpt-4o to gpt-4o-mini
            messages=[
                {"role": "system", "content": "You are a quality evaluation assistant for Bitcoin technical content. Your job is to assess response quality. Focus especially on whether the response has a proper beginning or appears cut off."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=800,  # Reduced from 1000 to be more efficient with mini model
            temperature=0.1,  # Low temperature for consistent evaluations
        )
        
        # Extract the evaluation
        evaluation_text = evaluation_response.choices[0].message.content.strip()
        
        # Use our utility function to extract JSON from text
        evaluation = extract_json_from_text(evaluation_text)
        
        # If we got an empty dict, something went wrong with the parsing
        if not evaluation:
            logger.warning(f"Failed to parse evaluation JSON: {evaluation_text[:100]}...")
            # Provide a basic evaluation
            return {
                "score": 75.0,
                "feedback": "Unable to properly evaluate response due to parsing issues.",
                "is_acceptable": overall_score >= 88.0
            }
            
        # Extract key metrics
        overall_score = evaluation.get("overall_score", 0)
        is_acceptable = evaluation.get("is_acceptable", False)
        
        # Check specifically for proper beginning issues - even more aggressive penalty
        beginning_score = evaluation.get("proper_beginning", {}).get("score", 0)
        if beginning_score < 85:  # Increased threshold from 80 to 85
            logger.warning(f"Response has poor beginning (score: {beginning_score}/100)")
            # Apply more aggressive penalty
            if overall_score > 70:
                overall_score = max(60, overall_score - 30)  # Increased penalty from 25 to 30
                is_acceptable = overall_score >= 88.0
        
        # Apply bonus for preferred beginnings
        if has_preferred_beginning:
            overall_score = min(100, overall_score + beginning_bonus)
            is_acceptable = overall_score >= 88.0
        
        # Combine improvement suggestions
        suggestions = evaluation.get("improvement_suggestions", [])
        criteria_suggestions = []
        for criterion in ["completeness", "clarity", "accuracy", "structure", "formatting", "proper_beginning"]:
            if criterion in evaluation and "suggestion" in evaluation[criterion]:
                suggestion = evaluation[criterion]["suggestion"]
                if suggestion and len(suggestion) > 5:  # Only add non-empty suggestions
                    criteria_suggestions.append(suggestion)
        
        # Prioritize beginning issues in suggestions
        if "proper_beginning" in evaluation and "suggestion" in evaluation["proper_beginning"]:
            beginning_suggestion = evaluation["proper_beginning"]["suggestion"]
            if beginning_suggestion and len(beginning_suggestion) > 5:
                # Add to beginning of suggestions list
                criteria_suggestions.insert(0, beginning_suggestion)
                # If beginning issues, add our specific requirements
                if beginning_score < 90:
                    criteria_suggestions.insert(0, "Start with one of these phrases: 'In Bitcoin...', 'To answer your question about...', 'The key to understanding...', or 'Let me explain...'")
        
        # Combine all suggestions, remove duplicates, and limit to top 3
        all_suggestions = suggestions + criteria_suggestions
        unique_suggestions = list(dict.fromkeys(all_suggestions))  # Remove duplicates while preserving order
        top_suggestions = unique_suggestions[:3]
        
        # Generate feedback prompt for regeneration
        feedback = "Improve the response by:\n" + "\n".join([f"- {s}" for s in top_suggestions])

        # More strict is_complete criteria - require 90+ for beginning score
        return {
            "quality_score": overall_score,
            "is_complete": evaluation.get("completeness", {}).get("score", 0) >= 85 and beginning_score >= 90,
            "feedback": feedback,
            "is_acceptable": is_acceptable,
            "raw_evaluation": evaluation  # Include raw evaluation for debugging
        }
        
    except Exception as e:
        logger.error(f"Error in response quality evaluation: {e}")
        # Fallback evaluation based on basic checks
        basic_score = 70.0  # Default score (reduced from 75 to be more conservative)
        
        # Adjust score based on basic checks
        if not ends_properly:
            basic_score -= 15.0  # Penalize for not ending properly
            
        if not starts_properly:
            basic_score -= 30.0  # Even more severe penalty for bad beginning (increased from 25)
            
        if not has_intro:
            basic_score -= 15.0  # Increased penalty for no clear introduction (from 10)
            
        if not has_conclusion:
            basic_score -= 10.0  # Penalize for no conclusion
        
        # Apply bonus for preferred beginnings
        if has_preferred_beginning:
            basic_score = min(100, basic_score + beginning_bonus)
            
        # Check for starting with code block
        if starts_with_code:
            basic_score -= 20.0  # Increased penalty for starting with code (from 15)
        
        is_acceptable = basic_score >= 88.0
        feedback = "The response MUST have a proper beginning and not start mid-sentence. Start with one of these phrases: 'In Bitcoin...', 'To answer your question about...', 'The key to understanding...', or 'Let me explain...'"
        
        return {
            "quality_score": basic_score,
            "is_complete": ends_properly and has_conclusion and starts_properly,
            "feedback": feedback,
            "is_acceptable": is_acceptable
        }


def validate_response_start(response_text: str) -> str:
    """
    Validates that the response has a proper beginning and fixes it if needed.
    
    Args:
        response_text: The generated response to validate
        
    Returns:
        A validated response with proper beginning
    """
    # Skip validation if response is too short
    if not response_text or len(response_text) < 20:
        return response_text
    
    # Trim any leading whitespace
    response_text = response_text.strip()
        
    # Check if response starts with one of our preferred beginnings
    preferred_beginnings = [
        "in bitcoin", 
        "to answer your question about", 
        "the key to understanding", 
        "let me explain"
    ]
    
    # If the response already starts with one of our preferred beginnings, return it unchanged
    for phrase in preferred_beginnings:
        if response_text.lower().startswith(phrase):
            return response_text
    
    # Additional check - some responses might start properly with "Bitcoin" or a complete sentence
    if response_text.startswith("Bitcoin") and len(response_text) > 10:
        first_period = response_text.find('.')
        if first_period > 10 and first_period < 100:  # First sentence of reasonable length
            # This starts with "Bitcoin" and has a proper first sentence, so it's likely fine
            return response_text
    
    # Check for lowercase beginning or starting with conjunction/pronoun
    starts_with_lowercase = response_text[0].islower() if response_text else False
    
    # Check if it starts with common pronouns or conjunctions
    bad_starts = [
        'it ', 'they ', 'their ', 'these ', 'those ', 'this ', 'that ', 
        'and ', 'or ', 'but ', 'so ', 'because ', 'since ', 'although ',
        'which ', 'where ', 'when ', 'how ', 'why ', 'what ', 'who ',
        'additionally', 'furthermore', 'moreover', 'therefore', 'thus',
        'however', 'nevertheless', 'also', 'yet', 'then', 'now'
    ]
    
    # Even stronger check for starting with a word fragment or anything that doesn't make sense
    starts_with_fragment = False
    if response_text:
        first_word = response_text.split()[0] if response_text.split() else ""
        # Check if first word is very short and not a common valid short word
        valid_short_words = ['a', 'an', 'the', 'in', 'on', 'at', 'by', 'for', 'to', 'i', 'we', 'you']
        if len(first_word) < 3 and first_word.lower() not in valid_short_words:
            starts_with_fragment = True
    
    starts_with_bad_word = any(response_text.lower().startswith(word) for word in bad_starts)
    
    # Check if the text might be truncated/corrupted
    has_weird_chars = False
    if response_text:
        # Check for weird Unicode characters that might indicate corruption
        weird_chars = ['۲', '₂', '…', '•', '¶', '◀', '►', '■', '□', '▪', '▫', '▬']
        has_weird_chars = any(char in response_text[:50] for char in weird_chars)
    
    # Check for a sentence fragment at the beginning (no verb in first phrase)
    first_fragment = response_text.split('.')[0] if '.' in response_text else response_text[:100]
    looks_like_fragment = len(first_fragment) < 20 and ' ' in first_fragment
    
    # If response passes ALL our checks, return it unchanged
    if (not starts_with_lowercase and 
        not starts_with_bad_word and 
        not starts_with_fragment and 
        not has_weird_chars and 
        not looks_like_fragment):
        return response_text
        
    # If we reach here, the response needs fixing
    # Extract the topic from the beginning or the content
    
    # First, try to find what the topic is from the first paragraph
    bitcoin_terms = ['bitcoin', 'blockchain', 'crypto', 'mining', 'wallet', 'transaction', 
                    'address', 'block', 'node', 'consensus', 'proof-of-work', 'merkle', 
                    'signature', 'hash', 'key', 'script', 'utxo', 'lightning', 'segwit']
    
    # Find relevant bitcoin term in first paragraph to use in introduction
    first_para = response_text.split('\n\n')[0] if '\n\n' in response_text else response_text[:300]
    found_term = None
    
    # More sophisticated topic extraction
    # Try to find a topic in the first paragraph - look for technical terms
    for term in bitcoin_terms:
        if term in first_para.lower():
            # Check if the term is a meaningful part of the content, not just a passing reference
            if first_para.lower().count(term) > 1 or term in response_text.lower()[:100]:
                found_term = term
                break
    
    # If no term found, look for P2* terms (P2SH, P2PKH, etc.) which are common in Bitcoin
    if not found_term and any(p in first_para.upper() for p in ['P2SH', 'P2PKH', 'P2WPKH', 'P2WSH']):
        for p in ['P2SH', 'P2PKH', 'P2WPKH', 'P2WSH']:
            if p in first_para.upper():
                found_term = p
                break
    
    # Still no term? Try to find any capitalized terms that might be important
    if not found_term:
        # Look for capitalized words that might be technical terms
        words = first_para.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2 and word.lower() not in ['the', 'this', 'that', 'these', 'those']:
                found_term = word
                break
    
    # Default term if none found
    if not found_term:
        # Try to identify what the content is about from keywords
        if 'transaction' in response_text.lower() or 'script' in response_text.lower():
            found_term = "transactions"
        elif 'address' in response_text.lower() or 'key' in response_text.lower():
            found_term = "addresses"
        elif 'block' in response_text.lower() or 'chain' in response_text.lower():
            found_term = "blockchain"
        else:
            found_term = "Bitcoin"
    
    # Format the term appropriately
    if found_term.upper() in ['P2SH', 'P2PKH', 'P2WPKH', 'P2WSH']:
        # Keep P2* terms in their original casing
        formatted_term = found_term
    else:
        # For normal terms, ensure proper casing
        formatted_term = found_term.lower()
    
    # Create a proper introduction
    # Try different templates based on the content to make it feel more natural
    
    # If it's about P2SH or similar address types
    if formatted_term.upper() in ['P2SH', 'P2PKH', 'P2WPKH', 'P2WSH']:
        fixed_response = f"In Bitcoin, {formatted_term} (Pay-to-Script-Hash) is a transaction type that allows sending bitcoins to a script hash instead of a public key hash. {response_text}"
    
    # If it's about transactions
    elif formatted_term.lower() in ['transaction', 'transactions']:
        fixed_response = f"In Bitcoin, transactions are the fundamental operations that transfer value between addresses. {response_text}"
    
    # If it's about addresses
    elif formatted_term.lower() in ['address', 'addresses', 'key', 'keys', 'public key', 'private key']:
        fixed_response = f"In Bitcoin, addresses are identifiers derived from public keys that allow users to receive funds. {response_text}"
    
    # If it's about mining
    elif formatted_term.lower() in ['mining', 'miner', 'miners', 'hash', 'proof-of-work']:
        fixed_response = f"In Bitcoin, mining is the process by which new transactions are added to the blockchain through proof-of-work. {response_text}"
    
    # Default case
    else:
        fixed_response = f"In Bitcoin, {formatted_term} is a fundamental concept in the protocol's design. {response_text}"
    
    # Check if the first paragraph already contains our term to avoid redundancy
    if len(response_text) > 100:
        first_hundred = response_text[:100].lower()
        if formatted_term.lower() in first_hundred:
            # If the term is already prominently featured in the beginning, use a more generic intro
            fixed_response = f"Let me explain how {formatted_term} works in Bitcoin. {response_text}"
    
    # Log that we fixed the response
    logger.warning(f"Fixed response beginning by adding proper introduction for topic: {formatted_term}")
    
    return fixed_response 