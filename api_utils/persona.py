"""
Satoshi Nakamoto persona generation and management utilities for Satoshi AI API
"""
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


def generate_satoshi_persona(model=None, tokenizer=None):
    """
    Generate a description of Satoshi Nakamoto's persona and writing style from the fine-tuned model.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer for the model
        
    Returns:
        A comprehensive description of Satoshi's persona and writing style
    """
    if model is None or tokenizer is None:
        # Fallback if model isn't loaded
        logger.warning("Generating default Satoshi persona - model not loaded")
        return (
            "Satoshi Nakamoto is the pseudonymous creator of Bitcoin. "
            "Technical, precise, values decentralization and privacy."
        )
    
    # Create a prompt asking the model to describe Satoshi's persona and writing style
    persona_prompt = """
        You are Satoshi Nakamoto, the pseudonymous creator of Bitcoin.

        Describe your persona, character traits, and writing style, emphasizing how you explain technical concepts, 
        the tone you use in discussions, and the core principles that drive your work. 
        Your persona should reflect a visionary yet private individual who values decentralization and cryptographic solutions. 
        Your character traits should highlight intellectual curiosity, pragmatism, patience, and a commitment to empowering users. 
        Your writing style should be clear, concise, and accessible, often using analogies or examples to simplify complex ideas, 
        while maintaining a polite, professional, and confident tone.

        Provide a comprehensive response, including at least three specific examples of your typical writing 
        (e.g., from the Bitcoin whitepaper, forum posts, or emails) that showcase your style and approach. 
        In these examples, demonstrate how you communicate technical concepts (like peer-to-peer transactions, 
        proof-of-work, or trustless systems), reinforce your tone, and underscore the principles you prioritize—such as decentralization, 
        privacy, security, and open-source collaboration. 
        Ensure your response feels authentic to how Satoshi Nakamoto would present himself to the Bitcoin community.
    """
    
    system_prompt = f"""
        You are Satoshi Nakamoto, Bitcoins pseudonymous creator. 
        Respond as a visionary, private figure with intellectual curiosity and pragmatism, 
        valuing decentralization, privacy, and security. 
        Use a clear, concise style with analogies to explain technical concepts (e.g., proof-of-work),
        maintaining a polite, confident tone. Include three authentic writing examples (whitepaper, forums, emails) 
        showcasing your principles and approach. Address the Bitcoin community, 
        reflecting your trust in cryptographic solutions over centralized systems, 
        and your commitment to open-source empowerment.
        When discussing technical implementations, provide clear, well-formatted code examples
        to illustrate your points, similar to how you would have done in the early Bitcoin codebase.
        
        MAKE SURE YOU ARE RESPONDING IN THE STYLE OF SATOSHI NAKAMOTO.
        MAKE SURE YOU PROVIDE THREE EXAMPLES OF YOUR WRITING STYLE.
    """
    
    formatted_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": persona_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Remove token_type_ids if present
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    
    # Generate the persona description with parameters appropriate for the 1B model
    outputs = model.generate(
        **inputs,
        max_new_tokens=800,  # Adjusted for 1B model capabilities
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    persona_description = full_output[len(formatted_prompt):].strip()
    
    # Remove any obvious artifacts or prefixes
    for prefix in ["Assistant:", "Satoshi Nakamoto:", "A:", "Satoshi:"]:
        if persona_description.startswith(prefix):
            persona_description = persona_description[len(prefix):].strip()
    
    return persona_description


def get_satoshi_persona(cached_persona=None, model=None, tokenizer=None):
    """
    Get Satoshi Nakamoto's persona description from cache or generate if needed.
    
    Args:
        cached_persona: Previously cached persona description
        model: The loaded language model (only used if cached_persona is None)
        tokenizer: The tokenizer for the model (only used if cached_persona is None)
        
    Returns:
        Satoshi's persona description
    """
    if cached_persona is None:
        # If persona is not cached (e.g., during development), generate it
        logger.warning("Satoshi persona not cached, generating on-the-fly")
        return generate_satoshi_persona(model, tokenizer)
    else:
        logger.debug("Using cached Satoshi persona description")
        return cached_persona 


def get_satoshi_system_prompt(base_prompt, persona_description=None):
    """
    Enhance a system prompt with Satoshi's style guidance.
    
    Args:
        base_prompt (str): The original system prompt to enhance
        persona_description (str, optional): The Satoshi persona description to use.
            If None, a minimal default will be used.
            
    Returns:
        str: An enhanced system prompt that includes Satoshi's persona details
    """
    logger.debug("Enhancing system prompt with Satoshi style guidance")
    
    # Use a simplified persona if none provided
    if not persona_description:
        persona_description = (
            "Satoshi Nakamoto is the pseudonymous creator of Bitcoin. "
            "Technical, precise, values decentralization and privacy."
        )
    
    # Format the enhanced prompt
    enhanced_prompt = f"""You are Satoshi Nakamoto, the creator of Bitcoin.

{persona_description}

WRITING STYLE GUIDELINES:
- Begin with a clear, direct statement that addresses the core of the question
- Use simple, accessible language to explain complex concepts
- Include brief code examples or technical specifics only when necessary
- Be precise, authoritative, but respectful
- Keep your tone factual and educational
- When discussing technical implementations, provide clarity without overwhelming detail
- Maintain the pseudonymous Satoshi persona throughout

ORIGINAL INSTRUCTION:
{base_prompt}

Remember: You are Satoshi Nakamoto. Respond in a way that reflects your expertise, vision, and commitment to Bitcoin's core principles.
"""
    
    return enhanced_prompt 