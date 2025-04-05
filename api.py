import os
import json
import logging
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Import utility modules
from api_utils.cost import calculate_token_cost
from api_utils.validation import evaluate_response_quality, validate_response_start
from api_utils.persona import generate_satoshi_persona, get_satoshi_persona, get_satoshi_system_prompt
from api_utils.models import load_model_and_tokenizer, initialize_openai_client
from api_utils.prompts import get_response_regeneration_prompt, get_image_prompt, get_llm_prompt_for_image, get_bitcoin_recommendations_prompt, get_openai_enhanced_response_prompt
from api_utils.openai_helpers import call_openai, call_openai_with_fallback, extract_json_from_text
from api_utils.fallback import handle_response_truncation, process_fallback_response, generate_fallback_recommendations
from api_utils.image import extract_core_description, create_text_image, download_image

# Load environment variables
load_dotenv()

# Configure logging to only show errors
logging.basicConfig(
    level=logging.ERROR,  # Only show ERROR level logs and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

# Set logging level for all third-party libraries to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Get logger for this module - will also use ERROR level from the basicConfig
logger = logging.getLogger(__name__)

# Global variables to cache the model and tokenizer
MODEL = None
TOKENIZER = None

# Initialize OpenAI client
openai_client = None

# Cache for Satoshi persona description
SATOSHI_PERSONA = None

app = FastAPI(
    title="Satoshi AI Model API",
    description="A FastAPI endpoint to query your fine-tuned Satoshi AI model with LoRA adapter.",
    version="1.0.0",
)

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class GenerationRequest(BaseModel):
    messages: List[Message]
    output_type: Literal["text", "image"] = "text"
    temperature: float = 0.5
    max_tokens: int = 800
    
    # Allow extra fields to be ignored (for backward compatibility)
    class Config:
        extra = "ignore"

class GenerationResponse(BaseModel):
    type: Literal["text", "image"]
    text: str
    url: Optional[str] = None
    recommendations: Optional[List[str]] = None  # Add a field for recommendations
    token_cost: Optional[int] = None  # Changed from float to int

@app.on_event("startup")
def startup_event():
    """Load the base model and LoRA adapter, along with the tokenizer and initialize clients."""
    global MODEL, TOKENIZER, openai_client, SATOSHI_PERSONA
    try:
        # Get Hugging Face token from environment
        hf_token = os.environ.get("HF_TOKEN")
        
        # Load the model and tokenizer
        MODEL, TOKENIZER = load_model_and_tokenizer(hf_token)
        
        # Initialize OpenAI client
        openai_client = initialize_openai_client()
        
        # Generate and cache Satoshi's persona description
        SATOSHI_PERSONA = generate_satoshi_persona(MODEL, TOKENIZER)
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Error during API startup: {e}", exc_info=True)
        raise e

@app.post("/generate")
def generate_response(request: GenerationRequest):
    """
    POST endpoint that generates a response based on the provided messages.
    If output_type is 'text', returns text response.
    If output_type is 'image', renders the text onto an image and returns that image.
    """
    global MODEL, TOKENIZER, openai_client
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=500, detail="Model is not loaded yet.")
    
    if openai_client is None:
        raise HTTPException(status_code=500, detail="OpenAI client is not initialized. Please set OPENAI_API_KEY.")

    # Track token usage
    llm_input_tokens = 0
    llm_output_tokens = 0
    openai_usage = None
    is_image = request.output_type.lower() == "image"

    try:
        # Extract system message and user message from the messages array
        system_message = next((msg.content for msg in request.messages if msg.role == "system"), 
                             "You are Satoshi Nakamoto, the creator of Bitcoin.")
        
        # Get the last user message
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="At least one user message is required")
        user_message = user_messages[-1].content
        
        # Branch here based on the requested output type
        if request.output_type.lower() == "text":
            # HANDLE TEXT GENERATION FLOW
            
            # Enhance the system message with more specific guidance about Satoshi's style
            system_message = get_satoshi_system_prompt(system_message)
            
            # Format the prompt following the proper Llama 3.2 chat template
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            # Use the tokenizer's built-in chat template which properly formats for Llama 3.2
            prompt = TOKENIZER.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize the prompt
            inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
            llm_input_tokens = len(inputs["input_ids"][0])  # Track input tokens
            
            # LLaMA models typically don't use token_type_ids
            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")
                
            # Always limit fine-tuned model generation to 500 tokens
            max_fine_tuned_tokens = 500
            
            logger.debug(f"Generating initial response with fine-tuned model")
            
            # Generate response with fine-tuned model with appropriate parameters for 1B model
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=max_fine_tuned_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=TOKENIZER.eos_token_id,
            )
            
            # Track output tokens
            llm_output_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            
            # Decode the fine-tuned model output
            full_output = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response - DO NOT STRIP anything from the beginning
            fine_tuned_response = full_output[len(prompt):]
            
            # Remove any obvious artifacts or prefixes, but be careful to not strip valid content
            for prefix in ["Assistant:", "Satoshi Nakamoto:", "A:", "Satoshi:"]:
                if fine_tuned_response.startswith(prefix):
                    fine_tuned_response = fine_tuned_response[len(prefix):]
                    # Only strip whitespace at the beginning if it's just a single space
                    if fine_tuned_response.startswith(" "):
                        fine_tuned_response = fine_tuned_response[1:]
            
            # No need to generate this every time - use the cached version
            persona_description = get_satoshi_persona(SATOSHI_PERSONA, MODEL, TOKENIZER)
            
            # Now use GPT-4o to generate the final response
            logger.info(f"Enhancing response with GPT-4o")
            
            if openai_client is None:
                logger.warning("OpenAI client not available. Using fine-tuned model response directly")
                # Default recommendations when OpenAI client is not available
                default_recommendations = [
                    "Read the Bitcoin whitepaper to understand the original vision",
                    "Explore Bitcoin's source code on GitHub to see cryptographic principles in action",
                    "Implement a simple blockchain in your preferred programming language to grasp the core concepts"
                ]
                # Calculate token cost without OpenAI
                token_cost = calculate_token_cost(
                    llm_input_tokens=llm_input_tokens,
                    llm_output_tokens=llm_output_tokens,
                    is_image=is_image
                )
                return GenerationResponse(
                    type="text",
                    text=fine_tuned_response,
                    recommendations=default_recommendations,
                    token_cost=token_cost
                )
            
            try:
                # Ensure we respect the user's requested max_tokens precisely
                # If max_tokens is suspiciously large (>5000), cap it reasonably
                requested_tokens = request.max_tokens
                if requested_tokens > 5000:
                    logger.warning(
                        f"Requested {requested_tokens} tokens seems excessive, capping to 1000"
                    )
                    requested_tokens = 1000
                elif requested_tokens < 500:
                    logger.warning(
                        f"Requested {requested_tokens} tokens seems too small for complete responses, increasing to 500"
                    )
                    requested_tokens = 500
                    
                # Add a small buffer to ensure complete thoughts (will still be capped by the model)
                max_openai_tokens = requested_tokens + 100  # Add buffer for completion
                
                # Add completion buffer to the logger message
                logger.info(f"Using token limit of {max_openai_tokens} (includes completion buffer)")

                # Add a strict length instruction to the prompt with rich text formatting instructions for better frontend rendering
                openai_system_prompt = get_openai_enhanced_response_prompt(
                    user_message=user_message,
                    draft_response=fine_tuned_response,
                    max_tokens=max_openai_tokens
                )

                # Step 3: Call OpenAI API using our utility function with fallback
                result = call_openai_with_fallback(
                    client=openai_client,
                    model="gpt-4o-mini",
                    fallback_model="gpt-3.5-turbo", # Use a fallback model
                    messages=[
                        {"role": "system", "content": openai_system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7 if request.temperature > 0.7 else request.temperature,
                    max_tokens=max_openai_tokens,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                # Extract relevant information from the result
                if result.get("content"):
                    response_text = result["content"]
                    total_tokens = result.get("usage", {}).get("total_tokens", 0)
                    finish_reason = result.get("finish_reason", "unknown")
                    model_used = result.get("model_used", "unknown")  # Track which model was actually used
                    logger.info(f"Response generated using model: {model_used}")
                else:
                    raise Exception("No content in OpenAI response")
                
                # Log API call results
                logger.info(f"OpenAI API response received, tokens: {total_tokens}, finish_reason: {finish_reason}")
                
                # Handle potential truncation using our fallback utility
                if finish_reason == "length":
                    response_text = handle_response_truncation(response_text)
                
                # Validate and fix the response beginning if needed
                response_text = validate_response_start(response_text)
                
                # Implement feedback loop for quality improvement
                max_attempts = 3  # Maximum number of regeneration attempts
                current_attempt = 1
                
                while current_attempt <= max_attempts:
                    try:
                        logger.info(f"Attempting response quality evaluation (attempt {current_attempt}/{max_attempts})")
                        
                        # Evaluate the quality of the generated response
                        quality_result = evaluate_response_quality(
                            response_text=response_text,
                            user_query=user_message,
                            model_client=openai_client
                        )
                        
                        # Extract quality information
                        quality_score = quality_result.get("score", 0)
                        feedback = quality_result.get("feedback", "No specific feedback provided.")
                        
                        logger.info(f"Response quality assessment (attempt {current_attempt}/{max_attempts}): {quality_score:.1f}%")
                        logger.info(f"Feedback: {feedback}")
                        
                        # If quality is below threshold, try to regenerate
                        if quality_score < 88.0:
                            logger.warning(f"Response quality below threshold ({quality_score:.1f}% < 88%), regenerating with feedback")
                            logger.info(f"Quality feedback: {feedback}")
                            
                            # Create an enhanced prompt with the feedback using utility function
                            regeneration_system_prompt = get_response_regeneration_prompt(
                                user_message=user_message,
                                previous_response=response_text,
                                feedback=feedback,
                                max_tokens=max_openai_tokens
                            )
                          
                            try:
                                logger.info("Attempting regeneration with feedback")
                                
                                # Call OpenAI API for regeneration using our utility function with fallback
                                regeneration_result = call_openai_with_fallback(
                                    client=openai_client,
                                    model="gpt-4o-mini",  # Using consistent model for regeneration
                                    fallback_model="gpt-3.5-turbo",  # Fallback model for regeneration
                                    messages=[
                                        {"role": "system", "content": regeneration_system_prompt},
                                        {"role": "user", "content": user_message}
                                    ],
                                    temperature=0.5,  # Lower temperature for more focused regeneration
                                    max_tokens=max_openai_tokens,
                                    frequency_penalty=0.0,
                                    presence_penalty=0.0
                                )
                                
                                # Extract the regenerated response
                                if regeneration_result.get("content"):
                                    regenerated_response = regeneration_result["content"]
                                    regeneration_finish_reason = regeneration_result.get("finish_reason", "unknown")
                                    model_used = regeneration_result.get("model_used", "unknown")
                                    logger.info(f"Regeneration used model: {model_used}")
                                    
                                    # Handle potential truncation
                                    if regeneration_finish_reason == "length":
                                        regenerated_response = handle_response_truncation(regenerated_response)
                                        
                                    # Update the response text with the regenerated version
                                    response_text = regenerated_response
                                    logger.info(f"Successfully regenerated response (attempt {current_attempt})")
                                else:
                                    logger.warning(f"Regeneration failed to produce content (attempt {current_attempt})")
                            except Exception as regen_error:
                                logger.error(f"Error during response regeneration: {regen_error}")
                                
                        else:
                            logger.info(f"Response quality is good ({quality_score:.1f}% >= 88%), no need for regeneration")
                            break
                            
                    except Exception as eval_error:
                        logger.error(f"Error during quality evaluation: {eval_error}")
                        # Continue to next attempt or break if we're done
                    
                    current_attempt += 1
                
                # If response is still too long, use our utility function to handle truncation
                if len(response_text) > (max_openai_tokens * 6):  # Very generous character-to-token ratio
                    logger.warning(f"Response too long, truncating to ~{max_openai_tokens} tokens")
                    # Truncate to approximate max tokens first
                    truncated_response = response_text[:max_openai_tokens * 4]  # 4 chars per token approximation
                    # Then use our utility function to properly find sentence boundaries
                    response_text = handle_response_truncation(truncated_response, "length")
                
            except Exception as openai_error:
                logger.error(f"Error calling OpenAI API: {openai_error}")
                logger.info("Falling back to fine-tuned model response")
                
                # Process the fallback response using our utility function
                response_text = process_fallback_response(fine_tuned_response, request.max_tokens)
            
            # Generate recommendations for further learning (only for text responses)
            recommendations = generate_recommendations(openai_client, user_message, persona_description)
            
            # Calculate the total token cost
            token_cost = calculate_token_cost(
                llm_input_tokens=llm_input_tokens,
                llm_output_tokens=llm_output_tokens,
                openai_usage=openai_usage,
                is_image=is_image
            )
            
            # Validate and fix the response text if needed
            validated_response = validate_response_start(response_text)
            
            # Return the text response with recommendations and token cost
            return GenerationResponse(
                type="text",
                text=validated_response,
                recommendations=recommendations,
                token_cost=token_cost
            )
            
        elif request.output_type.lower() == "image":
            # HANDLE IMAGE GENERATION FLOW
            
            # Step 1: Generate an image description using our fine-tuned Satoshi model
            logger.info("Generating Bitcoin image description using fine-tuned model...")
            
            # Use our utility function to get the LLM prompt for image generation
            image_prompt = get_llm_prompt_for_image(user_message)
            
            # Format the prompt following the proper Llama 3.2 chat template
            image_messages = [
                {"role": "system", "content": "You are Satoshi Nakamoto, the creator of Bitcoin. Create a clear, educational image description."},
                {"role": "user", "content": image_prompt}
            ]
            
            # Use the tokenizer's built-in chat template which properly formats for Llama 3.2
            image_prompt_formatted = TOKENIZER.apply_chat_template(
                image_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize the prompt and track tokens
            image_inputs = TOKENIZER(image_prompt_formatted, return_tensors="pt").to(MODEL.device)
            llm_input_tokens = len(image_inputs["input_ids"][0])
            
            # Remove token_type_ids if present
            if "token_type_ids" in image_inputs:
                image_inputs.pop("token_type_ids")
                
            # Generate response for image description with parameters appropriate for 1B model
            image_outputs = MODEL.generate(
                **image_inputs,
                max_new_tokens=250,  # Shorter for image descriptions
                do_sample=True,
                temperature=0.7,  # Slightly higher for creativity
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=TOKENIZER.eos_token_id,
            )
            
            # Track output tokens
            llm_output_tokens = len(image_outputs[0]) - len(image_inputs["input_ids"][0])
            
            # Decode the output
            full_image_output = TOKENIZER.decode(image_outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response without stripping from the beginning
            llm_description = full_image_output[len(image_prompt_formatted):]
            
            # Remove any obvious artifacts or prefixes, but be careful not to remove valid content
            for prefix in ["Assistant:", "Satoshi Nakamoto:", "A:", "Satoshi:", "Image description:"]:
                if llm_description.startswith(prefix):
                    llm_description = llm_description[len(prefix):]
                    # Only remove a single space if it exists at the beginning
                    if llm_description.startswith(" "):
                        llm_description = llm_description[1:]
            
            # Extract the core description using our utility function
            core_description = extract_core_description(llm_description, max_length=200)
                
            # Get the formatted prompt for DALL-E using our utility function
            image_description = get_image_prompt(core_description)
            
            logger.debug(f"Generated image description ({len(image_description)} chars):")
            logger.debug(image_description[:150] + "..." if len(image_description) > 150 else image_description)
            
            try:
                # Generate the image using our helper function
                image_result = generate_image_from_query(openai_client, core_description)
                
                # Extract the results
                image_data = image_result["image_data"]
                temp_path = image_result["temp_path"]
                revised_prompt = image_result["revised_prompt"]
                
                # For DALL-E, we can't directly get token usage, so estimate based on prompt length
                openai_usage = {
                    "total_tokens": len(image_description) // 4  # Rough estimate of tokens from chars
                }
                
                # Calculate token cost
                token_cost = calculate_token_cost(
                    llm_input_tokens=llm_input_tokens,
                    llm_output_tokens=llm_output_tokens,
                    openai_usage=openai_usage,
                    is_image=is_image
                )
                
                # Return the image with metadata
                return StreamingResponse(
                    image_data,
                    media_type="image/png",
                    headers={
                        "X-Response": json.dumps(GenerationResponse(
                            type="image",
                            text=image_description[:500] + "..." if len(image_description) > 500 else image_description,
                            url=f"/temp/{os.path.basename(temp_path)}",  # Use the local path
                            token_cost=token_cost
                        ).dict())
                    }
                )
                
            except Exception as e:
                logger.error(f"Error generating image with DALL-E: {e}")
                # Fallback to the old text-on-image method
                logger.info("Falling back to text-on-image method")
                
                # For the fallback, we'll need a text response to render
                # Generate a simple response with our fine-tuned model
                fallback_outputs = MODEL.generate(
                    **image_inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    pad_token_id=TOKENIZER.eos_token_id,
                )
                
                # Track additional output tokens for fallback
                llm_output_tokens += len(fallback_outputs[0]) - len(image_inputs["input_ids"][0])
                
                # Update the fallback response extraction to avoid stripping from the beginning
                fallback_text = TOKENIZER.decode(fallback_outputs[0], skip_special_tokens=True)
                fallback_response = fallback_text[len(image_prompt_formatted):]
                
                # Process the fallback response
                fallback_response = process_fallback_response(fallback_response, max_tokens=300)
                
                # Calculate token cost without OpenAI
                token_cost = calculate_token_cost(
                    llm_input_tokens=llm_input_tokens,
                    llm_output_tokens=llm_output_tokens,
                    is_image=is_image
                )
                
                # Create a simple image with the text response using our utility
                image_data, temp_path = create_text_image(fallback_response)
                
                # Return with truncated text in header if needed
                header_text = fallback_response[:500] + "..." if len(fallback_response) > 500 else fallback_response
                
                return StreamingResponse(
                    image_data, 
                    media_type="image/png",
                    headers={
                        "X-Response": json.dumps(GenerationResponse(
                            type="image",
                            text=header_text,
                            url=f"/temp/{os.path.basename(temp_path)}",
                            token_cost=token_cost
                        ).dict())
                    }
                )
        else:
            raise HTTPException(status_code=400, detail="Invalid output type. Must be 'text' or 'image'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_full_response/{response_id}")
def get_full_response(response_id: str):
    """
    Endpoint to retrieve a full response by ID.
    This can be used when responses are too large for headers.
    """
    # Implement storage and retrieval logic here
    # For now, just a placeholder
    return {"message": "Full response retrieval not yet implemented"}

@app.post("/admin/refresh_persona")
def refresh_persona():
    """Admin endpoint to refresh the cached Satoshi persona."""
    global SATOSHI_PERSONA, MODEL, TOKENIZER
    
    try:
        SATOSHI_PERSONA = generate_satoshi_persona(MODEL, TOKENIZER)
        return {"status": "success", "message": "Satoshi persona refreshed", "length": len(SATOSHI_PERSONA)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh persona: {str(e)}")

@app.get("/admin/view_persona")
def view_persona():
    """Admin endpoint to view the cached Satoshi persona."""
    global SATOSHI_PERSONA
    
    if SATOSHI_PERSONA is None:
        raise HTTPException(status_code=404, detail="Satoshi persona not cached yet")
    
    return {"persona": SATOSHI_PERSONA, "length": len(SATOSHI_PERSONA)}

def generate_recommendations(openai_client, query, satoshi_persona):
    """Generate three recommendations for a user query."""
    logger.info("Generating recommendations")
    try:
        # Prepare the system prompt for recommendations using our utility function
        recommendation_system_prompt = get_bitcoin_recommendations_prompt(satoshi_persona)

        # Call OpenAI API using our utility function with fallback
        result = call_openai_with_fallback(
            client=openai_client,
            model="gpt-4o-mini",
            fallback_model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": recommendation_system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0.7,
            max_tokens=150,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Process the raw recommendations using our fallback utility
        if result.get("content"):
            recommendations_text = result["content"]
            recommendations = generate_fallback_recommendations(recommendations_text)
            model_used = result.get("model_used", "unknown")
            logger.info(f"Recommendations generated using model: {model_used}")
            return recommendations
        else:
            logger.error("No recommendations content in OpenAI response")
            return generate_fallback_recommendations("")
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return generate_fallback_recommendations("")

def generate_image_from_query(openai_client, query, model="gpt-4o-mini"):
    """Generate an image based on user query using the OpenAI API."""
    try:
        logger.info(f"Generating image for query: {query}")
        
        # Get the image prompt from our utility
        image_prompt = get_image_prompt(query)
        
        # Generate the image using OpenAI DALL-E 3
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        
        # Extract image URL and revised prompt
        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt
        
        logger.info("Image generated successfully")
        
        # Download the image using our utility
        image_data, temp_path = download_image(image_url)
        
        return {
            "image_data": image_data,
            "temp_path": temp_path,
            "prompt": image_prompt,
            "revised_prompt": revised_prompt
        }
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        
        # Create a fallback text image
        error_message = f"Failed to generate image: {str(e)}"
        image_data, temp_path = create_text_image(error_message)
        
        return {
            "image_data": image_data,
            "temp_path": temp_path,
            "prompt": "Error generating image",
            "revised_prompt": error_message
        }

if __name__ == "__main__":
    import uvicorn
    # Run the app on port 8000, accessible on localhost
    uvicorn.run(app, host="0.0.0.0", port=8000)

