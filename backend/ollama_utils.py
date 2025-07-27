import subprocess
import json
from typing import Optional, Dict, Any

# Default model - you can change this to any model you have installed
MODEL = "mistral"  # or "llama3.2:latest", "mistral", etc.


def llama_chat(user_message: str, context: Optional[str] = None) -> str:
    """
    Chat with Ollama for plant care assistance
    """
    # Enhanced system prompt for plant care
    system_prompt = """You are an expert plant care assistant and botanist. You provide:

- Detailed, practical plant care advice (watering, light, soil, fertilizer)
- Solutions for plant problems (yellowing leaves, pests, diseases, root rot)
- Plant recommendations based on user environment and skill level
- Safety information about plant toxicity to pets and humans
- Seasonal care guidance and timing for plant care activities
- Propagation and repotting advice

Keep your responses:
- Helpful and practical with specific actionable steps
- Safe and accurate, mentioning toxicity when relevant
- Encouraging for plant parents of all skill levels
- Concise but thorough enough to be useful

If asked about plant identification from descriptions, ask clarifying questions about leaf shape, size, growth pattern, flowers, etc."""

    # Build the prompt
    full_prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

    if context:
        full_prompt += f"Context: {context}\n\n"

    full_prompt += f"[INST] {user_message} [/INST]\n"

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, full_prompt],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Ollama CLI error:\n{result.stderr}")

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama request timed out after 60 seconds")
    except FileNotFoundError:
        raise RuntimeError("Ollama CLI not found. Please install Ollama and ensure it's in your PATH")
    except Exception as e:
        raise RuntimeError(f"Error calling Ollama: {str(e)}")


def llama_diagnose_problem(symptoms: str, plant_name: Optional[str] = None,
                           additional_context: Optional[str] = None) -> str:
    """
    Specialized function for plant problem diagnosis
    """
    diagnostic_prompt = f"""You are a plant doctor. A user is describing symptoms with their plant.

Symptoms: {symptoms}"""

    if plant_name:
        diagnostic_prompt += f"\nPlant: {plant_name}"

    if additional_context:
        diagnostic_prompt += f"\nAdditional context: {additional_context}"

    diagnostic_prompt += """

Please provide:
1. Most likely causes of these symptoms
2. Immediate steps to take
3. Prevention tips for the future
4. When to be concerned and seek additional help

Be specific and practical in your advice."""

    return llama_chat(diagnostic_prompt)


def llama_recommend_plants(requirements: str) -> str:
    """
    Specialized function for plant recommendations
    """
    recommendation_prompt = f"""A user is looking for plant recommendations. Here are their requirements:

{requirements}

Please recommend 3-5 specific plants that would be perfect for their situation. For each plant, include:
- Plant name
- Why it fits their needs
- Basic care requirements
- Any special considerations (pet safety, difficulty level, etc.)

Make your recommendations practical and consider the user's experience level."""

    return llama_chat(recommendation_prompt)


def llama_seasonal_advice(season: str, plants: Optional[str] = None) -> str:
    """
    Get seasonal plant care advice
    """
    seasonal_prompt = f"""Provide seasonal plant care advice for {season}."""

    if plants:
        seasonal_prompt += f" Focus on these plants: {plants}"

    seasonal_prompt += """

Include advice about:
- Watering adjustments
- Light requirements changes
- Fertilizing schedule
- Common seasonal issues to watch for
- Any seasonal care tasks (repotting, pruning, etc.)

Be specific about timing and practical steps."""

    return llama_chat(seasonal_prompt)


def test_ollama_connection() -> Dict[str, Any]:
    """
    Test if Ollama is working properly
    """
    try:
        # Test with a simple query
        response = llama_chat("Hello, are you working?")
        return {
            "status": "success",
            "model": MODEL,
            "response": response[:100] + "..." if len(response) > 100 else response
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": MODEL
        }


def get_available_models() -> list:
    """
    Get list of available Ollama models
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return []

        # Parse the output to extract model names
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        models = []
        for line in lines:
            if line.strip():
                model_name = line.split()[0]  # First column is model name
                models.append(model_name)

        return models
    except Exception:
        return []


def set_model(model_name: str) -> bool:
    """
    Change the active model
    """
    global MODEL
    try:
        # Test if the model works
        result = subprocess.run(
            ["ollama", "run", model_name, "test"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            MODEL = model_name
            return True
        else:
            return False
    except Exception:
        return False