from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import json
import os
from typing import List, Optional
from utils import load_plant_catalog, search_plants_by_text, get_plant_by_id

app = FastAPI(title="Plant Care Assistant API", version="1.0.0")

# Add CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama client
client = ollama.Client()

# Load plant catalog on startup
plant_catalog = load_plant_catalog()


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None


class RecommendRequest(BaseModel):
    query: str
    plant_type: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: Optional[List[str]] = None


class PlantInfo(BaseModel):
    id: int
    name: str
    description: str
    care_instructions: dict
    tags: List[str]
    confidence: Optional[float] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Plant Care Assistant API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    Chat endpoint for general plant care questions using Ollama/LLaMA
    """
    try:
        # Create a plant care focused prompt
        system_prompt = """You are a helpful plant care assistant. You provide expert advice on:
        - Plant care (watering, light, fertilizer, soil)
        - Common plant problems (yellowing leaves, pests, diseases)
        - Plant recommendations for different conditions
        - General gardening tips

        Keep your responses helpful, concise, and practical."""

        full_prompt = f"{system_prompt}\n\nUser question: {request.message}"
        if request.context:
            full_prompt += f"\nContext: {request.context}"

        response = client.generate(model="llama2", prompt=full_prompt)

        # Generate some suggestions based on the query
        suggestions = generate_suggestions(request.message)

        return ChatResponse(
            response=response['response'],
            suggestions=suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/recommend", response_model=List[PlantInfo])
async def get_recommendations(request: RecommendRequest):
    """
    Text-based plant recommendations (will use SBERT + FAISS later)
    For now, uses simple text matching
    """
    try:
        # For now, use simple text search until we implement SBERT + FAISS
        matching_plants = search_plants_by_text(plant_catalog, request.query, request.plant_type)

        return [
            PlantInfo(
                id=plant["id"],
                name=plant["name"],
                description=plant["description"],
                care_instructions=plant["care_instructions"],
                tags=plant["tags"],
                confidence=plant.get("confidence", 0.8)  # Placeholder confidence
            )
            for plant in matching_plants[:5]  # Return top 5 matches
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@app.post("/search-by-image", response_model=List[PlantInfo])
async def search_by_image(file: UploadFile = File(...)):
    """
    Image-based plant identification (will use CLIP + FAISS later)
    For now, returns a placeholder response
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # For now, return a placeholder response
        # Later, this will use CLIP to analyze the image
        placeholder_results = [
            PlantInfo(
                id=1,
                name="Image Analysis Placeholder",
                description="CLIP + FAISS image analysis will be implemented here",
                care_instructions={"note": "Image processing coming soon"},
                tags=["placeholder"],
                confidence=0.0
            )
        ]

        return placeholder_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search error: {str(e)}")


@app.get("/plants/{plant_id}", response_model=PlantInfo)
async def get_plant_details(plant_id: int):
    """Get detailed information about a specific plant"""
    try:
        plant = get_plant_by_id(plant_catalog, plant_id)
        if not plant:
            raise HTTPException(status_code=404, detail="Plant not found")

        return PlantInfo(
            id=plant["id"],
            name=plant["name"],
            description=plant["description"],
            care_instructions=plant["care_instructions"],
            tags=plant["tags"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving plant: {str(e)}")


@app.get("/plants", response_model=List[PlantInfo])
async def list_all_plants():
    """List all plants in the catalog"""
    try:
        return [
            PlantInfo(
                id=plant["id"],
                name=plant["name"],
                description=plant["description"],
                care_instructions=plant["care_instructions"],
                tags=plant["tags"]
            )
            for plant in plant_catalog
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing plants: {str(e)}")


def generate_suggestions(message: str) -> List[str]:
    """Generate helpful suggestions based on user message"""
    message_lower = message.lower()
    suggestions = []

    if "water" in message_lower:
        suggestions.extend(["How often should I water my plants?", "Signs of overwatering"])
    if "yellow" in message_lower or "brown" in message_lower:
        suggestions.extend(["Common causes of leaf discoloration", "How to diagnose plant problems"])
    if "light" in message_lower:
        suggestions.extend(["Best plants for low light", "How much light do plants need?"])
    if "fertilizer" in message_lower:
        suggestions.extend(["When to fertilize plants", "Organic vs synthetic fertilizers"])

    return suggestions[:3]  # Return max 3 suggestions


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)