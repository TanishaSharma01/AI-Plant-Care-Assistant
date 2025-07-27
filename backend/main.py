import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Dict, Any

# Try to import our utilities with error handling
try:
    from ollama_utils import (
        llama_chat, llama_diagnose_problem, llama_recommend_plants,
        llama_seasonal_advice, test_ollama_connection, get_available_models
    )

    OLLAMA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Ollama utils not available: {e}")
    OLLAMA_AVAILABLE = False

try:
    from utils import (
        search_plants_by_text_embeddings, search_plants_by_image,
        recommend_plants_smart, diagnose_plant_problem_smart,
        get_plant_by_id, filter_plants_by_care_level,
        filter_plants_by_toxicity, filter_plants_by_location,
        get_plants_with_benefit, get_seasonal_care_advice,
        get_plant_safety_info, plant_catalog, get_system_status
    )

    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Utils not available: {e}")
    UTILS_AVAILABLE = False
    plant_catalog = []

app = FastAPI(title="Plant Care Assistant API", version="2.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to serve static files if they exist
try:
    if Path("images").exists():
        app.mount("/images", StaticFiles(directory="images"), name="images")
except Exception as e:
    print(f"⚠️ Warning: Could not mount images directory: {e}")


# Pydantic models
class ChatInput(BaseModel):
    message: str
    context: Optional[str] = None


class RecommendInput(BaseModel):
    message: str
    care_level: Optional[str] = None
    pet_safe: Optional[bool] = None
    location: Optional[str] = None


class DiagnoseInput(BaseModel):
    symptoms: str
    plant_name: Optional[str] = None
    additional_context: Optional[str] = None


class SeasonalAdviceInput(BaseModel):
    season: Optional[str] = None
    plants: Optional[str] = None


# Basic endpoints
@app.get("/")
async def index():
    """Serve the main page or API info"""
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        return {
            "message": "Plant Care Assistant API",
            "version": "2.0.0",
            "status": "running",
            "features": {
                "ollama": OLLAMA_AVAILABLE,
                "utils": UTILS_AVAILABLE,
                "plants_loaded": len(plant_catalog) if UTILS_AVAILABLE else 0
            }
        }


@app.get("/health")
async def health():
    """Enhanced health check with dependency status"""
    base_status = {
        "status": "ok",
        "api_version": "2.0.0",
        "plants_loaded": len(plant_catalog) if UTILS_AVAILABLE else 0,
    }

    # Add Ollama status if available
    if OLLAMA_AVAILABLE:
        try:
            ollama_status = test_ollama_connection()
            base_status["ollama_status"] = ollama_status["status"]
            base_status["ollama_model"] = ollama_status.get("model", "unknown")
        except Exception as e:
            base_status["ollama_status"] = "error"
            base_status["ollama_error"] = str(e)
    else:
        base_status["ollama_status"] = "not_available"

    # Add ML components status if available
    if UTILS_AVAILABLE:
        try:
            ml_status = get_system_status()
            base_status["ml_components"] = ml_status
        except Exception as e:
            base_status["ml_error"] = str(e)

    base_status["features"] = {
        "text_embeddings": UTILS_AVAILABLE,
        "image_search": UTILS_AVAILABLE,
        "ai_chat": OLLAMA_AVAILABLE,
        "plant_diagnosis": UTILS_AVAILABLE,
        "smart_recommendations": UTILS_AVAILABLE
    }

    return base_status


# Chat endpoint
@app.post("/chat")
async def chat_endpoint(input: ChatInput):
    """Chat endpoint with fallback"""
    if not OLLAMA_AVAILABLE:
        return {
            "response": "AI chat is currently unavailable. Please install and configure Ollama.",
            "suggestions": ["Check installation", "Configure Ollama", "Try basic search"],
            "error": "ollama_not_available"
        }

    try:
        ai_response = llama_chat(input.message, input.context)
        suggestions = generate_basic_suggestions(input.message)
        safety_warning = check_basic_safety(input.message)

        response = {
            "response": ai_response,
            "suggestions": suggestions
        }

        if safety_warning:
            response["safety_warning"] = safety_warning

        return response

    except Exception as e:
        return {"error": f"Chat error: {str(e)}"}


# Plant recommendations
@app.post("/recommend")
async def recommend_endpoint(input: RecommendInput):
    """Plant recommendations with fallback"""
    if not UTILS_AVAILABLE:
        return {
            "recommendations": [],
            "ai_advice": "Plant recommendations currently unavailable due to missing dependencies.",
            "search_method": "unavailable",
            "error": "utils_not_available"
        }

    try:
        plant_matches = recommend_plants_smart(input.message, k=5)

        # Apply filters if available
        if input.care_level or input.pet_safe is not None or input.location:
            filtered_matches = []
            for plant in plant_matches:
                if input.care_level and plant.get("care_level", "").lower() != input.care_level.lower():
                    continue
                if input.pet_safe is not None:
                    toxicity = plant.get("toxicity", {})
                    is_pet_safe = "non-toxic" in toxicity.get("pets", "").lower()
                    if input.pet_safe != is_pet_safe:
                        continue
                if input.location:
                    if not any(input.location.lower() in loc.lower() for loc in plant.get("best_location", [])):
                        continue
                filtered_matches.append(plant)
            plant_matches = filtered_matches

        # Get AI advice if available
        ai_advice = "Basic recommendations provided."
        if OLLAMA_AVAILABLE:
            try:
                recommendation_context = f"User query: {input.message}"
                if plant_matches:
                    recommendation_context += f"\nTop plant matches: {', '.join([p['name'] for p in plant_matches[:3]])}"
                ai_advice = llama_recommend_plants(recommendation_context)
            except Exception as e:
                ai_advice = f"AI advice unavailable: {str(e)}"

        return {
            "recommendations": plant_matches,
            "ai_advice": ai_advice,
            "search_method": plant_matches[0].get("search_method", "unknown") if plant_matches else "none"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


# Plant diagnosis
@app.post("/diagnose")
async def diagnose_endpoint(input: DiagnoseInput):
    """Plant diagnosis with fallback"""
    if not UTILS_AVAILABLE:
        return {
            "diagnosis": [],
            "ai_analysis": "Plant diagnosis currently unavailable due to missing dependencies.",
            "recommendations": [],
            "error": "utils_not_available"
        }

    try:
        diagnosis_results = diagnose_plant_problem_smart(input.symptoms, input.plant_name)

        formatted_diagnosis = []
        for plant, issues in diagnosis_results[:5]:
            formatted_diagnosis.append({
                "plant": {
                    "id": plant["id"],
                    "name": plant["name"],
                    "care_level": plant.get("care_level", "unknown")
                },
                "matching_issues": [
                    {
                        "problem": issue["problem"],
                        "causes": issue["causes"],
                        "solutions": issue["solutions"]
                    }
                    for issue in issues
                ],
                "confidence": plant.get("confidence", 0.5)
            })

        # Get AI analysis if available
        ai_analysis = "Basic diagnosis provided based on plant database."
        if OLLAMA_AVAILABLE:
            try:
                ai_analysis = llama_diagnose_problem(input.symptoms, input.plant_name, input.additional_context)
            except Exception as e:
                ai_analysis = f"AI analysis unavailable: {str(e)}"

        recommendations = []
        if diagnosis_results:
            for plant, issues in diagnosis_results[:2]:
                for issue in issues[:1]:
                    recommendations.extend(issue.get("solutions", []))

        return {
            "diagnosis": formatted_diagnosis,
            "ai_analysis": ai_analysis,
            "recommendations": list(set(recommendations))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis error: {str(e)}")


# Image search
@app.post("/search-by-image")
async def image_search_endpoint(file: UploadFile = File(...)):
    """Image search with fallback"""
    if not UTILS_AVAILABLE:
        return {
            "results": [{
                "id": 0,
                "name": "Image Search Unavailable",
                "description": "Image search requires ML dependencies to be installed.",
                "confidence": 0.0,
                "tags": ["unavailable"]
            }],
            "confidence_scores": [0.0],
            "search_method": "unavailable",
            "error": "utils_not_available"
        }

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        content = await file.read()
        image_buffer = io.BytesIO(content)

        results = search_plants_by_image(image_buffer, k=5)
        confidence_scores = [result.get("confidence", 0.0) for result in results]

        return {
            "results": results,
            "confidence_scores": confidence_scores,
            "search_method": results[0].get("search_method", "unknown") if results else "none"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search error: {str(e)}")


# Plant catalog endpoints
@app.get("/plants")
async def list_plants(
        care_level: Optional[str] = Query(None),
        pet_safe: Optional[bool] = Query(None),
        location: Optional[str] = Query(None),
        limit: int = Query(20, le=100)
):
    """List plants with basic filtering"""
    if not UTILS_AVAILABLE:
        return {
            "plants": [],
            "total_count": 0,
            "error": "Plant catalog not available"
        }

    try:
        filtered_plants = plant_catalog

        if care_level:
            filtered_plants = filter_plants_by_care_level(filtered_plants, care_level)
        if pet_safe is not None:
            filtered_plants = filter_plants_by_toxicity(filtered_plants, pet_safe)
        if location:
            filtered_plants = filter_plants_by_location(filtered_plants, location)

        return {
            "plants": filtered_plants[:limit],
            "total_count": len(filtered_plants),
            "filters_applied": {
                "care_level": care_level,
                "pet_safe": pet_safe,
                "location": location
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing plants: {str(e)}")


@app.get("/plants/{plant_id}")
async def get_plant_details(plant_id: int):
    """Get detailed plant information"""
    if not UTILS_AVAILABLE:
        return {"error": "Plant catalog not available"}

    try:
        plant = get_plant_by_id(plant_catalog, plant_id)
        if not plant:
            raise HTTPException(status_code=404, detail="Plant not found")

        safety_info = get_plant_safety_info(plant)

        return {
            "plant": plant,
            "safety_info": safety_info
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving plant: {str(e)}")


# # Utility endpoints
# @app.get("/models")
# async def list_ollama_models():
#     """List available Ollama models"""
#     if not OLLAMA_AVAILABLE:
#         return {"error": "Ollama not available", "available_models": []}
#
#     try:
#         models = get_available_models()
#         return {"available_models": models}
#     except Exception as e:
#         return {"error": f"Could not fetch models: {str(e)}", "available_models": []}
#

@app.get("/status")
async def get_detailed_status():
    """Get detailed system status"""
    status = {
        "api": "running",
        "version": "2.0.0",
        "dependencies": {
            "ollama_utils": OLLAMA_AVAILABLE,
            "utils": UTILS_AVAILABLE
        }
    }

    if UTILS_AVAILABLE:
        try:
            ml_status = get_system_status()
            status["ml_components"] = ml_status
        except Exception as e:
            status["ml_error"] = str(e)

    if OLLAMA_AVAILABLE:
        try:
            ollama_status = test_ollama_connection()
            status["ollama"] = ollama_status
        except Exception as e:
            status["ollama_error"] = str(e)

    return status


# Helper functions
def generate_basic_suggestions(message: str) -> List[str]:
    """Generate basic suggestions"""
    message_lower = message.lower()
    suggestions = []

    if any(word in message_lower for word in ["yellow", "brown", "wilting"]):
        suggestions.extend(["Check watering", "Review light conditions", "Inspect for pests"])
    elif any(word in message_lower for word in ["recommend", "suggest"]):
        suggestions.extend(["Beginner plants", "Pet-safe options", "Low light plants"])
    elif any(word in message_lower for word in ["water", "watering"]):
        suggestions.extend(["Watering guide", "Soil moisture check", "Drainage tips"])

    return suggestions[:3]


def check_basic_safety(message: str) -> Optional[str]:
    """Basic safety check"""
    message_lower = message.lower()

    if any(word in message_lower for word in ["pet", "cat", "dog", "child"]):
        return "Please check plant toxicity information if you have pets or small children."

    return None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)