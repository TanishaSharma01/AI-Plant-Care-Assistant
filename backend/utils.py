import os
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

# Try to import ML libraries with graceful fallbacks
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    print("Warning: PIL not available. Image processing disabled.")
    PIL_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Some ML features disabled.")
    TORCH_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel

    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: Transformers not available. CLIP features disabled.")
    CLIP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
except ImportError:
    print("Warning: SentenceTransformers not available. Embedding search disabled.")
    SBERT_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: FAISS not available. Vector search disabled.")
    FAISS_AVAILABLE = False

# Load plant catalog
catalog_path = os.path.join(os.path.dirname(__file__), "plant_catalog.json")
try:
    with open(catalog_path, "r", encoding='utf-8') as f:
        plant_catalog = json.load(f)
    print(f"✅ Loaded {len(plant_catalog)} plants from catalog")
except Exception as e:
    print(f"⚠️ Warning: Could not load plant catalog: {e}")
    plant_catalog = []

# Initialize ML components with error handling
sbert = None
index_text = None
clip_model = None
clip_processor = None
index_image = None
valid_image_indices = []

# Initialize SBERT if available
if SBERT_AVAILABLE and FAISS_AVAILABLE and plant_catalog:
    try:
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device}")

        sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

        # Create text representations
        plant_texts = []
        for plant in plant_catalog:
            text_parts = [
                plant.get("name", ""),
                plant.get("scientific_name", ""),
                " ".join(plant.get("common_names", [])),
                plant.get("description", ""),
                " ".join(plant.get("tags", [])),
                " ".join(plant.get("benefits", [])),
                " ".join(plant.get("best_location", [])),
                plant.get("care_level", ""),
                plant.get("origin", ""),
                plant.get("care_instructions", {}).get("watering", ""),
                plant.get("care_instructions", {}).get("light", ""),
                plant.get("care_instructions", {}).get("fertilizer", ""),
                " ".join([issue.get("problem", "") for issue in plant.get("common_issues", [])]),
                plant.get("toxicity", {}).get("pets", ""),
                plant.get("toxicity", {}).get("humans", "")
            ]
            combined_text = " ".join(filter(None, text_parts))
            plant_texts.append(combined_text)

        # Generate embeddings
        text_embeddings = sbert.encode(plant_texts, convert_to_numpy=True, normalize_embeddings=True)

        # Create FAISS index
        index_text = faiss.IndexFlatIP(text_embeddings.shape[1])
        index_text.add(text_embeddings)

        print(f"✅ Created text embeddings for {len(plant_texts)} plants")

    except Exception as e:
        print(f"⚠️ Warning: Could not initialize SBERT: {e}")
        sbert = None
        index_text = None

# Initialize CLIP if available
if CLIP_AVAILABLE and FAISS_AVAILABLE and PIL_AVAILABLE and plant_catalog:
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Pre-compute image embeddings if images exist
        image_embeddings = []
        valid_image_indices = []

        for i, plant in enumerate(plant_catalog):
            img_path = plant.get("image_path", "")
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    inputs = clip_processor(images=img, return_tensors="pt")
                    with torch.no_grad():
                        embed = clip_model.get_image_features(**inputs)
                    image_embeddings.append(embed[0].cpu().numpy())
                    valid_image_indices.append(i)
                except Exception as e:
                    print(f"⚠️ Warning: Could not process image for {plant.get('name', '')}: {e}")

        if image_embeddings:
            image_embeddings = np.vstack(image_embeddings)
            index_image = faiss.IndexFlatL2(image_embeddings.shape[1])
            index_image.add(image_embeddings)
            print(f"✅ Created image embeddings for {len(image_embeddings)} plant images")
        else:
            print("ℹ️ No valid plant images found for CLIP processing")

    except Exception as e:
        print(f"⚠️ Warning: Could not initialize CLIP: {e}")
        clip_model = None
        clip_processor = None


# ─── Text Search Functions ────────────────────────────────────────────────────

def search_plants_by_text_embeddings(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search plants using SBERT embeddings and FAISS"""
    if not sbert or not index_text:
        print("⚠️ SBERT not available, using fallback search")
        return search_plants_by_text_fallback(query, k)

    try:
        q_emb = sbert.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        similarities, indices = index_text.search(q_emb, min(k, len(plant_catalog)))

        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(plant_catalog):
                plant = plant_catalog[idx].copy()
                plant["confidence"] = float(similarity)
                plant["search_rank"] = i + 1
                plant["search_method"] = "embeddings"
                results.append(plant)

        return results
    except Exception as e:
        print(f"⚠️ Error in SBERT search: {e}")
        return search_plants_by_text_fallback(query, k)


def search_plants_by_text_fallback(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Fallback text search without embeddings"""
    query_lower = query.lower()
    results = []

    for plant in plant_catalog:
        score = 0

        # Search in name (highest priority)
        if query_lower in plant.get("name", "").lower():
            score += 15

        # Search in scientific name
        if query_lower in plant.get("scientific_name", "").lower():
            score += 12

        # Search in common names
        for common_name in plant.get("common_names", []):
            if query_lower in common_name.lower():
                score += 10
                break

        # Search in description
        if query_lower in plant.get("description", "").lower():
            score += 8

        # Search in benefits and tags
        for benefit in plant.get("benefits", []):
            if query_lower in benefit.lower():
                score += 6

        for tag in plant.get("tags", []):
            if query_lower in tag.lower():
                score += 5

        # Search in care instructions
        care_instructions = plant.get("care_instructions", {})
        for key, value in care_instructions.items():
            if isinstance(value, str) and query_lower in value.lower():
                score += 3

        if score > 0:
            plant_copy = plant.copy()
            plant_copy["confidence"] = min(score / 15.0, 1.0)
            plant_copy["search_method"] = "keyword"
            results.append(plant_copy)

    results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return results[:k]


def search_plants_by_image(image_bytes, k: int = 3) -> List[Dict[str, Any]]:
    """Search plants using CLIP embeddings"""
    if not clip_model or not clip_processor or not index_image or not PIL_AVAILABLE:
        print("⚠️ CLIP not available, returning placeholder")
        return get_placeholder_image_results()

    try:
        img = Image.open(image_bytes).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt")

        with torch.no_grad():
            q_emb = clip_model.get_image_features(**inputs)[0].cpu().numpy()

        distances, indices = index_image.search(q_emb.reshape(1, -1), min(k, len(valid_image_indices)))

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(valid_image_indices):
                plant_idx = valid_image_indices[idx]
                plant = plant_catalog[plant_idx].copy()

                similarity = 1.0 / (1.0 + distance)
                plant["confidence"] = float(similarity)
                plant["search_rank"] = i + 1
                plant["search_method"] = "clip"
                results.append(plant)

        return results
    except Exception as e:
        print(f"⚠️ Error in CLIP search: {e}")
        return get_placeholder_image_results()


def get_placeholder_image_results() -> List[Dict[str, Any]]:
    """Placeholder results when CLIP is not available"""
    return [{
        "id": 0,
        "name": "Image Search Unavailable",
        "description": "Image search requires CLIP model. Please check dependencies.",
        "confidence": 0.0,
        "tags": ["placeholder"],
        "search_method": "placeholder"
    }]


def recommend_plants_smart(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Smart recommendations with keyword matching fallback"""

    query_lower = query.lower()
    results = []

    print(f"🔍 Processing query: '{query}'")  # Debug log

    # Method 1: Check for specific patterns first

    # Pet-safe requests
    if any(word in query_lower for word in ["pet", "cat", "dog", "safe", "toxic", "non-toxic"]):
        print("🐕 Looking for pet-safe plants...")
        for plant in plant_catalog:
            toxicity = plant.get("toxicity", {}).get("pets", "").lower()
            if "non-toxic" in toxicity or "safe" in toxicity:
                plant_copy = plant.copy()
                plant_copy["confidence"] = 0.9
                plant_copy["search_method"] = "pet_safe_filter"
                results.append(plant_copy)
        if results:
            print(f"✅ Found {len(results)} pet-safe plants")
            return results[:k]

    # Beginner plants
    if any(word in query_lower for word in ["beginner", "easy", "simple", "first", "new"]):
        print("🌱 Looking for beginner plants...")
        for plant in plant_catalog:
            if plant.get("care_level", "").lower() == "beginner":
                plant_copy = plant.copy()
                plant_copy["confidence"] = 0.8
                plant_copy["search_method"] = "beginner_filter"
                results.append(plant_copy)
        if results:
            print(f"✅ Found {len(results)} beginner plants")
            return results[:k]

    # Low light plants
    if any(word in query_lower for word in ["low light", "dark", "shade", "dim"]):
        print("🌙 Looking for low light plants...")
        for plant in plant_catalog:
            tags = [t.lower() for t in plant.get("tags", [])]
            if "low light" in tags or any("low light" in tag for tag in tags):
                plant_copy = plant.copy()
                plant_copy["confidence"] = 0.8
                plant_copy["search_method"] = "low_light_filter"
                results.append(plant_copy)
        if results:
            print(f"✅ Found {len(results)} low light plants")
            return results[:k]

    # Indoor plants
    if any(word in query_lower for word in ["indoor", "house", "inside"]):
        print("🏠 Looking for indoor plants...")
        for plant in plant_catalog:
            tags = [t.lower() for t in plant.get("tags", [])]
            if "indoor" in tags:
                plant_copy = plant.copy()
                plant_copy["confidence"] = 0.7
                plant_copy["search_method"] = "indoor_filter"
                results.append(plant_copy)
        if results:
            print(f"✅ Found {len(results)} indoor plants")
            return results[:k]

    # Method 2: Use SBERT semantic search if available, otherwise keyword search
    if sbert and index_text:
        print("🔍 Using SBERT semantic search...")
        return search_plants_by_text_embeddings(query, k)
    else:
        print("🔍 Using general keyword search...")

    for plant in plant_catalog:
        score = 0

        # Search in name
        if query_lower in plant.get("name", "").lower():
            score += 10

        # Search in description
        if query_lower in plant.get("description", "").lower():
            score += 5

        # Search in tags
        for tag in plant.get("tags", []):
            if query_lower in tag.lower() or any(word in tag.lower() for word in query_lower.split()):
                score += 3

        # Search in benefits
        for benefit in plant.get("benefits", []):
            if query_lower in benefit.lower() or any(word in benefit.lower() for word in query_lower.split()):
                score += 3

        # Search in care level
        if query_lower in plant.get("care_level", "").lower():
            score += 8

        # Search in best locations
        for location in plant.get("best_location", []):
            if query_lower in location.lower():
                score += 4

        if score > 0:
            plant_copy = plant.copy()
            plant_copy["score"] = score
            plant_copy["confidence"] = min(score / 10.0, 1.0)
            plant_copy["search_method"] = "keyword_search"
            results.append(plant_copy)

    # Sort by score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    print(f"✅ Keyword search found {len(results)} results")

    # Method 3: If still no results, return some default beginner plants
    if not results:
        print("⚠️ No matches found, returning default beginner plants...")
        for plant in plant_catalog:
            if plant.get("care_level", "").lower() == "beginner":
                plant_copy = plant.copy()
                plant_copy["confidence"] = 0.5
                plant_copy["search_method"] = "default_beginner"
                results.append(plant_copy)

        # If no beginner plants, just return first few plants
        if not results:
            print("⚠️ No beginner plants found, returning first few plants...")
            for plant in plant_catalog[:3]:
                plant_copy = plant.copy()
                plant_copy["confidence"] = 0.3
                plant_copy["search_method"] = "fallback"
                results.append(plant_copy)

    return results[:k]


def diagnose_plant_problem_smart(symptoms: str, plant_name: Optional[str] = None) -> List[
    Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Plant problem diagnosis with fallback"""
    target_plants = plant_catalog

    if plant_name:
        target_plants = [p for p in plant_catalog if plant_name.lower() in p.get("name", "").lower()]
        if not target_plants:
            target_plants = plant_catalog

    diagnosis_results = []
    symptoms_lower = symptoms.lower()

    for plant in target_plants:
        matching_issues = []

        for issue in plant.get("common_issues", []):
            problem = issue.get("problem", "").lower()
            causes = [cause.lower() for cause in issue.get("causes", [])]

            if (symptoms_lower in problem or
                    any(symptoms_lower in cause for cause in causes) or
                    any(symptom.strip() in problem for symptom in symptoms_lower.split() if len(symptom.strip()) > 2)):
                matching_issues.append(issue)

        if matching_issues:
            diagnosis_results.append((plant, matching_issues))

    diagnosis_results.sort(key=lambda x: len(x[1]), reverse=True)
    return diagnosis_results


# ─── Compatibility Functions ──────────────────────────────────────────────────

def load_plant_catalog(file_path: str = "plant_catalog.json") -> List[Dict[str, Any]]:
    """Load the plant catalog (for compatibility)"""
    return plant_catalog


def search_plants_by_text(catalog: List[Dict[str, Any]], query: str, plant_type: Optional[str] = None,
                          care_level: Optional[str] = None, pet_safe: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Enhanced text search with filters"""
    results = search_plants_by_text_embeddings(query, k=10)

    # Apply filters
    if plant_type or care_level or pet_safe is not None:
        filtered_results = []
        for plant in results:
            if plant_type and plant_type.lower() not in [tag.lower() for tag in plant.get("tags", [])]:
                continue
            if care_level and plant.get("care_level", "").lower() != care_level.lower():
                continue
            if pet_safe is not None:
                toxicity = plant.get("toxicity", {})
                is_pet_safe = "non-toxic" in toxicity.get("pets", "").lower()
                if pet_safe != is_pet_safe:
                    continue
            filtered_results.append(plant)
        return filtered_results

    return results


def get_plant_by_id(catalog: List[Dict[str, Any]], plant_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific plant by its ID"""
    for plant in plant_catalog:
        if plant.get("id") == plant_id:
            return plant
    return None


def filter_plants_by_care_level(catalog: List[Dict[str, Any]], care_level: str) -> List[Dict[str, Any]]:
    """Filter plants by care level"""
    return [plant for plant in plant_catalog if plant.get("care_level", "").lower() == care_level.lower()]


def filter_plants_by_toxicity(catalog: List[Dict[str, Any]], pet_safe: bool = True) -> List[Dict[str, Any]]:
    """Filter plants by pet safety"""
    filtered_plants = []
    for plant in plant_catalog:
        toxicity = plant.get("toxicity", {})
        is_pet_safe = "non-toxic" in toxicity.get("pets", "").lower()
        if pet_safe == is_pet_safe:
            filtered_plants.append(plant)
    return filtered_plants


def filter_plants_by_location(catalog: List[Dict[str, Any]], location: str) -> List[Dict[str, Any]]:
    """Filter plants by best location"""
    location_lower = location.lower()
    return [plant for plant in plant_catalog
            if any(location_lower in loc.lower() for loc in plant.get("best_location", []))]


def get_plants_with_benefit(catalog: List[Dict[str, Any]], benefit: str) -> List[Dict[str, Any]]:
    """Get plants with specific benefit"""
    benefit_lower = benefit.lower()
    return [plant for plant in plant_catalog
            if any(benefit_lower in b.lower() for b in plant.get("benefits", []))]


def get_seasonal_care_advice(catalog: List[Dict[str, Any]], season: Optional[str] = None) -> Dict[
    str, List[Dict[str, Any]]]:
    """Get seasonal care advice"""
    if season is None:
        month = datetime.now().month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"

    season_lower = season.lower()
    advice = []

    for plant in plant_catalog:
        seasonal_care = plant.get("seasonal_care", {})
        if season_lower in seasonal_care:
            advice.append({
                "plant": plant,
                "advice": seasonal_care[season_lower]
            })

    return {season_lower: advice}


def get_plant_recommendations_by_criteria(catalog: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Get recommendations by multiple criteria"""
    return recommend_plants_smart(" ".join(str(v) for v in kwargs.values() if v), k=kwargs.get("limit", 5))


def get_plant_safety_info(plant: Dict[str, Any]) -> Dict[str, str]:
    """Extract safety information"""
    toxicity = plant.get("toxicity", {})
    return {
        "pet_safety": toxicity.get("pets", "Unknown"),
        "human_safety": toxicity.get("humans", "Unknown"),
        "care_level": plant.get("care_level", "Unknown"),
        "safety_notes": f"Pet safety: {toxicity.get('pets', 'Unknown')}. Human safety: {toxicity.get('humans', 'Unknown')}."
    }


def diagnose_plant_problem(catalog: List[Dict[str, Any]], symptoms: str) -> List[
    Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Diagnose plant problems (compatibility wrapper)"""
    return diagnose_plant_problem_smart(symptoms)


def get_system_status() -> Dict[str, Any]:
    """Get status of all ML components"""
    return {
        "PIL": PIL_AVAILABLE,
        "torch": TORCH_AVAILABLE,
        "transformers": CLIP_AVAILABLE,
        "sentence_transformers": SBERT_AVAILABLE,
        "faiss": FAISS_AVAILABLE,
        "sbert_loaded": sbert is not None,
        "clip_loaded": clip_model is not None,
        "text_index_ready": index_text is not None,
        "image_index_ready": index_image is not None,
        "plants_loaded": len(plant_catalog)
    }