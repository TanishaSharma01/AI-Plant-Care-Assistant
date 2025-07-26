import json
import os
from typing import List, Dict, Optional, Any


def load_plant_catalog(file_path: str = "plant_catalog.json") -> List[Dict[str, Any]]:
    """
    Load the plant catalog from JSON file
    """
    try:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, returning empty catalog")
            return []

        with open(file_path, 'r', encoding='utf-8') as file:
            catalog = json.load(file)
            print(f"Loaded {len(catalog)} plants from catalog")
            return catalog
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return []
    except Exception as e:
        print(f"Error loading plant catalog: {e}")
        return []


def search_plants_by_text(catalog: List[Dict[str, Any]], query: str, plant_type: Optional[str] = None) -> List[
    Dict[str, Any]]:
    """
    Simple text-based search through plant catalog
    This will be replaced with SBERT + FAISS later
    """
    if not catalog or not query:
        return []

    query_lower = query.lower()
    results = []

    for plant in catalog:
        score = 0

        # Search in name
        if query_lower in plant.get("name", "").lower():
            score += 10

        # Search in description
        if query_lower in plant.get("description", "").lower():
            score += 5

        # Search in tags
        for tag in plant.get("tags", []):
            if query_lower in tag.lower():
                score += 3

        # Search in care instructions
        care_instructions = plant.get("care_instructions", {})
        for key, value in care_instructions.items():
            if isinstance(value, str) and query_lower in value.lower():
                score += 2

        # Filter by plant type if specified
        if plant_type and plant_type.lower() not in [tag.lower() for tag in plant.get("tags", [])]:
            continue

        # Add to results if there's a match
        if score > 0:
            plant_copy = plant.copy()
            plant_copy["confidence"] = min(score / 10.0, 1.0)  # Normalize score to 0-1
            results.append(plant_copy)

    # Sort by confidence score (highest first)
    results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return results


def get_plant_by_id(catalog: List[Dict[str, Any]], plant_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific plant by its ID
    """
    for plant in catalog:
        if plant.get("id") == plant_id:
            return plant
    return None


def filter_plants_by_tags(catalog: List[Dict[str, Any]], tags: List[str]) -> List[Dict[str, Any]]:
    """
    Filter plants by specific tags
    """
    if not tags:
        return catalog

    filtered_plants = []
    tags_lower = [tag.lower() for tag in tags]

    for plant in catalog:
        plant_tags = [tag.lower() for tag in plant.get("tags", [])]
        if any(tag in plant_tags for tag in tags_lower):
            filtered_plants.append(plant)

    return filtered_plants


def get_care_tip_by_issue(issue: str) -> str:
    """
    Get care tips for common plant issues
    """
    tips = {
        "yellowing leaves": "Yellow leaves often indicate overwatering, underwatering, or nutrient deficiency. Check soil moisture and consider fertilizing.",
        "brown tips": "Brown leaf tips usually mean low humidity, fluoride in water, or overfertilizing. Try using filtered water and increasing humidity.",
        "dropping leaves": "Leaf drop can be caused by stress from changes in light, water, or temperature. Ensure consistent care routine.",
        "not growing": "Slow growth may indicate need for more light, nutrients, or larger pot. Check if plant is root-bound.",
        "pest problems": "Common pests include aphids, spider mites, and mealybugs. Use insecticidal soap or neem oil treatment.",
        "wilting": "Wilting usually indicates watering issues - either too much or too little. Check soil moisture level."
    }

    issue_lower = issue.lower()
    for key, tip in tips.items():
        if key in issue_lower:
            return tip

    return "For specific plant issues, consider factors like watering, light, humidity, and nutrients. If problems persist, consult a local garden center."


def validate_plant_data(plant: Dict[str, Any]) -> bool:
    """
    Validate that a plant entry has required fields
    """
    required_fields = ["id", "name", "description", "care_instructions", "tags"]

    for field in required_fields:
        if field not in plant:
            return False

    # Validate data types
    if not isinstance(plant["id"], int):
        return False
    if not isinstance(plant["name"], str):
        return False
    if not isinstance(plant["description"], str):
        return False
    if not isinstance(plant["care_instructions"], dict):
        return False
    if not isinstance(plant["tags"], list):
        return False

    return True


def add_plant_to_catalog(catalog: List[Dict[str, Any]], new_plant: Dict[str, Any]) -> bool:
    """
    Add a new plant to the catalog after validation
    """
    if not validate_plant_data(new_plant):
        return False

    # Check if ID already exists
    if any(plant.get("id") == new_plant["id"] for plant in catalog):
        return False

    catalog.append(new_plant)
    return True