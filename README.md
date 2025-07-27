# 🌱 AI Plant Care Assistant Backend

A sophisticated **AI-powered plant care assistant** that helps users identify plants, get care recommendations, and diagnose plant problems using cutting-edge machine learning technologies.

## ✨ Features

### 🧠 **AI-Powered Search & Recommendations**
- **Semantic Text Search** - Understands "plants that are impossible to kill" → finds drought-tolerant, low-maintenance plants
- **Image Recognition** - Upload plant photos for instant species identification using CLIP
- **Smart Recommendations** - Context-aware suggestions based on care level, pet safety, and location
- **Problem Diagnosis** - Describe symptoms to get targeted solutions and care advice

### 💬 **Intelligent Chat Interface**
- **Natural Language Processing** - Ask questions in plain English about plant care
- **Contextual Responses** - Powered by Ollama/LLaMA for expert plant advice
- **Safety Warnings** - Automatic alerts about plant toxicity for pets and children

### 📊 **Comprehensive Plant Database**
- **10 Detailed Plant Profiles** - From beginner-friendly to intermediate care levels
- **Rich Metadata** - Scientific names, care instructions, common issues, seasonal advice
- **Smart Filtering** - By care level, pet safety, location, and plant benefits

## 🚀 Tech Stack

- **FastAPI** - High-performance async API framework
- **SBERT + FAISS** - Semantic embeddings and vector search for intelligent text matching
- **CLIP + FAISS** - Vision-language model for image-based plant identification  
- **Ollama** - Local LLM integration for conversational plant advice
- **PyTorch** - Deep learning framework powering the AI features

## 📦 Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai) (for chat functionality)

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd AI-Plant-Care-Assistant/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama model (for chat features)
ollama pull mistral
```

### Run the API
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Core Endpoints

#### 🔍 **Search & Recommendations**
```bash
# Get plant recommendations
POST /recommend
{
  "message": "I need beginner-friendly plants for my dark apartment",
  "care_level": "beginner",
  "pet_safe": true,
  "location": "indoor"
}

# Search by image
POST /search-by-image
# Upload image file with key "file"

# Text-based plant search  
GET /plants?care_level=beginner&pet_safe=true&limit=5
```

#### 💬 **Chat & Diagnosis**
```bash
# Chat with AI assistant
POST /chat
{
  "message": "Why are my peace lily leaves turning yellow?",
  "context": "I water it weekly and it sits by a window"
}

# Diagnose plant problems
POST /diagnose
{
  "symptoms": "brown tips and yellowing leaves",
  "plant_name": "snake plant",
  "additional_context": "recently moved to new pot"
}
```

#### 📋 **Plant Information**
```bash
# Get all plants
GET /plants

# Get specific plant details
GET /plants/{plant_id}

# Get seasonal care advice
GET /seasonal-advice?season=spring
```

## 🌟 Example Usage

### Smart Semantic Search
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "plants that are impossible to kill"
  }'
```

**Response:** Intelligently finds drought-tolerant, low-maintenance plants like Snake Plant and Pothos.

### Image Recognition
```bash
curl -X POST "http://localhost:8000/search-by-image" \
  -F "file=@lavender_photo.jpg"
```

**Response:** Identifies the plant species with confidence scores.

### Problem Diagnosis
```bash
curl -X POST "http://localhost:8000/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "leaves turning yellow and drooping"
  }'
```

**Response:** Matches symptoms to plants and provides specific solutions.

## 🎯 AI Capabilities

### **Semantic Understanding**
The system understands natural language queries:
- "low maintenance plants" → finds drought-tolerant, easy-care species
- "pet safe greenery" → filters for non-toxic plants
- "bathroom plants" → suggests humidity-loving varieties

### **Visual Recognition**
CLIP-powered image analysis can identify:
- Plant species from photos
- Visual similarity matching
- Confidence scoring for identification accuracy

### **Intelligent Recommendations**
Multi-criteria matching considers:
- Care difficulty (beginner → advanced)
- Environmental needs (light, humidity, space)
- Safety requirements (pet/child-friendly)
- Specific use cases (air purification, decoration)

## 📊 Plant Catalog Structure

Each plant entry includes:
```json
{
  "id": 1,
  "name": "Snake Plant",
  "scientific_name": "Sansevieria trifasciata",
  "care_level": "beginner",
  "toxicity": {
    "pets": "mildly toxic to cats and dogs",
    "humans": "generally safe"
  },
  "benefits": ["air purifier", "low maintenance", "drought tolerant"],
  "care_instructions": {
    "watering": "Every 2-3 weeks",
    "light": "Low to bright indirect light",
    "temperature": "60-85°F (15-29°C)"
  },
  "common_issues": [
    {
      "problem": "root rot",
      "causes": ["overwatering", "poor drainage"],
      "solutions": ["reduce watering", "improve drainage"]
    }
  ]
}
```

## 🔧 System Status

Check all AI components:
```bash
GET /status
```

Returns status of:
- SBERT embeddings (semantic search)
- CLIP model (image recognition)  
- FAISS indices (vector search)
- Ollama connection (chat functionality)

## 🚦 Current Status

✅ **Working Features:**
- Semantic text search with SBERT + FAISS
- Image recognition with CLIP + FAISS  
- Conversational AI with Ollama
- Smart plant recommendations
- Problem diagnosis system
- Comprehensive filtering and search

🔮 **Planned Features:**
- React frontend interface
- Care reminders and scheduling

---

**Built with ❤️ for plant lovers everywhere** 🌿

*Powered by state-of-the-art AI to make plant care accessible and enjoyable for everyone.*