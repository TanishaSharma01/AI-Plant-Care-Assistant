import requests
import json

BASE_URL = "http://localhost:8000"


def test_status():
    print("🔍 Checking system status...")
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        status = response.json()
        print(f"Utils available: {status['dependencies']['utils']}")
        print(f"Ollama available: {status['dependencies']['ollama_utils']}")
        if 'ml_components' in status:
            print(f"ML components: {status['ml_components']}")
    print("-" * 50)


def test_health():
    print("🔍 Checking health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        health = response.json()
        print(f"Plants loaded: {health['plants_loaded']}")
        print(f"Features: {health['features']}")
    print("-" * 50)


def test_plants_list():
    print("🔍 Checking plant catalog...")
    response = requests.get(f"{BASE_URL}/plants?limit=5")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total plants: {result['total_count']}")
        if result['plants']:
            print("Sample plants:")
            for plant in result['plants'][:3]:
                print(f"  - {plant['name']}")
        else:
            print("No plants found!")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)


def test_recommend():
    print("🔍 Testing recommend endpoint...")

    data = {"message": "recommend beginner plants"}

    response = requests.post(
        f"{BASE_URL}/recommend",
        headers={"Content-Type": "application/json"},
        json=data
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response keys: {result.keys()}")

        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            recommendations = result.get('recommendations', [])
            print(f"Found {len(recommendations)} recommendations")

            if recommendations:
                print("Top recommendations:")
                for i, plant in enumerate(recommendations[:3]):
                    print(f"  {i + 1}. {plant.get('name', 'Unknown')}")
                    print(f"     Description: {plant.get('description', 'No description')[:60]}...")
            else:
                print("❌ No recommendations returned")

            print(f"AI advice: {result.get('ai_advice', 'No AI advice')[:100]}...")
            print(f"Search method: {result.get('search_method', 'Unknown')}")
    else:
        print(f"❌ HTTP Error: {response.text}")

    print("-" * 50)


if __name__ == "__main__":
    print("🐛 Debugging Plant Recommendation System")
    print("=" * 60)

    try:
        test_status()
        test_health()
        test_plants_list()
        test_recommend()

        print("✅ Debug completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
        print("Make sure server is running: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")