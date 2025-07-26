# ollama_utils.py
import ollama

client = ollama.Client()

def get_ollama_response(prompt: str):
    response = client.generate(model="llama2", prompt=prompt)
    return response.response
