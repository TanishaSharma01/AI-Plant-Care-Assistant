import ollama

# Initialize the Ollama client
client = ollama.Client()

# Define the model and the input prompt
model = "llama2"  # Replace with your model name
prompt = "What is Python?"

# Send the query to the model
response = client.generate(model=model, prompt=prompt)

# Print the response from the model (accessing as dictionary)
print("Response from Ollama:")
print(response['response'])  # Changed from response.response to response['response']

# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# import ollama
#
# app = FastAPI()
# client = ollama.Client()
#
# class PromptRequest(BaseModel):
#     prompt: str
#
# @app.post("/generate")
# async def generate_response(request: PromptRequest):
#     try:
#         response = client.generate(model="llama2", prompt=request.prompt)
#         return {"response": response['response']}
#     except Exception as e:
#         return {"error": str(e)}