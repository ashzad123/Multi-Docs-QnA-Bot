import google.generativeai as genai
import os

# Assuming you have configured the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
available_models = genai.list_models()

# Iterate over the generator and print each model
for model in available_models:
    print(model)
