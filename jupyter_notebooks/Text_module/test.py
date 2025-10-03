import os
from google.generativeai import configure, GenerativeModel

# Load API key
api_key = os.getenv("GOOGLE_API_KEY")  # or directly paste for test
configure(api_key=api_key)

# Use a supported model
model = GenerativeModel("gemini-1.5-flash")

# Send a simple prompt
response = model.generate_content("Hello Gemini, just testing my API key.")
print(response.text)
