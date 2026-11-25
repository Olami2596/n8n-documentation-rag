import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the project root .env file
load_dotenv(Path(".env"))

# Retrieve the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

print("Groq API key loaded successfully.")
print("API key is configured:", bool(groq_api_key))
