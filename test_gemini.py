import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

client = genai.Client(api_key=api_key)

# Models from your system's list
candidate_models = [
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-3.5-flash-lite",
]

print("Connecting to Gemini API...")

success = False
for model_name in candidate_models:
    print(f"Trying model: {model_name}...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Explain Retrieval-Augmented Generation (RAG) in one sentence for a developer."
        )
        print("\n--- Gemini Response ---")
        print(response.text.strip())
        print("-----------------------")
        print(f"✅ Success using model: '{model_name}'!")
        print("✅ Phase 1 API Connection Successful!")
        success = True
        break
    except Exception as e:
        print(f"❌ Failed with {model_name}: {e}\n")

if not success:
    print("Could not reach a working model endpoint. Please double check API Key permissions.")