from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from google import genai

# Load .env from the same folder as this test file.
PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

if not api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    print(f"Expected .env file at: {ENV_FILE}")
    print("Add GEMINI_API_KEY=your_real_key to that file.")
    sys.exit(1)

print("API key loaded (value hidden).")
print(f"Testing model: {model_name}")

try:
    client = genai.Client(api_key=api_key)
    # Prevent the SDK from hanging on 429 rate limits
    if hasattr(client.interactions, "sdk_configuration"):
        client.interactions.sdk_configuration.retry_config = None

    response = client.interactions.create(
        model=model_name,
        input="Reply with exactly: Gemini connection successful."
    )
    print("\nSUCCESS:")
    print(response.output_text or "(Request succeeded but returned no text.)")

except Exception as exc:
    message = str(exc)
    print("\nGemini test failed.")

    if "429" in message or "Rate limit" in message or "too_many_requests" in message:
        print(f"RATE LIMIT EXCEEDED on '{model_name}'.")
        print("Free tier limit may have been reached for this specific model.")
        fallback = "gemini-3.5-flash"
        if model_name != fallback:
            print(f"Attempting fallback to '{fallback}'...")
            try:
                fb_res = client.interactions.create(
                    model=fallback,
                    input="Reply with exactly: Gemini connection successful via fallback."
                )
                print(f"SUCCESS with fallback model '{fallback}':")
                print(fb_res.output_text)
                print(f"TIP: Update GEMINI_MODEL={fallback} in your .env file.")
            except Exception as fb_err:
                print(f"Fallback also failed: {fb_err}")
    elif "API_KEY_INVALID" in message or "API key not valid" in message:
        print("Google rejected the key. Create/check it at https://aistudio.google.com/apikey")
    elif "No API key was provided" in message:
        print("The SDK did not receive the key. Verify .env path and variable name.")
    elif "NOT_FOUND" in message or "not available" in message:
        print(f"Model '{model_name}' may not be available for this key/account.")
        print("Check the model list available to your account or try 'gemini-3.5-flash'.")
    else:
        print(message)

