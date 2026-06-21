from dotenv import load_dotenv
import os

load_dotenv()

def get_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}. Check your .env file.")
    return val

HF_TOKEN = get_env("HF_TOKEN")
