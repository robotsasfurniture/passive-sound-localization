import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TranscriberConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")  # Replace with your actual OpenAI API key
    model_name: str = "whisper-1"  # The name of the OpenAI Whisper model to use
    language: str = "en"  # Language code for transcription (e.g., 'en' for English)