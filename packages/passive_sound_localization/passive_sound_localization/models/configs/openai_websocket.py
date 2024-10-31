from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass(frozen=True)
class OpenAIWebsocketConfig:
    api_key: str = os.getenv(
        "OPENAI_API_KEY", ""
    )  # Replace with your actual OpenAI API key
    websocket_url: str = (
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"  # The name of the OpenAI Websocket url,
    )
