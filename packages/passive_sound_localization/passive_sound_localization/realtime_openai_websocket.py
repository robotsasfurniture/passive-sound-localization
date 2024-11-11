from websockets import WebSocketClientProtocol
from websockets.sync.client import connect
import json
import base64
import logging
from typing import Optional

from passive_sound_localization.models.configs.openai_websocket import (
    OpenAIWebsocketConfig,
)

# from models.configs.openai_websocket import (
#     OpenAIWebsocketConfig,
# )  # Only needed to run with `realtime_audio.py`

logger = logging.getLogger(__name__)


class InvalidWebsocketURIError(Exception):
    def __init__(self, websocket_url: str) -> None:
        super().__init__(f"Invalid Websocker URI was passed: {websocket_url}")


class SessionNotCreatedError(Exception):
    def __init__(self) -> None:
        super().__init__("Session was not created")


class SessionNotUpdatedError(Exception):
    def __init__(self) -> None:
        super().__init__("Session was not updated")


class OpenAIWebsocketError(Exception):
    def __init__(self, error_code: str, error_message: str) -> None:
        super().__init__(
            f"OpenAI websocket erred with error type `{error_code}`: {error_message}"
        )


class OpenAIRateLimitError(Exception):
    def __init__(self) -> None:
        super().__init__("Hit OpenAI Realtime API rate limit")


class OpenAITimeoutError(Exception):
    def __init__(self, timeout: float) -> None:
        super().__init__(
            f"OpenAI websocket timed out because it did not receive a message in {timeout} seconds"
        )


INSTRUCTIONS = """
    The instructor robot will receive audio input to determine movement actions based on command cues. For each command, the robot should perform a corresponding movement action as follows:

- **Audio cues for 'Come here'** – MOVE_TO
- **Audio cues for 'Over here'** – MOVE_TO


The robot should only respond using these commands. The robot should analyze audio input continuously and prioritize the most recent command. If ambiguous commands are detected (e.g., unclear or overlapping), the robot should remain in its last known state until a clearer command is received.
    """


# TODO: Make it take in Hydra config
class OpenAIWebsocketClient:
    def __init__(self, config: OpenAIWebsocketConfig):
        self.api_key: str = config.api_key
        self.websocket_url: str = config.websocket_url
        self.session_id: Optional[str] = None
        self.instructions: str = INSTRUCTIONS
        self.ws: Optional[WebSocketClientProtocol] = None

    def __enter__(self):
        self._connect()
        self._configure_session()
        print("Connected websocket...")
        return self

    def __exit__(self):
        self._close()

    def _connect(self) -> None:
        self.ws = connect(
            uri=self.websocket_url,
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )

        message = json.loads(self.ws.recv())
        self.session_id = message["session"]["id"]
        if message["type"] != "session.created":
            raise SessionNotCreatedError()

    def _configure_session(self) -> None:
        self.ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "instructions": self.instructions,
                        "input_audio_format": "pcm16",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                        },
                        "temperature": 0.8,
                        "max_response_output_tokens": 4096,
                    },
                }
            )
        )

        message = json.loads(self.ws.recv())
        if message["type"] != "session.updated":
            raise SessionNotUpdatedError()

    def send_audio(self, audio_chunk: bytes) -> None:
        # Audio needs to be encoded in Base64 before being sent to the OpenAI Realtime API
        audio_b64 = base64.b64encode(audio_chunk).decode()

        self.ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
        )

    def receive_text_response(self, timeout: Optional[float] = None) -> str:
        try:
            # Tries to receive the next message (in a blocking manner) from the OpenAI websocket
            # If the message doesn't arrive in 300ms, then it raises a TimeoutError
            message = json.loads(self.ws.recv(timeout=timeout))
        except TimeoutError:
            raise OpenAITimeoutError(timeout=timeout)

        # Print message just to see what we're receiving
        # print(message)

        # Checks to see any general errors
        if message["type"] == "error":
            raise OpenAIWebsocketError(
                error_code=message["error"]["code"],
                error_message=message["error"]["message"],
            )

        # Checks to see whether OpenAI is specifically rate limiting our responses
        if (
            message["type"] == "response.done"
            and message["response"]["status_details"]["error"]["code"]
            == "rate_limit_exceeded"
        ):
            raise OpenAIRateLimitError()

        # Checks to see if an actual text response was sent, and returns the text
        if message["type"] == "response.text.done":
            return message["text"]

    def _close(self) -> None:
        if self.ws:
            self.ws.close()
            self.ws = None
