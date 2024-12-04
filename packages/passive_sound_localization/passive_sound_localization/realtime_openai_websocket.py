from websockets import WebSocketClientProtocol
from websockets.sync.client import connect
import json
import base64
import logging
from typing import List, Optional, TypedDict
from enum import Enum

# from passive_sound_localization.models.configs.openai_websocket import (
#     OpenAIWebsocketConfig,
# )
# from passive_sound_localization.models.ring_buffer import AudioRingBuffer

from models.configs.openai_websocket import (
    OpenAIWebsocketConfig,
)  # Only needed to run with `realtime_audio.py`
from models.ring_buffer import AudioRingBuffer

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


class OpenAIResponseType(Enum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"
    COMPLETED = "COMPLETED"
    NONE = "NONE"


class OpenAIResponse(TypedDict):
    type: OpenAIResponseType
    text: str = ""
    audio_chunks: List[List[bytes]] = []


INSTRUCTIONS = """
    The instructor robot will receive audio input to determine movement actions based on command cues. For each command, the robot should perform a corresponding movement action as follows:

- **Audio cues for 'Table Bot come here'** – MOVE_TO
- **Audio cues for 'Table Bot over here'** – MOVE_TO


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
        self.audio_ring_buffer: AudioRingBuffer = AudioRingBuffer(max_chunks=10000)
        self.current_ms: int = 0
        self.started_ms: int = 0
        self.stopped_ms: int = 0
        self.speech_queue: List[tuple[int, int]] = []

    def __enter__(self):
        self._connect()
        self._configure_session()
        logger.info("Connected websocket...")
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

    def store_audio(self, audio_chunks: List[bytes]) -> None:
        # Calculate audio duration in milliseconds
        self.audio_ring_buffer.add_chunk(self.current_ms, audio_chunks)
        self.current_ms += 10

    def receive_response(self) -> OpenAIResponse:
        message = json.loads(self.ws.recv())

        # TODO(@john2360): Fix this. The error checking returns key errors.
        # # Checks to see any general errors
        # if message["type"] == "error":
        #     logger.error(f"OpenAI websocket error: {message['error']}")
        #     raise OpenAIWebsocketError(
        #         error_code=message["error"]["code"],
        #         error_message=message["error"]["message"],
        #     )

        # # Checks to see whether OpenAI is specifically rate limiting our responses
        # if (
        #     message["type"] == "response.done"
        #     and message["response"]["status_details"]["error"]["code"]
        #     == "rate_limit_exceeded"
        # ):
        #     logger.error("Hit OpenAI Realtime API rate limit")
        #     raise OpenAIRateLimitError()

        # Checks to see if an actual text response was sent, and returns the text
        if message["type"] == "response.text.done":
            # return OpenAIResponse(type=OpenAIResponseType.TEXT, text=message["text"])
            speech_start_ms, speech_end_ms = self.speech_queue.pop(0)
            audio_chunks = self.audio_ring_buffer.get_chunks(
                start_ms=speech_start_ms, end_ms=speech_end_ms
            )

            return OpenAIResponse(
                type=OpenAIResponseType.COMPLETED,
                audio_chunks=audio_chunks,
                text=message["text"],
            )

        # Speech started, so we need to get the audio chunks
        if message["type"] == "input_audio_buffer.speech_started":
            self.started_ms = self.current_ms

        # Speech ended, so we need to reset the current time
        if message["type"] == "input_audio_buffer.speech_stopped":
            self.started_ms = 0
            self.speech_queue.append((self.started_ms, self.current_ms))
            # return OpenAIResponse(
            #     type=OpenAIResponseType.AUDIO, audio_chunks=audio_chunks
            # )

        return OpenAIResponse(type=OpenAIResponseType.NONE)

    def _close(self) -> None:
        if self.ws:
            self.ws.close()
            self.ws = None
