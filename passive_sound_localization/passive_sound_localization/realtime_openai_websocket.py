from websockets import Data, ConnectionClosed, InvalidHandshake, InvalidURI, WebSocketClientProtocol
import websockets
import asyncio
import json
import base64
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class InvalidWebsocketURIError(Exception):
    pass

class WebsocketTCPError(Exception):
    pass

class InvalidWebsocketHandshakeError(Exception):
    pass

class WebsocketTimeOutError(Exception):
    pass

class SessionNotCreatedError(Exception):
    pass

class SessionNotUpdatedError(Exception):
    pass


# TODO: Azure OpenAI Websocket Client Reference: https://github.com/Azure-Samples/aoai-realtime-audio-sdk/blob/main/python/rtclient/__init__.py#L616
# TODO: https://github.com/Azure-Samples/aoai-realtime-audio-sdk/blob/main/python/rtclient/low_level_client.py
# TODO: Make it take in Hydra config

class CustomMessageQueue():
    def __init__(self):
        self.message_queue = asyncio.Queue()

    async def put(self, message: Any):
        await self.message_queue.put(message)

    async def receive(self):
        pass



class RealtimeOpenAIWebsocketClient:
    def __init__(self, api_key: str, websocket_url: str):
        self.api_key: str = api_key
        self.websocket_url: str = websocket_url
        self.ws: Optional[WebSocketClientProtocol] = None
        self.message_queue = CustomMessageQueue()

    async def __aenter__(self):
        await self.connect()
        return self
    
    async def connect(self) -> None:
        try:
            self.ws = await websockets.connect(
                self.websocket_url,
                extra_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
        except InvalidURI:
            raise InvalidWebsocketURIError(f"Invalid Websocker URI was passed: {self.websocket_url}")
        except OSError:
            raise WebsocketTCPError(f"Error occurred with Websocket TCP connection")
        except InvalidHandshake:
            raise InvalidWebsocketHandshakeError("Error occurred with Websocket handshake")
        except TimeoutError:
            raise WebsocketTimeOutError("Websocket timed out") 
    async def configure(self) -> None:
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": "Your knowledge cutoff is 2023-10. You are a helpful assistant.",
                "input_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad"
                }
            }
        }))

        # Wait for session.updated message
        message = await self.message_queue.receive(lambda m: m.get("type") == "session.updated")

        if message["type"] == "error":
            raise SessionNotUpdatedError("Session was not updated")
    
    async def close(self) -> None:
        if self.ws:
            await self.ws.close()
            self.ws = None
    async def __aexit__(self, *args):
        await self.close()
    
    async def send_audio(self, audio_chunk: bytes) -> None:
        audio_b64 = base64.b64encode(audio_chunk).decode()

        await self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
    
    async def _receive_message(self) -> Optional[Data]:
        async for message in self.ws:
            return message
        return None
    
    async def receive_text_response(self):
        pass