from queue import Queue
from websockets import Data, ConnectionClosed, InvalidHandshake, InvalidURI, WebSocketClientProtocol
import websockets
import asyncio
import json
import base64
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class InvalidWebsocketURIError(Exception):
    def __init__(self, websocket_url: str) -> None:
        super().__init__(f"Invalid Websocker URI was passed: {websocket_url}")

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

# class CustomMessageQueue():
    # def __init__(self):
    #     self.queue = Queue()

    # async def put(self, message: Any):
    #     await self.queue.put(message)

    # async def receive(self, condition):
    #     while True:
    #         message = await self.queue.get()
    #         if condition(message):
    #             return message
    #         await self.queue.put(message)

class OpenAIWebsocketClient:
    def __init__(self,  api_key: str, websocket_url: str):
        self.api_key: str = api_key
        self.websocket_url: str = websocket_url
        self.ws: Optional[WebSocketClientProtocol] = None

    def __enter__(self):
        self._connect()
        self._configure_session()
        return self
    
    def __exit__(self):
        self._close()

    def _connect(self) -> None:
        self.ws = websockets.connect(
            self.websocket_url,
            extra_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        )

        message = json.loads(self.ws.recv())
        if message["type"] != "session.created":
            raise SessionNotCreatedError("Session was not created")

    def _configure_session(self) -> None:
        self.ws.send(json.dumps({
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

        message = json.loads(self.ws.recv())
        if message["type"] == "session.updated":
            raise SessionNotUpdatedError("Session was not updated")



    def send_audio(self, audio_chunk: bytes) -> None:
        audio_b64 = base64.b64encode(audio_chunk).decode()

        self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))

    def receive_text_response(self) -> None:
        for message in self.ws:
            response = json.loads(message)
            if response["type"] == "response.done":
                return response["output"][0]["content"][0]["text"]

    def _close(self) -> None:
        if self.ws:
            self.ws.close()
            self.ws = None


# class RealtimeOpenAIWebsocketClient:
#     def __init__(self, api_key: str, websocket_url: str):
#         self.api_key: str = api_key
#         self.websocket_url: str = websocket_url
#         self.ws: Optional[WebSocketClientProtocol] = None
#         self.message_queue = CustomMessageQueue(self._receive_message)

#     async def __aenter__(self):
#         await self.connect()
#         return self
    
#     async def connect(self) -> None:
#         try:
#             self.ws = await websockets.connect(
#                 self.websocket_url,
#                 extra_headers={
#                     "Authorization": f"Bearer {self.api_key}",
#                     "OpenAI-Beta": "realtime=v1"
#                 }
#             )
#         except InvalidURI:
#             raise InvalidWebsocketURIError(self.websocket_url)
#         except OSError:
#             raise WebsocketTCPError("Error occurred with Websocket TCP connection")
#         except InvalidHandshake:
#             raise InvalidWebsocketHandshakeError("Error occurred with Websocket handshake")
#         except TimeoutError:
#             raise WebsocketTimeOutError("Websocket timed out")
        
#         message = await self.message_queue.receive(lambda m: m.get("type") == "session.created")
#         if message["type"] == "error":
#             raise SessionNotCreatedError("Session was not created")
        
#         await self._configure()


#     async def _configure(self) -> None:
#         await self.ws.send(json.dumps({
#             "type": "session.update",
#             "session": {
#                 "modalities": ["text"],
#                 "instructions": "Your knowledge cutoff is 2023-10. You are a helpful assistant.",
#                 "input_audio_format": "pcm16",
#                 "turn_detection": {
#                     "type": "server_vad"
#                 }
#             }
#         }))

#         # Wait for session.updated message
#         message = await self.message_queue.receive(lambda m: m.get("type") == "session.updated")

#         if message["type"] == "error":
#             raise SessionNotUpdatedError("Session was not updated")
    
#     async def close(self) -> None:
#         if self.ws:
#             await self.ws.close()
#             self.ws = None
#     async def __aexit__(self, *args):
#         await self.close()
    
#     async def send_audio(self, audio_chunk: bytes) -> None:
#         audio_b64 = base64.b64encode(audio_chunk).decode()

#         await self.ws.send(json.dumps({
#             "type": "input_audio_buffer.append",
#             "audio": audio_b64
#         }))
    
#     async def _receive_message(self) -> Optional[Data]:
#         async for message in self.ws:
#             return message
#         return None
    
#     async def receive_text_response(self):
#         pass