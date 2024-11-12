from models.configs.openai_websocket import (
    OpenAIWebsocketConfig,
)
from models.configs.realtime_streamer import (
    RealtimeAudioStreamerConfig,
)
from realtime_openai_websocket import OpenAIWebsocketClient
from realtime_audio_streamer import RealtimeAudioStreamer
import threading

# Configuration for OpenAI WebSocket
openai_config = OpenAIWebsocketConfig(
    api_key="nah",
    websocket_url="wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
)


def send_audio(websocket_client, audio_streamer):
    print("Sending audio")
    for mic_streams in audio_streamer.audio_generator():
        audio_chunk = audio_streamer.resample_stream(mic_streams[0])

        try:
            if audio_chunk is None:
                continue

            websocket_client.send_audio(audio_chunk)
        except Exception as e:
            print(f"Error in send_audio: {e}")
            raise e


def receive_audio(websocket_client):
    print("Receiving audio")
    while True:
        try:
            response = websocket_client.receive_text_response(timeout=None)
            print("Received response:", response)
        except Exception as e:
            print(f"Error in receive_audio: {e}")


# Main function to test audio streaming to OpenAI
def main():
    with (
        OpenAIWebsocketClient(openai_config) as websocket_client,
        RealtimeAudioStreamer(
            mic_indices=[1],
            sample_rate=44100,
            chunk_size=1024,
        ) as audio_streamer,
        
    ):
        

        # Create threads for sending and receiving audio
        sender_thread = threading.Thread(
            target=send_audio,
            args=(websocket_client, audio_streamer),
        )
        receiver_thread = threading.Thread(
            target=receive_audio, args=(websocket_client,)
        )

        # Start the threads
        sender_thread.start()
        receiver_thread.start()

        # Optionally, join the threads if you want to wait for them to finish
        sender_thread.join()
        receiver_thread.join()


if __name__ == "__main__":
    main()
