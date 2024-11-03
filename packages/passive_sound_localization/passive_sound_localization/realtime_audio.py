from concurrent.futures import ThreadPoolExecutor
# from localization import SoundLocalizer
from realtime_audio_streamer import RealtimeAudioStreamer
from models.configs import LocalizationConfig
from models.configs import OpenAIWebsocketConfig
from realtime_openai_websocket import OpenAIWebsocketClient

import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def send_audio_continuously(client, single_channel_generator):
    for audio_chunk in single_channel_generator():
        client.send_audio(audio_chunk)

def receive_text_messages(client):
    while True:
        try:
            command = client.receive_text_response()
            if command:
                print(f"Received command: {command}")
        except Exception as e:
            print(f"Error receiving response: {e}")
            break  # Exit loop if server disconnects
          

def main():
    localizer_config = LocalizationConfig()
    websocket_config = OpenAIWebsocketConfig()
    print("Running realtime audio...")

    # localizer = SoundLocalizer(localizer_config)

    with (
            RealtimeAudioStreamer(
                 sample_rate=localizer_config.sample_rate,
                 channels=1,
                 chunk=localizer_config.fft_size
            ) as streamer,
            OpenAIWebsocketClient(websocket_config) as client
        ):
            # for single_channel_audio, multi_channel_audio in zip(streamer.single_channel_gen(), streamer.multi_channel_gen()):
            for single_channel_audio in streamer.single_channel_gen():
            # for multi_channel_audio in streamer.multi_channel_gen():
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit both functions to the thread pool
                    executor.submit(send_audio_continuously, client, streamer.single_channel_gen)
                    executor.submit(receive_text_messages, client)
                    

                # for localization_results in localizer.localize_stream(
                #     [multi_channel_audio[k] for k in multi_channel_audio.keys()]
                # ):
                #     print(f"Location results: {str(localization_results)}")
                #     for result in localization_results:
                #         print(f"Estimated source at angle: {result.angle} degrees, distance: {result.distance} meters")

                #         coordinate_representation = (
                #             localizer.compute_cartesian_coordinates(
                #                 result.distance, result.angle
                #             )
                #         )
                #         print(
                #             f"Cartesian Coordinates: x={coordinate_representation[0]}, y={coordinate_representation[1]}"
                #         )