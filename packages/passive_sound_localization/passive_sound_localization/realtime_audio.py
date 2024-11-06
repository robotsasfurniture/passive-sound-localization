from concurrent.futures import ThreadPoolExecutor
from localization import SoundLocalizer
from realtime_audio_streamer import RealtimeAudioStreamer
from models.configs import RealtimeAudioStreamerConfig
from models.configs import LocalizationConfig
from models.configs import OpenAIWebsocketConfig
from realtime_openai_websocket import OpenAIWebsocketClient


from queue import Queue

import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


commands = Queue(maxsize=10)
locations = Queue(maxsize=10)

# commands = []
# locations = []
def send_audio_continuously(client, single_channel_generator):
    print("Threading...")
    for single_channel_audio in single_channel_generator:
        client.send_audio(single_channel_audio)


def receive_text_messages(client, logger):
    logger.info("OpanAI: Listening to audio stream")
    while True:
        try:
            command = client.receive_text_response()
            if command and command.strip() == "MOVE_TO":
                print(command)
                logger.info(f"Received command: {command}")
                
                if commands.full():
                    commands.get()
                    commands.task_done()
                commands.put(command)
                # commands.append(command)
        except Exception as e:
            print(f"Error receiving response: {e}")
            break  # Exit loop if server disconnects

def realtime_localization(multi_channel_stream, localizer, logger):
    logger.info("Localization: Listening to audio stream")
    try:
        did_get = True
        for audio_data in multi_channel_stream:
            #  Stream audio data and pass it to the localizer
            localization_stream = localizer.localize_stream(
                [audio_data[k] for k in audio_data.keys()]
            )

            for localization_results in localization_stream:
                # locations.append(localization_results)
                if locations.full():
                    locations.get()
                
                locations.put(localization_results)
            
            if did_get:
                locations.task_done()
            did_get = False
                
    except Exception as e:
        print(f"Realtime Localization error: {e}")

def command_executor(publisher, logger):
    logger.info("Executor: listening for command")
    while True:
        try:
            # if len(commands) > 0:
            if commands.qsize() > 0:
                logger.info(f"Got command, locations: {locations}")
                commands.get()
                commands.task_done()
                # commands.pop()
                # if len(locations) > 0:
                if locations.qsize() > 0:
                    publisher(locations.get())
            
                locations.task_done()
                    # publisher(locations.pop())
        except Exception as e:
            print(f"Command executor error: {e}")


def publish_results(localization_results):
    print(localization_results)
        
          

def main():
    print("Hello world")
    audio_streamer_config = RealtimeAudioStreamerConfig()
    localizer_config = LocalizationConfig()
    websocket_config = OpenAIWebsocketConfig()
    print("Running realtime audio...")

    localizer = SoundLocalizer(localizer_config)

    with (
            RealtimeAudioStreamer(audio_streamer_config) as streamer,
            OpenAIWebsocketClient(websocket_config) as client
        ):
            multi_channel_stream = streamer.multi_channel_gen()
            single_channel_stream = streamer.single_channel_gen()

            with ThreadPoolExecutor(max_workers=4) as executor:
                logger.info("Threading log")
                executor.submit(send_audio_continuously, client, single_channel_stream)
                executor.submit(receive_text_messages, client, logger)
                executor.submit(realtime_localization, multi_channel_stream, localizer, logger)
                executor.submit(command_executor, lambda x: publish_results(x), logger)

if __name__ == "__main__":
    main()