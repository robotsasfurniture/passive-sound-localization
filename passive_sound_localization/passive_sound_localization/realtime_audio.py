from passive_sound_localization.realtime_audio_streamer import RealtimeAudioStreamer
from passive_sound_localization.realtime_openai_websocket import RealtimeOpenAIWebsocketClient

import asyncio
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)


def send_audio(single_channel_generator, client: RealtimeOpenAIWebsocketClient):
    pass

def receive_messages():
    pass

async def main():
    print("Running realtime audio...")
    api_key = os.getenv(
        "OPENAI_API_KEY", ""
    )
    websocket_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"


    async with (
        RealtimeAudioStreamer() as streamer,
        RealtimeOpenAIWebsocketClient(api_key=api_key, websocket_url=websocket_url) as client
    ):
        await client.configure()

        async for single_channel_audio, multi_chanel_audio in zip(streamer.single_channel_gen(), streamer.multi_channel_gen()):
            await client.send_audio(single_channel_audio)
            command = await client.receive_text_response()
            localization_results = await localizer.localize(multi_chanel_audio)
            await execute_robot(command, localization_results)

    # async with (
    #     RealtimeAudioStreamer() as streamer, 
    #     RealtimeOpenAIWebsocketClient(api_key=api_key, websocket_url=websocket_url) as client
    # ):
    #     audio_generator = streamer.audio_generator()
    #     mixed_audio_generator = streamer.mixed_audio_generator()
    #     await client.configure()

        
    #     await asyncio.gather(send_audio(client, audio_file_path), receive_messages(client, out_dir))
        # audio_gen = streamer.audio_generator()
        
        # ws_queue = asyncio.Queue()

        # async def audio_producer(queue):
        #     async for audio_data in audio_gen:
        #         await queue.put(audio_data)

        
        # async def websocker_consumer(queue: asyncio.Queue, client: RealtimeOpenAIWebsocketClient):
        #     while True:
        #         audio_data = await queue.get()
        #         await client.stream_audio(audio_chunk=audio_data)
        #         queue.task_done()

        # handle_messages_task = asyncio.create_task(client.handle_messages())
        # audio_producer_task = asyncio.create_task(audio_producer(queue=ws_queue))
        # websocker_consumer_task = asyncio.create_task(websocker_consumer(
        #     queue=ws_queue, 
        #     client=client
        # ))

        # await asyncio.gather(
        #     handle_messages_task, 
        #     audio_producer_task,
        #     websocker_consumer_task
        # )


# Usage Example
if __name__ == "__main__":
    asyncio.run(main())
