# from realtime_audio_streamer import RealtimeAudioStreamer
from localization import SoundLocalizer
from models.configs.localization import LocalizationConfig
# from models.configs.realtime_streamer import RealtimeAudioStreamerConfig

def main() -> None:
    localizer = SoundLocalizer(config=LocalizationConfig())
    # with RealtimeAudioStreamer(config=RealtimeAudioStreamerConfig()) as streamer:
    #     return
    #     for audio_streams in streamer.audio_generator():
    #         try:
    #             if audio_streams is None:
    #                 print("Audio streams are None")
    #                 continue

    #             print("Got audio!")
    #         except Exception as e:
    #             print(f"Realtime audio error: {e}")


if __name__ == "__main__":
    main()