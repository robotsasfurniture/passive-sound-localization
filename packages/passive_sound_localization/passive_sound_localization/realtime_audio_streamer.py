import logging
from typing import Generator, List
import numpy as np
import queue
import threading
from pydub import AudioSegment
import pyaudio

# from passive_sound_localization.models.configs.realtime_streamer import (
#     RealtimeAudioStreamerConfig,
# )
from models.configs.realtime_streamer import RealtimeAudioStreamerConfig


logger = logging.getLogger(__name__)


class RealtimeAudioStreamer:
    def __init__(self, config: RealtimeAudioStreamerConfig):
        self.mic_indices = config.device_indices
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.chunk_size = config.chunk
        self.is_running = False
        self.pyaudio_instance = pyaudio.PyAudio()
        self.streams = []
        self.audio_queues = [queue.Queue() for _ in self.mic_indices]

        logger.info(f"Mic indices: {self.mic_indices}")
        for mic_index in self.mic_indices:
            logger.debug(f"Opening stream for mic index: {mic_index}")
            stream = self.pyaudio_instance.open(
                rate=self.sample_rate,
                channels=self.channels,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=self.chunk_size,
            )
            self.streams.append(stream)

    def __enter__(self):
        self.is_running = True
        self.start_stream_threads()
        return self

    def start_stream_threads(self):
        """Start a thread for each audio stream to continuously push audio data to its queue."""
        self.stream_threads = []
        for idx, stream in enumerate(self.streams):
            thread = threading.Thread(
                target=self.stream_to_queue, args=(idx, stream), daemon=True
            )
            thread.start()
            self.stream_threads.append(thread)

    def stream_to_queue(self, idx: int, stream):
        """Reads audio data from the stream and places it in the appropriate queue."""
        while self.is_running:
            try:
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queues[idx].put(audio_data)
            except Exception as e:
                logger.error(f"Error in stream_to_queue for mic {idx}: {e}")
                self.audio_queues[idx].put(b"")

    def audio_generator(self) -> Generator[List[bytes], None, None]:
        """Thread-safe generator that yields a list of audio data chunks from all microphones."""
        while self.is_running:
            yield [q.get() for q in self.audio_queues]

    def __exit__(self, *args, **kwargs):
        self.is_running = False
        for stream in self.streams:
            stream.stop_stream()
            stream.close()
        self.pyaudio_instance.terminate()

    def resample_stream(self, stream: bytes, target_sample_rate: int = 24000) -> bytes:
        try:
            np_stream = np.frombuffer(stream, dtype=np.int16).astype(np.int16).tobytes()
            audio_segment = AudioSegment(
                np_stream, frame_rate=self.sample_rate, sample_width=2, channels=1
            )
            audio_segment = audio_segment.set_frame_rate(
                target_sample_rate
            ).set_channels(1)
            return audio_segment.raw_data

        except Exception as e:
            logger.error(f"Error in resample_stream: {e}")
            return b""
