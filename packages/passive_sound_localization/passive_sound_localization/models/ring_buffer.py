from collections import deque
from typing import List


class AudioRingBuffer:
    def __init__(self, max_chunks: int):
        """
        Initialize the ring buffer.

        :param max_chunks: Maximum number of chunks to store.
        """
        self.buffer = deque(maxlen=max_chunks)  # Circular buffer

    def add_chunk(self, current_time_ms: int, audio_chunks: List[bytes]):
        """
        Add a new audio chunk to the buffer.

        :param audio_chunk: Bytes or numpy array representing the audio chunk.
        """
        self.buffer.append((current_time_ms, audio_chunks))

    def get_chunks(self, start_ms: int, end_ms: int) -> List[List[bytes]]:
        """
        Retrieve audio data for a given millisecond range.

        :param start_ms: Start of the range in milliseconds.
        :param end_ms: End of the range in milliseconds.
        :return: List of audio chunks within the time range.
        """
        return [
            chunk for timestamp, chunk in self.buffer if start_ms <= timestamp < end_ms
        ]

    def __len__(self):
        return len(self.buffer)
