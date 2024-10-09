import numpy as np
import pyaudio
import wave
import sys


def record_audio(filename_prefix, duration, sample_rate=44100, channels=2, chunk=1024):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Find the device index for a 2-channel input device
    device_index = None
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info["maxInputChannels"] >= channels:
            device_index = i
            break

    if device_index is None:
        print("Error: Could not find a 2-channel input device.")
        p.terminate()
        return

    # Open stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk,
    )

    print(f"* Recording {channels} channels of audio...")

    frames = []

    # Calculate the number of chunks to record
    chunks_to_record = int(sample_rate / chunk * duration)

    # Record audio
    for _ in range(chunks_to_record):
        data = stream.read(chunk)
        frames.append(data)

    print("* Done recording")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    p.terminate()

    # Convert the byte string to a NumPy array
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

    # Reshape the data to separate channels
    audio_channels = audio_data.reshape(-1, channels)

    # Save each channel to a separate WAV file
    for i in range(channels):
        filename = f"audio_files/{filename_prefix}_channel_{i+1}.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono file for each channel
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_channels[:, i].tobytes())
        print(f"* Audio for channel {i+1} saved as {filename}")


if __name__ == "__main__":
    output_filename_prefix = "output"
    record_duration = 5  # Duration in seconds

    if len(sys.argv) > 1:
        record_duration = int(sys.argv[1])

    record_audio(output_filename_prefix, record_duration)
