import pyaudio
import wave
import sys

def record_audio(filename, duration, sample_rate=44100, channels=1, chunk=1024):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("* Recording audio...")

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

    # Save as WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"* Audio saved as {filename}")

if __name__ == "__main__":
    output_filename = "output.wav"
    record_duration = 5  # Duration in seconds

    if len(sys.argv) > 1:
        record_duration = int(sys.argv[1])

    record_audio(output_filename, record_duration)