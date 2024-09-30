import numpy as np
from scipy.io import wavfile
import os

# Constants
SAMPLE_RATE = 44100  # Sampling rate in Hz
DURATION = 2.0  # Duration of the sound in seconds
FREQUENCY = 440.0  # Frequency of the sound (A4 pitch)
SPEED_OF_SOUND = 343.0  # Speed of sound in m/s


# Function to generate a sine wave
def generate_sine_wave(frequency, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    return wave


# Function to create time-delayed signals for microphones
def simulate_microphone_signals(
    microphones, source_position, sample_rate, frequency, duration
):
    source_signal = generate_sine_wave(frequency, sample_rate, duration)
    signals = []

    for mic in microphones:
        # Calculate distance between the source and microphone
        distance = np.linalg.norm(source_position - mic)
        # Calculate time delay based on the distance and speed of sound
        time_delay = distance / SPEED_OF_SOUND
        # Convert time delay into sample delay
        sample_delay = int(time_delay * sample_rate)
        # Shift the signal by inserting zeros at the beginning (simulate delay)
        delayed_signal = np.pad(source_signal, (sample_delay, 0), mode="constant")[
            : len(source_signal)
        ]
        signals.append(delayed_signal)

    return signals


# Function to save signals to WAV files
def save_signals_to_wav(signals, sample_rate, folder="output"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, signal in enumerate(signals):
        file_name = os.path.join(folder, f"mic{i+1}.wav")
        wavfile.write(
            file_name, sample_rate, np.int16(signal * 32767)
        )  # Scale to int16 for WAV format
        print(f"Saved {file_name}")


# Example microphone positions (in meters)
microphones = np.array(
    [
        [0.0, 0.0],  # Mic 1
        [1.0, 0.0],  # Mic 2
        [0.0, 1.0],  # Mic 3
        [1.0, 1.0],  # Mic 4
    ]
)

# Sound source position (in meters)
source_position = np.array([5, 5])  # Source is 2 meters away from the array

# Simulate microphone signals
signals = simulate_microphone_signals(
    microphones, source_position, SAMPLE_RATE, FREQUENCY, DURATION
)

# Save the signals as .wav files
save_signals_to_wav(signals, SAMPLE_RATE)
