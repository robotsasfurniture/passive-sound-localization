import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
from scipy.optimize import minimize

# Speed of sound in air (m/s)
SPEED_OF_SOUND = 343.0


def load_wav(file_path):
    # Load the wav file (assuming all files are the same sample rate)
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data


def time_delay(signal1, signal2, sample_rate):
    # Cross-correlate the two signals to find the time delay
    corr = correlate(signal1, signal2, mode="full")
    lag = np.argmax(corr) - (len(signal2) - 1)
    time_delay = lag / sample_rate
    return time_delay


def estimate_source_position(microphones, time_differences):
    # Define an error function that computes the total squared error
    def error_function(position):
        total_error = 0
        pair_idx = 0  # Track which time difference to use
        for i in range(len(microphones)):
            for j in range(i + 1, len(microphones)):
                mic1 = microphones[i]
                mic2 = microphones[j]
                predicted_distance_diff = np.linalg.norm(
                    position - mic1
                ) - np.linalg.norm(position - mic2)
                time_diff_in_distance = time_differences[pair_idx] * SPEED_OF_SOUND
                total_error += (predicted_distance_diff - time_diff_in_distance) ** 2
                pair_idx += 1
        return total_error

    # Initial guess for the source position (average of microphone positions)
    initial_guess = np.mean(microphones, axis=0)
    result = minimize(error_function, initial_guess)
    return result.x


# Example usage

# Microphone positions (in meters)
microphones = np.array(
    [
        [0.0, 0.0],  # Mic 1 position
        [1.0, 0.0],  # Mic 2 position
        [0.0, 1.0],  # Mic 3 position
        [1.0, 1.0],  # Mic 4 position
    ]
)

# Load signals from the microphones
sample_rate, mic1 = load_wav("output/mic1.wav")
_, mic2 = load_wav("output/mic2.wav")
_, mic3 = load_wav("output/mic3.wav")
_, mic4 = load_wav("output/mic4.wav")

# Compute the time differences between pairs of microphones
time_diffs = [
    time_delay(mic1, mic2, sample_rate),
    time_delay(mic1, mic3, sample_rate),
    time_delay(mic1, mic4, sample_rate),
    time_delay(mic2, mic3, sample_rate),
    time_delay(mic2, mic4, sample_rate),
    time_delay(mic3, mic4, sample_rate),
]

# Estimate the source position
source_position = estimate_source_position(microphones, time_diffs)
print(f"Estimated source position: {source_position}")
