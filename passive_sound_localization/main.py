import numpy as np
import pyroomacoustics as pra
import scipy.signal as signal
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Constants
c = 343  # Speed of sound in m/s
fs = 48000  # Sampling frequency in Hz
window_size = 4096
overlap = 2048

# Step 1: Define Room and Microphones
room_dim = [5, 4, 3]  # Room dimensions (in meters)
room = pra.ShoeBox(room_dim, fs=fs, max_order=3, absorption=0.4)

# Define microphone array (e.g., 3 microphones placed in a line)
mic_array = np.array([[1, 1.5, 1], [2, 1.5, 1], [3, 1.5, 1]]).T  # 3 microphones in 3D
room.add_microphone_array(pra.MicrophoneArray(mic_array, room.fs))

# Step 2: Add a Sound Source
# Define a simple sound source (e.g., a 1-second white noise signal)
duration = 1  # seconds
n_samples = int(fs * duration)
white_noise = np.random.randn(n_samples)  # Generate white noise

# Place the sound source in the room (e.g., at coordinates [2, 3, 1.5])
source_location = [2, 3, 1.5]
room.add_source(source_location, signal=white_noise)

# Step 3: Simulate the room acoustics
room.simulate()

# Step 4: Retrieve the microphone signals
mic_signals = room.mic_array.signals  # This is a 2D array (num_mics, num_samples)


# Step 5: GCC-PHAT to Estimate TDOA
def gcc_phat(sig1, sig2, fs):
    """Calculate time delay of arrival between two signals using GCC-PHAT."""
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.fft(sig1, n=n)
    SIG2 = np.fft.fft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    cc = np.fft.ifft(R / np.abs(R))
    max_shift = len(sig1)
    tdoa = np.argmax(np.abs(cc)) - max_shift // 2
    return tdoa / fs


# Step 6: Estimate Pairwise Distances
def estimate_distances(mic_signals, fs):
    num_mics = len(mic_signals)
    distances = np.zeros((num_mics, num_mics))

    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            tdoa = gcc_phat(mic_signals[i], mic_signals[j], fs)
            distance = np.abs(tdoa) * c  # Calculate distance in meters
            distances[i, j] = distance
            distances[j, i] = distance  # Symmetric matrix

    return distances


# Step 7: Multidimensional Scaling (MDS) for localization
def localize_mics(distances):
    mds = MDS(n_components=2, dissimilarity="precomputed")
    mic_positions = mds.fit_transform(distances)
    return mic_positions


# Step 8: Visualization
def plot_mic_positions(mic_positions):
    plt.scatter(mic_positions[:, 0], mic_positions[:, 1])
    plt.title("Estimated Microphone Positions")
    plt.xlabel("X coordinate (m)")
    plt.ylabel("Y coordinate (m)")
    plt.grid(True)
    plt.show()


# Run the example
distances = estimate_distances(mic_signals, fs)
mic_positions = localize_mics(distances)
plot_mic_positions(mic_positions)
