import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def calculate_tdoa(microphones, source, speed_of_sound):
    """
    Calculate the Time Difference of Arrival (TDOA) for a given set of microphones and a sound source.
    """
    distances = np.linalg.norm(microphones - source, axis=1)
    toa = distances / speed_of_sound
    tdoa = toa - toa[0]
    return tdoa


def estimate_source(tdoa, microphones, speed_of_sound):
    """
    Estimate the sound source location given the TDOA and microphone positions.
    """

    def error_function(source_guess):
        estimated_tdoa = calculate_tdoa(microphones, source_guess, speed_of_sound)
        return np.sum((estimated_tdoa - tdoa) ** 2)

    initial_guess = np.mean(microphone_positions, axis=0)
    result = minimize(error_function, initial_guess)
    return result.x


def plot_microphones_and_source(microphones, actual_source, estimated_source):
    """
    Plot the positions of the microphones, the actual sound source, and the estimated sound source.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(
        microphones[:, 0], microphones[:, 1], c="blue", label="Microphones", s=100
    )
    plt.scatter(
        actual_source[0],
        actual_source[1],
        c="green",
        label="Actual Source",
        s=100,
        marker="x",
    )
    plt.scatter(
        estimated_source[0],
        estimated_source[1],
        c="red",
        label="Estimated Source",
        s=100,
        marker="o",
    )

    plt.title("Microphone Array and Sound Source")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Define the speed of sound in meters per second
speed_of_sound = 343  # m/s

# Microphone positions in 2D (x, y)
# TODO: Change the microphone positions to experiment with different configurations
microphone_positions = np.array([[-0.5, 0], [0, 0.5], [0.5, 0]])

# Generate random sound sources within the range (-2, -2) to (2, 2)
num_sources = 5
actual_sources = np.column_stack(
    (np.random.uniform(-2, 2, num_sources), np.random.uniform(0, 2, num_sources))
)

# Lists to store estimated sources and errors
estimated_sources = []
errors = []

for sound_source in actual_sources:
    # Calculate TDOA
    tdoa = calculate_tdoa(microphone_positions, sound_source, speed_of_sound)

    # Estimate the sound source location
    estimated_source = estimate_source(tdoa, microphone_positions, speed_of_sound)
    estimated_sources.append(estimated_source)

    # Calculate the error
    error = np.linalg.norm(sound_source - estimated_source)
    errors.append(error)

    print(
        f"Actual source: {sound_source}, Estimated source: {estimated_source}, Error: {error}"
    )

# Convert lists to numpy arrays for plotting
estimated_sources = np.array(estimated_sources)
errors = np.array(errors)

# Plot the microphones, actual sources, and estimated sources
plt.figure(figsize=(8, 8))
plt.scatter(
    microphone_positions[:, 0],
    microphone_positions[:, 1],
    c="blue",
    label="Microphones",
    s=100,
)
plt.scatter(
    actual_sources[:, 0],
    actual_sources[:, 1],
    c="green",
    label="Actual Sources",
    s=100,
    marker="x",
)
plt.scatter(
    estimated_sources[:, 0],
    estimated_sources[:, 1],
    c="red",
    label="Estimated Sources",
    s=100,
    marker="o",
)

# Draw lines connecting actual sources to estimated sources
for actual, estimated in zip(actual_sources, estimated_sources):
    plt.plot(
        [actual[0], estimated[0]],
        [actual[1], estimated[1]],
        "k--",  # black dashed line
        linewidth=1,
    )

plt.title("Microphone Array and Sound Sources")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)
plt.show()
