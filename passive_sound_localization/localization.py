import numpy as np
from config.localization_config import LocalizationConfig
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class LocalizationResult:
    distance: float  # Estimated distance to the sound source
    angle: float     # Estimated angle to the sound source in degrees

class SoundLocalizer:
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.speed_of_sound = config.speed_of_sound
        self.mic_distance = config.mic_distance
        self.sample_rate = config.sample_rate
        self.fft_size = config.fft_size
        self.angle_resolution = config.angle_resolution
        self.num_mics = None  # To be set when data is received

    def localize(self, multi_channel_data: list, sample_rate: int, num_sources: int = 1) -> list:
        """
        Performs sound source localization for both single-source and multi-source cases using a 
        frequency-domain steered beamformer.

        Parameters:
        - multi_channel_data: List of numpy arrays containing audio data from each microphone.
        - sample_rate: The sample rate of the audio data.
        - num_sources: The number of sound sources to localize (default is 1 for single-source localization).

        Returns:
        - List of LocalizationResult objects, each containing the estimated distance and angle of a localized source.
        """
        logger.info(f"Performing sound source localization for {num_sources} source(s).")

        self.num_mics = len(multi_channel_data)
        if self.num_mics < 2:
            logger.error("At least two microphones are required for localization.")
            raise ValueError("Insufficient number of microphones for localization.")

        # Ensure all channels have the same length
        min_length = min(len(data) for data in multi_channel_data)
        multi_channel_data = [data[:min_length] for data in multi_channel_data]

        # Stack the multi-channel data into a 2D array (num_mics x num_samples)
        data = np.vstack(multi_channel_data)

        # Compute the cross-power spectrum of the signals
        cross_spectrum = self.compute_cross_spectrum(data, fft_size=self.fft_size)

        # Generate spherical grid points for direction searching
        grid_points = self.generate_spherical_grid()

        # List to hold localization results for each source
        results = []

        # Iteratively localize each source
        for _ in range(num_sources):
            best_direction = self.search_best_direction(cross_spectrum, grid_points)
            if best_direction is not None:
                # Convert direction into an angle for the result
                estimated_angle = np.degrees(best_direction[0])
                estimated_distance = 1.0  # Placeholder value for distance (can be enhanced later)

                # Append the localization result for the source
                results.append(LocalizationResult(distance=estimated_distance, angle=estimated_angle))

                # Remove the contribution of the localized source to find the next source
                delays = self.compute_delays(best_direction)
                cross_spectrum = self.remove_source_contribution(cross_spectrum, delays)

        logger.info(f"Localization complete: {len(results)} source(s) found.")
        return results

    def compute_cross_spectrum(self, mic_signals, fft_size=1024):
        """Compute the cross-power spectrum between microphone pairs."""
        num_mics = mic_signals.shape[0]
        cross_spectrum = np.zeros((num_mics, num_mics, fft_size), dtype=np.complex)

        # Compute the FFT of each microphone signal
        mic_fft = np.fft.rfft(mic_signals, fft_size)

        # Compute the cross-power spectrum for each microphone pair
        for i in range(num_mics):
            for j in range(i, num_mics):
                cross_spectrum[i, j] = mic_fft[i] * np.conj(mic_fft[j])
                cross_spectrum[j, i] = np.conj(cross_spectrum[i, j])

        return cross_spectrum

    def generate_spherical_grid(self, num_points=2562):
        """Generate a grid of points on the surface of a sphere."""
        phi = np.linspace(0, 2 * np.pi, num_points)
        theta = np.linspace(0, np.pi, num_points)
        theta, phi = np.meshgrid(theta, phi)
        return np.stack([theta.ravel(), phi.ravel()], axis=1)

    def search_best_direction(self, cross_spectrum, grid_points):
        """Search the spherical grid for the direction with maximum beamformer output."""
        best_direction = None
        max_energy = 0

        for direction in grid_points:
            delays = self.compute_delays(direction)
            energy = self.compute_beamformer_energy(cross_spectrum, delays)
            if energy > max_energy:
                max_energy = energy
                best_direction = direction

        return best_direction

    def compute_delays(self, direction):
        """
        Compute time delays for each microphone given a direction on the spherical grid.
        """
        unit_vector = np.array([np.sin(direction[0]) * np.cos(direction[1]),
                                np.sin(direction[0]) * np.sin(direction[1]),
                                np.cos(direction[0])])
        delays = np.dot(self.mic_positions, unit_vector) / self.speed_of_sound
        return delays

    def compute_beamformer_energy(self, cross_spectrum, delays):
        """Compute the beamformer energy given the cross-spectrum and delays."""
        num_mics = cross_spectrum.shape[0]
        energy = 0
        for i in range(num_mics):
            for j in range(i, num_mics):
                tau = delays[i] - delays[j]
                energy += np.sum(cross_spectrum[i, j] * np.exp(1j * 2 * np.pi * tau))
        return np.abs(energy)

    def remove_source_contribution(self, cross_spectrum, delays):
        """Remove the contribution of a localized source."""
        num_mics = cross_spectrum.shape[0]
        for i in range(num_mics):
            for j in range(i, num_mics):
                tau = delays[i] - delays[j]
                cross_spectrum[i, j] -= np.exp(1j * 2 * np.pi * tau)
        return cross_spectrum