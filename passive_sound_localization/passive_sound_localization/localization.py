from typing import List
from passive_sound_localization.models.configs.localization import LocalizationConfig
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LocalizationResult:
    distance: float  # Estimated distance to the sound source
    angle: float  # Estimated angle to the sound source in degrees

# TODO: Take in multi-channel audio data of bytes
# TODO: Make sure all functions are compatible
class SoundLocalizer:
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.mic_positions = np.array(
            config.mic_positions, dtype=np.float32
        )  # Get mic positions from config
        self.speed_of_sound = config.speed_of_sound
        self.mic_distance = config.mic_distance
        self.sample_rate = config.sample_rate
        self.fft_size = config.fft_size
        self.angle_resolution = config.angle_resolution
        self.num_mics = None  # To be set when data is received

        # Generate circular plane of grid points for direction searching
        self.grid_points = self._generate_circular_grid()

        # Precompute delays and phase shifts
        self.delays = self._compute_all_delays()
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)
        self.phase_shifts = self._compute_all_phase_shifts(self.freqs)
    
    def localize(
        self, multi_channel_data: List[bytes], num_sources: int = 1
    ) -> List[LocalizationResult]:
        """
        Performs sound source localization for both single-source and multi-source cases using a
        frequency-domain steered beamformer.

        Parameters:
        - multi_channel_data: List of numpy arrays containing audio data from each microphone.
        - num_sources: The number of sound sources to localize (default is 1 for single-source localization).

        Returns:
        - List of LocalizationResult objects, each containing the estimated distance and angle of a localized source.
        """
        logger.info(
            f"Performing sound source localization for {num_sources} source(s)."
        )

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
        cross_spectrum = self._compute_cross_spectrum(data, fft_size=self.fft_size)

        # List to hold localization results for each source
        results = []

        # Iteratively localize each source
        for _ in range(num_sources):
            # best_direction = self._search_best_direction(cross_spectrum, grid_points)
            best_direction, estimated_distance, source_idx = self._search_best_direction(cross_spectrum)
            if best_direction is not None:
                # Convert direction into an angle for the result
                estimated_angle = np.degrees(best_direction[0])

                # Append the localization result for the source
                results.append(
                    LocalizationResult(
                        distance=estimated_distance, angle=estimated_angle
                    )
                )

                # Remove the contribution of the localized source to find the next source
                cross_spectrum = self._remove_source_contribution(cross_spectrum, source_idx)

        logger.info(f"Localization complete: {len(results)} source(s) found.")
        return results

    def _compute_cross_spectrum(self, mic_signals, fft_size=1024):
        """Compute the cross-power spectrum between microphone pairs."""
        num_mics = mic_signals.shape[0]

        # Correct shape: (num_mics, num_mics, fft_size // 2 + 1) for the rfft result
        cross_spectrum = np.zeros(
            (num_mics, num_mics, fft_size // 2 + 1), dtype=np.complex64
        )

        # Compute the FFT of each microphone signal
        mic_fft = np.fft.rfft(mic_signals, fft_size)

        # Compute the cross-power spectrum for each microphone pair
        for i in range(num_mics):
            for j in range(i, num_mics):
                cross_spectrum[i, j] = mic_fft[i] * np.conj(mic_fft[j])
                cross_spectrum[j, i] = np.conj(cross_spectrum[i, j])

        return cross_spectrum
    
    def _generate_circular_grid(self, radius=1.0, num_points_radial=50, num_points_angular=360):
        """Generate a grid of points on a circular plane, optimized for speed."""
        # Create radial distances from 0 to the specified radius
        r = np.linspace(0, radius, num_points_radial, dtype=np.float32)
        
        # Create angular values from 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, num_points_angular, dtype=np.float32)
        
        # Compute x and y directly using broadcasting without creating a meshgrid
        r = r[:, np.newaxis]  # Convert r to column vector for broadcasting
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Return the points stacked as (x, y) pairs
        return np.column_stack((x.ravel(), y.ravel()))
    
    def _search_best_direction(self, cross_spectrum):
        """Search the circular grid for the direction with maximum beamformer output."""
        energies = self._compute_beamformer_energies(cross_spectrum)
        best_direction_idx = np.argmax(energies)
        best_direction = self.grid_points[best_direction_idx]
        return best_direction, best_direction_idx
    

    def _compute_all_phase_shifts(self, freqs):
        """
        Precompute phase shifts for all grid points, microphone pairs, and frequency bins.
        """
        # Compute tau (time delays between microphone pairs for all grid points)
        tau = self.delays[:, :, np.newaxis] - self.delays[:, np.newaxis, :]  # Shape: (num_grid_points, num_mics, num_mics)

        # Compute phase shifts for all frequencies
        phase_shifts = np.exp(-1j * 2 * np.pi * tau[:, :, :, np.newaxis] * freqs[np.newaxis, np.newaxis, np.newaxis, :]) # Shape: (num_grid_points, num_mics, num_mics, num_freq_bins)
        return phase_shifts
    
    def _compute_all_delays(self):
        """
        Precompute delays for all grid points and microphones.
        """
        # Compute distances from each microphone to each source position
        mic_positions_2d = self.mic_positions[:, :2]  # Shape: (num_mics, 2)
        # source_positions = self.source_positions  # Shape: (num_grid_points, 2)

        # Calculate distances between microphones and source positions
        distances = np.linalg.norm(
            mic_positions_2d[np.newaxis, :, :] - self.grid_points[:, np.newaxis, :],
            axis=2
        )  # Shape: (num_grid_points, num_mics)

        # Compute delays: distances divided by speed of sound
        delays = distances / self.speed_of_sound  # Shape: (num_grid_points, num_mics)

        return delays
    
    def _compute_beamformer_energies(self, cross_spectrum):
        """Compute the beamformer energy given the cross-spectrum and delays."""
        cross_spectrum_expanded = cross_spectrum[np.newaxis, :, :, :]
        # Multiply and sum over mics and frequency bins
        product = cross_spectrum_expanded * self.phase_shifts  # Shape: (num_grid_points, num_mics, num_mics, num_freq_bins)
        energies = np.abs(np.sum(product, axis=(1, 2, 3)))  # Shape: (num_grid_points,)
        return energies

    
    def _remove_source_contribution(self, cross_spectrum, source_idx):
        """
        Remove the contribution of a localized source using vectorized operations.
        """
        # Get the phase shifts for the localized source
        phase_shift = self.phase_shifts[source_idx]  # Shape: (num_mics, num_mics, num_freq_bins)

        # Subtract the contribution from the cross-spectrum
        cross_spectrum -= phase_shift
        return cross_spectrum

    def computer_cartesian_coordinates(self, distance, angle):
        """
        Compute the Cartesian coordinates of a sound source given its distance and angle.
        """
        x = distance * np.cos(np.radians(angle))
        y = distance * np.sin(np.radians(angle))
        return x, y
