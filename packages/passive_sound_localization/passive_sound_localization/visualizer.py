import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.animation import FuncAnimation

import numpy as np
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, microphone_positions: List[Tuple[float, float]]):
        """
        Initialize the visualizer with microphone positions and target position.

        :param microphone_positions: List of tuples representing microphone coordinates [(x1, y1), (x2, y2), ...]
        """
        # Convert from 3D to 2D by dropping the z-coordinate
        self.microphone_positions = np.array(microphone_positions)
        self.fig, self.ax = plt.subplots()
        plt.ion()

        self.continue_execution = True
        self.grid_points = []

    def plot(
        self,
        angle: float,
        distance: float,
        selected_grid_point: Tuple[float, float],
        average_point: Tuple[float, float],
    ):
        """
        Plot the positions of the microphones and the target in real-time.
        """

        self.ax.clear()  # Clear previous plots

        # Plot grid points if available
        if len(self.grid_points) > 0:
            grid_points = np.array(self.grid_points)
            self.ax.scatter(
                grid_points[:, 0],
                grid_points[:, 1],
                c="blue",
                label="Grid Points",
                s=1,
            )

        # Plot microphone positions
        self.ax.scatter(
            self.microphone_positions[:, 0],
            self.microphone_positions[:, 1],
            c="green",
            label="Microphones",
        )

        # Plot the selected grid point
        self.ax.scatter(
            selected_grid_point[0],
            selected_grid_point[1],
            c="red",
            label="Selected Grid Point",
        )

        # Plot the average point
        self.ax.scatter(
            average_point[0],
            average_point[1],
            c="purple",
            label="Average Point",
        )

        # Draw a line to represent the angle and distance
        # x_line = self.microphone_positions[0, 0] + distance * np.cos(angle)
        # y_line = self.microphone_positions[0, 1] + distance * np.sin(angle)
        # self.ax.plot(
        #     [self.microphone_positions[0, 0], x_line],
        #     [self.microphone_positions[0, 1], y_line],
        #     c="red",
        # )

        x_line = [self.microphone_positions[0, 0], selected_grid_point[0]]
        y_line = [self.microphone_positions[0, 1], selected_grid_point[1]]
        self.ax.plot(x_line, y_line, c="red")

        # Add labels, title, and legend
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title(f"Angle: {angle:.2f}, Distance: {distance:.2f}")
        self.ax.legend()

        # Update the plot
        plt.draw()
        plt.pause(0.1)  # Pause briefly to refresh the plot

    def set_grid_points(self, grid_points: List[Tuple[float, float]]):
        """
        Plot a grid of points.
        """
        self.grid_points = grid_points

    def open_loading_screen(self):
        """
        Open a loading screen plot without pausing execution.
        """
        self.ax.clear()
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title("Loading... Please wait.")

        # Display a loading message
        self.ax.text(
            0.5,
            0.5,
            "Loading...",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
            fontsize=20,
            color="gray",
        )

        # Draw the plot without blocking execution
        plt.draw()
        plt.pause(0.1)


class AudioVisualizer:
    def __init__(self, sampling_rate):
        """
        Initialize the MicrophonePlotter class.

        Args:
            sampling_rate (int): Sampling rate in Hz (e.g., 44100).
        """
        self.sampling_rate = sampling_rate
        self.fig, self.axs = None, None  # For managing the plot
        self.microphone_data = []  # Store microphone data lists

    def plot(self, mic_data_list):
        """
        Plot the microphone data.

        Args:
            mic_data_list (list of np.ndarray): List of int16 arrays for microphone data.
        """
        # Check if new microphones need to be added
        num_mics = len(mic_data_list)
        if len(self.microphone_data) < num_mics:
            self.microphone_data.extend(
                [[] for _ in range(num_mics - len(self.microphone_data))]
            )

        # Append new data to existing data
        for i, data in enumerate(mic_data_list):
            self.microphone_data[i] = np.concatenate((self.microphone_data[i], data))[
                -2000:
            ]

        # Create or update the plot
        if self.fig is None or self.axs is None or len(self.axs) != num_mics:
            self.fig, self.axs = plt.subplots(
                num_mics, 1, figsize=(10, 3 * num_mics), sharex=True
            )
            if num_mics == 1:
                self.axs = [self.axs]

        time = [
            np.linspace(
                0,
                len(self.microphone_data[i]) / self.sampling_rate,
                len(self.microphone_data[i]),
            )
            for i in range(num_mics)
        ]

        for i, ax in enumerate(self.axs):
            ax.clear()
            ax.plot(time[i], self.microphone_data[i], label=f"Microphone {i + 1}")
            ax.set_title(f"Microphone {i + 1}")
            ax.set_ylabel("Amplitude")
            ax.set_ylim(-2000, 2000)
            ax.legend(loc="upper right")

        self.axs[-1].set_xlabel("Time (s)")

        plt.pause(0.1)  # Allow the plot to update dynamically

    def clear(self):
        """Clear all stored data and reset the plot."""
        self.microphone_data = []
        if self.fig:
            plt.close(self.fig)
            self.fig, self.axs = None, None


# Example usage
if __name__ == "__main__":
    mic_positions = [(1, 2), (3, 4), (5, 6)]
    target_pos = (4, 5)
    visualizer = Visualizer(mic_positions)
    visualizer.open_loading_screen()
    visualizer.plot(target_position=target_pos)
