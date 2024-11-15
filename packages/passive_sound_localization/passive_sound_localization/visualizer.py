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


# Example usage
if __name__ == "__main__":
    mic_positions = [(1, 2), (3, 4), (5, 6)]
    target_pos = (4, 5)
    visualizer = Visualizer(mic_positions)
    visualizer.open_loading_screen()
    visualizer.plot(target_position=target_pos)
