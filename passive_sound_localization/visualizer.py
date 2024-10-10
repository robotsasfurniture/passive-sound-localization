import matplotlib.pyplot as plt

from passive_sound_localization.config.visualizer_config import VisualizerConfig


class Visualizer:
    def __init__(self, config: VisualizerConfig):
        """
        Initialize the visualizer with microphone positions and target position.

        :param microphone_positions: List of tuples representing microphone coordinates [(x1, y1), (x2, y2), ...]
        """
        # Convert from 3D to 2D by dropping the z-coordinate
        self.microphone_positions = [(x, y) for x, y, z in config.microphone_positions]
        self.fig, self.ax = plt.subplots()
        self.continue_execution = config.continue_execution

    def plot(self, target_position):
        """
        Plot the positions of the microphones and the target.
        """
        self.ax.clear()

        # Plot microphones
        mic_x, mic_y = zip(*self.microphone_positions)
        self.ax.scatter(mic_x, mic_y, c="blue", label="Microphones")

        # Plot target
        target_x, target_y = target_position
        self.ax.scatter(target_x, target_y, c="red", label="Target")

        # Annotate microphones
        for i, (x, y) in enumerate(self.microphone_positions):
            self.ax.annotate(
                f"Mic {i+1}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # Annotate target
        self.ax.annotate(
            "Target",
            (target_x, target_y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        # Set labels and title
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title("Microphone and Target Positions")
        self.ax.legend()

        plt.draw()

        if self.continue_execution:
            plt.pause(0.1)

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
