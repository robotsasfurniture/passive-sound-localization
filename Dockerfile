# Use ROS 2 Humble base image
FROM osrf/ros:humble-desktop

# Install additional packages if needed
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    python3-pip \
    portaudio19-dev \
    python-all-dev \
    && rm -rf /var/lib/apt/lists/* 

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip3 install poetry

# Install the dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root

# Create a workspace directory
RUN mkdir -p /root/ros2_ws/src

# Set the workspace directory as the working directory
WORKDIR /root/ros2_ws

# Copy the package from the host into the container (replace `my_python_package` with your package name)
# COPY ./passive_sound_localization /root/ros2_ws/src/passive_sound_localization
COPY ./packages/ /root/ros2_ws/src/

# Build the workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Source the workspace and launch the ROS 2 node
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch passive_sound_localization localization_launch.py"]
