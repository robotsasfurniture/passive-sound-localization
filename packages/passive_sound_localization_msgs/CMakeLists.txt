cmake_minimum_required(VERSION 3.5)
project(passive_sound_localization_msgs)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate ROS interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LocalizationResult.msg"
  DEPENDENCIES std_msgs
)

ament_package()