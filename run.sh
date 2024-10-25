git pull
cd ../
cp -rf passive-sound-localization/passive_sound_localization ros2_ws/src/
cd ros2_ws/src
colcon build --packages-select passive_sound_localization
source install/setup.bash
ros2 launch passive_sound_localization passive_sound_localization.launch.py
