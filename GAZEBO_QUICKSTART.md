# Gazebo Classic Simulation - Quick Start Guide

## What Was Created

Two new ROS2 packages for Gazebo Classic simulation:

1. **`ball_e_description`** - Robot model (URDF/xacro)
   - Differential drive robot with camera
   - Movement control via /cmd_vel
   - Modular sensor design
   - Located: `ros2_ws/src/ball_e_description/`

2. **`ball_e_gazebo`** - Simulation worlds and launch files
   - Empty test world
   - Home room with furniture and 3 animated people
   - Located: `ros2_ws/src/ball_e_gazebo/`

## Prerequisites

✅ **Gazebo Classic is already installed** in the Docker container!

It comes with `ros-humble-desktop-full` which is part of your Dockerfile. No extra installation needed!

## Setup Steps (In Docker Container)

### 1. Verify Installation

```bash
# Check Gazebo Classic is available
gazebo --version
# Should show: Gazebo multi-robot simulator, version 11.x.x

# Check ROS2 Gazebo packages
ros2 pkg list | grep gazebo_ros
# Should show: gazebo_ros, gazebo_plugins, etc.
```

### 2. Build the Packages

```bash
cd /ball-e/ros2_ws

# Build both new packages
colcon build --packages-select ball_e_description ball_e_gazebo

# Source the workspace
source install/setup.bash
```

### 3. Launch the Simulation

**Option A: Home Room with People (Recommended)**

```bash
source /ball-e/ros2_ws/install/setup.bash
ros2 launch ball_e_gazebo home_sim.launch.py
```

**Option B: Empty World (For Testing)**

```bash
ros2 launch ball_e_gazebo empty_world.launch.py
```

### 4. Verify Camera Stream

```bash
# In a new terminal
source /ball-e/ros2_ws/install/setup.bash

# Check camera topic exists
ros2 topic list | grep camera

# Check publishing rate
ros2 topic hz /camera/image_raw

# View camera feed
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

### 5. Control the Robot

```bash
# Move forward
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}" -r 10

# Rotate in place
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.5}}" -r 10

# Move forward and turn
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.3}}" -r 10

# Stop the robot
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{}" -1
```

**Use keyboard teleop for easier control:**

```bash
sudo apt install ros-humble-teleop-twist-keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### 6. Run Perception Pipeline on Simulation

```bash
# Terminal 1: Simulation
ros2 launch ball_e_gazebo home_sim.launch.py

# Terminal 2: Perception pipeline
source /ball-e/ros2_ws/install/setup.bash
ros2 launch robot_bringup identification_pipeline_launch.py
```

Your YOLO, person tracker, and face recognition will now process simulated people!

## What You Get

✅ **Gazebo Classic** with realistic home room environment
✅ **Ball-e robot** with working camera sensor and movement
✅ **3 animated people** walking around the room
✅ **Furniture**: couch, coffee table, chairs, bookshelf
✅ **Camera streaming** to `/camera/image_raw` (same topic as real camera)
✅ **Robot control** via `/cmd_vel` topic
✅ **Odometry** published to `/odom`
✅ **Compatible** with your existing perception pipeline
✅ **No extra installation required** - works out of the box!

## Home Room Layout

- **Size**: 8m × 6m × 3m living room
- **Furniture**: Couch, coffee table, 2 chairs, bookshelf, side table
- **Lighting**: Directional sun + 2 ceiling lights
- **People**: 3 actors with walking animations on different paths

## Complete Demo Workflow

```bash
# Terminal 1: Launch simulation
ros2 launch ball_e_gazebo home_sim.launch.py

# Terminal 2: Control robot (move it around the room)
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 3: Run perception to detect people
ros2 launch robot_bringup identification_pipeline_launch.py

# Terminal 4: View detection results
ros2 run rqt_image_view rqt_image_view /visualization/annotated_image
```

Drive the robot around and watch it detect the simulated people!

## ROS2 Topics

**Published by simulation:**
- `/camera/image_raw` - Camera feed (30 Hz)
- `/camera/camera_info` - Camera calibration
- `/odom` - Robot odometry (50 Hz)
- `/tf` - Transform tree (odom→base_footprint)

**Subscribed by simulation:**
- `/cmd_vel` - Robot velocity commands

## Troubleshooting

### If Gazebo won't start:

```bash
# Check if installed
gazebo --version

# Should already be there, but if missing:
sudo apt install gazebo ros-humble-gazebo-ros-pkgs
```

### If camera not publishing:

```bash
# Check robot spawned correctly
ros2 topic list | grep camera

# Check for plugin errors in Gazebo terminal
# Look for lines mentioning "libgazebo_ros_camera.so"
```

### If actors (people) not visible:

First launch requires internet to download actor meshes. Wait a few minutes and check Gazebo terminal for download progress.

### If robot not moving:

```bash
# Check differential drive plugin loaded
ros2 topic echo /odom

# If odom not publishing, check Gazebo terminal for errors
# Look for lines mentioning "libgazebo_ros_diff_drive.so"
```

### Build errors:

```bash
# Make sure dependencies are installed
sudo apt install ros-humble-xacro ros-humble-gazebo-ros-pkgs

# Clean and rebuild
cd /ball-e/ros2_ws
rm -rf build/ install/ log/
colcon build --packages-select ball_e_description ball_e_gazebo
```

## File Locations

```
ros2_ws/src/
├── ball_e_description/
│   ├── urdf/
│   │   ├── ball_e.urdf.xacro       # Main robot with diff drive
│   │   └── camera.xacro            # Camera sensor
│   ├── CMakeLists.txt
│   ├── package.xml
│   └── README.md
│
└── ball_e_gazebo/
    ├── worlds/
    │   ├── empty.world             # Empty test world
    │   └── home_room.world         # Room with furniture & people
    ├── launch/
    │   ├── empty_world.launch.py   # Empty world launcher
    │   └── home_sim.launch.py      # Home room launcher
    ├── CMakeLists.txt
    ├── package.xml
    └── README.md
```

## Customization

### Change number of people:

Edit: `ros2_ws/src/ball_e_gazebo/worlds/home_room.world`

Copy an `<actor>` block and modify the trajectory waypoints.

### Change camera resolution/FOV:

Edit: `ros2_ws/src/ball_e_description/urdf/camera.xacro`

Modify `<width>`, `<height>`, or `<horizontal_fov>`.

### Change robot speed limits:

Edit: `ros2_ws/src/ball_e_description/urdf/ball_e.urdf.xacro`

Find the differential drive plugin section and adjust:
- `<max_wheel_torque>` - Maximum wheel power
- `<max_wheel_acceleration>` - How fast it can accelerate

### Add furniture:

Edit: `ros2_ws/src/ball_e_gazebo/worlds/home_room.world`

Add new `<model>` blocks with visual/collision geometry.

## Why Gazebo Classic?

We chose Gazebo Classic over Gazebo Fortress because:

✅ **Zero setup** - Already included with ros-humble-desktop-full
✅ **No bridge needed** - Plugins publish directly to ROS2 topics
✅ **Mature & stable** - Battle-tested in production
✅ **Simple architecture** - Easier to understand and debug
✅ **Better documentation** - Years of tutorials and examples

## Next Steps

### Test Your Perception Pipeline

1. Launch the simulation
2. Drive the robot around with keyboard teleop
3. Watch YOLO detect the walking people
4. Verify person tracking assigns consistent IDs
5. Test face recognition (even though simulated faces are simple)

### Experiment with Robot Control

```bash
# Try different movement patterns
# Forward for 2 seconds
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}" -r 10 &
sleep 2 && ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{}" -1

# Circle pattern
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.5}}" -r 10
```

### Monitor Robot State

```bash
# Watch odometry
ros2 topic echo /odom

# View TF tree
ros2 run rqt_tf_tree rqt_tf_tree

# Check transform from odom to camera
ros2 run tf2_ros tf2_echo odom camera_optical_frame
```

## Phase 2 Plans (Future)

### Microphone Array
- Add 4-microphone array to robot
- Test audio source localization in simulation

### Advanced Navigation
- Add obstacle avoidance
- Test path planning algorithms
- Multi-room environments

## Resources

- **Detailed docs**: See `ros2_ws/src/ball_e_gazebo/README.md`
- **Robot description**: See `ros2_ws/src/ball_e_description/README.md`
- **Full summary**: See `ros2_ws/src/SIMULATION_SUMMARY.md`
- **Gazebo Classic docs**: http://classic.gazebosim.org/
- **Gazebo ROS2 integration**: http://classic.gazebosim.org/tutorials?tut=ros2_overview

---

## Summary

You now have a **complete Gazebo Classic simulation** that:
1. Simulates your Ball-e robot with camera and movement
2. Provides a realistic home environment with people
3. Uses the same ROS2 topics as your real robot
4. Requires zero extra installation (included in Docker)
5. Lets you test perception while driving around

**Launch it and test your pipeline!** 🤖

```bash
# Quick start command
cd /ball-e/ros2_ws && \
source install/setup.bash && \
ros2 launch ball_e_gazebo home_sim.launch.py
```
