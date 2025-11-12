# Ball-e Gazebo Classic Simulation

This package provides Gazebo Classic simulation environments for the Ball-e robot, including a home room with animated people for testing the perception and tracking pipeline.

## Overview

The simulation includes:
- **Ball-e robot model** with camera sensor and differential drive
- **Empty world** for basic testing
- **Home room world** with furniture and 3 animated people walking around
- **Direct ROS2 integration** (no bridge needed with Gazebo Classic)
- Camera streaming to `/camera/image_raw` (compatible with existing perception nodes)
- Robot movement control via `/cmd_vel`

## Prerequisites

Gazebo Classic is already included in `ros-humble-desktop-full`, so no additional installation is needed if you're using the Ball-e Docker container!

### Verify Installation

```bash
# Check Gazebo Classic is available
gazebo --version
# Should output: Gazebo multi-robot simulator, version 11.x.x

# Check ROS2 Gazebo packages
ros2 pkg list | grep gazebo_ros
# Should show: gazebo_ros, gazebo_plugins, gazebo_ros_pkgs
```

## Package Structure

```
ball_e_gazebo/
├── launch/
│   ├── empty_world.launch.py    # Basic empty world
│   └── home_sim.launch.py       # Home room with people
├── models/
│   └── (future custom models)
└── worlds/
    ├── empty.world              # Empty test world
    └── home_room.world          # Living room with furniture & people
```

## Building the Packages

```bash
# Navigate to workspace
cd /ball-e/ros2_ws

# Build both description and gazebo packages
colcon build --packages-select ball_e_description ball_e_gazebo

# Source the workspace
source install/setup.bash
```

## Launching the Simulation

### Option 1: Home Room Simulation (Recommended)

Launch the complete home environment with people:

```bash
source /ball-e/ros2_ws/install/setup.bash
ros2 launch ball_e_gazebo home_sim.launch.py
```

This will:
- ✅ Start Gazebo Classic with home room world
- ✅ Spawn Ball-e robot at origin
- ✅ Start 3 animated people walking around
- ✅ Publish camera feed to `/camera/image_raw`
- ✅ Enable robot control via `/cmd_vel`
- ✅ Enable simulation time (`use_sim_time:=true`)

### Option 2: Empty World (For Testing)

Launch with just an empty world:

```bash
ros2 launch ball_e_gazebo empty_world.launch.py
```

Use this for:
- Testing robot model
- Debugging sensors
- Quick iterations without distractions

## Verifying Camera Stream

After launching the simulation, check the camera is working:

```bash
# List available topics
ros2 topic list | grep camera
# Should show: /camera/image_raw, /camera/camera_info

# Check image publishing rate
ros2 topic hz /camera/image_raw
# Should show: ~30 Hz

# View camera feed
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

## Controlling the Robot

Move the robot around in the simulation:

```bash
# Move forward
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}" -r 10

# Rotate in place
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.5}}" -r 10

# Move forward while turning
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.3}}" -r 10

# Stop the robot
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{}" -1
```

You can also use keyboard teleop:

```bash
# Install teleop_twist_keyboard if needed
sudo apt install ros-humble-teleop-twist-keyboard

# Control with keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Running Perception Pipeline on Simulated Data

The camera topic `/camera/image_raw` is compatible with your existing perception nodes. Run the perception pipeline:

```bash
# Terminal 1: Launch simulation
ros2 launch ball_e_gazebo home_sim.launch.py

# Terminal 2: Launch perception pipeline
source /ball-e/ros2_ws/install/setup.bash
ros2 launch robot_bringup identification_pipeline_launch.py
```

Your YOLO detector, person tracker, and face recognition will process the simulated camera feed just like the real robot!

## World Details

### Home Room World

**Dimensions:** 8m × 6m × 3m living room

**Furniture:**
- Coffee table (center)
- Couch (south wall)
- 2 chairs (near table)
- Bookshelf (east wall)
- Side table

**People Actors:**
- **Person 1**: Walking along east wall (north-south)
- **Person 2**: Walking diagonally across room
- **Person 3**: Walking in a rectangular path near the couch

**Lighting:**
- Directional sunlight
- 2 ceiling point lights for ambient illumination

### Empty World

Basic ground plane with directional lighting. Good for:
- Robot model debugging
- Sensor testing
- Performance benchmarking

## Customization

### Modifying the Home Room

Edit the world file:

```bash
nano /ball-e/ros2_ws/src/ball_e_gazebo/worlds/home_room.world
```

**Add furniture:**
Add new `<model>` blocks with visual/collision geometry.

**Change room size:**
Modify the floor, ceiling, and wall dimensions.

**Adjust lighting:**
Change `<light>` parameters for different ambiance.

### Modifying Actor Behavior

Actors follow scripted trajectories. To change person movement:

```xml
<actor name="person_1">
  <script>
    <trajectory id="0" type="walking">
      <waypoint>
        <time>0.0</time>
        <pose>X Y Z Roll Pitch Yaw</pose>
      </waypoint>
      <!-- Add more waypoints -->
    </trajectory>
  </script>
</actor>
```

**Parameters:**
- `<time>`: Time in seconds to reach waypoint
- `<pose>`: Position (X Y Z) and orientation (Roll Pitch Yaw in radians)
- `<delay_start>`: Seconds before actor starts moving

### Adding More People

Copy an existing `<actor>` block and change:
- `name` (must be unique)
- Initial `<pose>`
- Trajectory waypoints
- `<delay_start>` (stagger start times)

### Changing Camera Parameters

Edit the camera sensor in the robot description:

```bash
nano /ball-e/ros2_ws/src/ball_e_description/urdf/camera.xacro
```

**Adjustable parameters:**
- `<update_rate>`: FPS (default: 30)
- `<horizontal_fov>`: Field of view in radians (default: 1.396 ≈ 80°)
- `<width>` / `<height>`: Image resolution (default: 640×480)
- `<near>` / `<far>`: Clip planes

## Troubleshooting

### Gazebo won't start

```bash
# Check if Gazebo Classic is installed
gazebo --version

# If not installed (unlikely in Docker):
sudo apt install gazebo ros-humble-gazebo-ros-pkgs
```

### Camera not publishing

```bash
# Check robot_description is published
ros2 topic echo /robot_description --once

# Check camera topic exists
ros2 topic list | grep camera

# Check for errors in Gazebo terminal
# Look for plugin loading errors
```

### Actors not visible

Gazebo may download actor meshes on first launch. This requires internet connection and may take a few minutes.

Check Gazebo terminal output for download progress or errors.

### Robot not spawning

```bash
# Check robot_state_publisher is running
ros2 node list | grep robot_state_publisher

# Manually spawn if needed
ros2 run gazebo_ros spawn_entity.py -topic robot_description -entity ball_e
```

### Simulation running slow

```bash
# Launch without GUI (headless mode)
gzserver worlds/home_room.world

# Reduce physics update rate (edit world file)
# Change <max_step_size> from 0.001 to 0.01
```

### Robot not responding to /cmd_vel

```bash
# Check differential drive plugin is loaded
ros2 topic list | grep cmd_vel

# Check for plugin errors in Gazebo terminal

# Try echoing odometry to verify plugin is active
ros2 topic echo /odom
```

## Integration with Real Robot

The simulation uses the **same ROS2 topics** as the real robot:

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | RGB camera feed |
| `/camera/camera_info` | sensor_msgs/CameraInfo | Camera calibration |
| `/cmd_vel` | geometry_msgs/Twist | Robot velocity commands |
| `/odom` | nav_msgs/Odometry | Wheel odometry |

To switch between simulation and real robot:

```bash
# Simulation
ros2 launch ball_e_gazebo home_sim.launch.py

# Real robot
ros2 launch robot_bringup camera_yolo_tracker_launch.py
```

Your perception nodes don't need any changes!

## ROS2 Topics Published

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | 30 Hz | Camera RGB stream |
| `/camera/camera_info` | sensor_msgs/CameraInfo | 30 Hz | Camera intrinsics |
| `/odom` | nav_msgs/Odometry | 50 Hz | Robot odometry |
| `/tf` | tf2_msgs/TFMessage | 50 Hz | Transform tree |
| `/robot_description` | std_msgs/String | Latched | URDF model |

## ROS2 Topics Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | geometry_msgs/Twist | Robot velocity commands |

## Future Enhancements

### Phase 2: Microphone Array
- Add microphone sensors to robot URDF
- Configure audio plugins
- Test audio source localization

### Phase 3: Advanced Navigation
- Add obstacle avoidance testing
- Multi-room environments
- Dynamic obstacles

### Phase 4: Advanced Environments
- Outdoor scenes
- Dynamic lighting conditions
- More complex actor behaviors

## Tips & Best Practices

1. **Always use simulation time** when running with Gazebo:
   - Launch files automatically set `use_sim_time:=true`
   - Your nodes should respect this parameter

2. **Start simple**: Test with empty world first, then progress to home room

3. **Monitor performance**: Check Gazebo terminal for real-time factor

4. **Save actor meshes**: After first download, actors load much faster

5. **Use RViz2 alongside Gazebo**: Visualize ROS2 topics while simulation runs

```bash
# Launch RViz2 with simulation
ros2 run rviz2 rviz2
```

## Comparison: Gazebo Classic vs Gazebo Fortress

This simulation uses **Gazebo Classic** (Gazebo 11) which is:

✅ **Already installed** with ros-humble-desktop-full
✅ **Well-documented** with mature ecosystem
✅ **Stable** and battle-tested
✅ **Simpler** - no bridge needed, plugins integrate directly
✅ **Compatible** with most ROS2 Humble tutorials

Gazebo Fortress (the newer version) offers better performance and modern architecture but requires additional installation and setup.

## Resources

- **Gazebo Classic Documentation**: http://classic.gazebosim.org/
- **Gazebo ROS2 Integration**: http://classic.gazebosim.org/tutorials?tut=ros2_overview
- **Actor Tutorial**: http://classic.gazebosim.org/tutorials?tut=actor
- **SDF Format**: http://sdformat.org/

## Support

For issues specific to this simulation:
1. Check this README's troubleshooting section
2. Verify all dependencies are installed
3. Check Gazebo terminal output for errors
4. Rebuild packages: `colcon build --packages-select ball_e_description ball_e_gazebo`

---

Happy simulating! 🤖
