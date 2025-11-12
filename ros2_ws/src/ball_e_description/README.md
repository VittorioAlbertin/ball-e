# Ball-e Robot Description

This package contains the URDF/xacro files that describe the Ball-e robot's physical structure, sensors, and properties.

## Overview

The robot model includes:
- **Differential drive base** (two wheels + caster)
- **Camera sensor** with Gazebo Classic plugins
- **Differential drive controller** for robot movement
- **Modular design** using xacro macros
- **Proper coordinate frames** following ROS conventions

## Package Structure

```
ball_e_description/
├── urdf/
│   ├── ball_e.urdf.xacro    # Main robot description
│   └── camera.xacro         # Camera sensor macro
├── meshes/                  # 3D models (future)
└── launch/                  # (future launch files)
```

## Robot Components

### Base

- **Type**: Cylindrical body (0.15m radius, 0.1m height)
- **Mass**: 2.0 kg
- **Color**: Blue

### Wheels

- **Type**: Differential drive (2 driven wheels + 1 caster)
- **Wheel radius**: 0.033m
- **Wheel separation**: 0.16m (center to center)
- **Control**: Differential drive plugin for velocity commands via `/cmd_vel`

### Camera

- **Resolution**: 640×480
- **Frame rate**: 30 Hz
- **FOV**: 80° horizontal
- **Position**: Mounted on top of robot (0.25m above base)
- **Optical frame**: Follows ROS camera conventions (X forward, Y left, Z up)
- **Plugin**: Gazebo Classic camera plugin (`libgazebo_ros_camera.so`)

### Movement Control

- **Plugin**: Differential drive controller (`libgazebo_ros_diff_drive.so`)
- **Subscribes to**: `/cmd_vel` (geometry_msgs/Twist)
- **Publishes**: `/odom` (nav_msgs/Odometry)
- **Max wheel torque**: 20 Nm
- **Odometry**: Published with TF transforms

## Coordinate Frames

The robot follows standard ROS conventions:

```
odom (published by diff drive)
    └── base_footprint (on ground)
        └── base_link (robot center)
            ├── left_wheel
            ├── right_wheel
            ├── caster_wheel
            └── camera_mount
                └── camera_link
                    └── camera_optical_frame
```

## Usage

### View Robot Model

```bash
# Install urdf_tutorial if not already installed
sudo apt install ros-humble-urdf-tutorial

# Launch URDF viewer
ros2 launch urdf_tutorial display.launch.py model:=$(ros2 pkg prefix ball_e_description)/share/ball_e_description/urdf/ball_e.urdf.xacro
```

### Use in Simulation

The robot description is automatically used by the `ball_e_gazebo` launch files. See the `ball_e_gazebo` package for simulation instructions.

### Control the Robot

When launched in Gazebo, control the robot with:

```bash
# Move forward
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}" -r 10

# Rotate in place
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.5}}" -r 10

# Move forward and turn
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.3}}" -r 10

# Stop
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{}" -1
```

### Check URDF Syntax

```bash
# Process xacro and check for errors
xacro $(ros2 pkg prefix ball_e_description)/share/ball_e_description/urdf/ball_e.urdf.xacro

# Check URDF structure
ros2 run urdf_parser check_urdf ball_e.urdf
```

## Customization

### Changing Camera Position

Edit `ball_e.urdf.xacro`:

```xml
<!-- Find this line: -->
<xacro:camera_sensor
  parent="camera_mount"
  name="camera"
  xyz="0 0 0.15"    <!-- Change these values -->
  rpy="0 0 0"/>      <!-- Roll, Pitch, Yaw -->
```

### Changing Camera Parameters

Edit `camera.xacro`:

```xml
<camera name="${name}">
  <horizontal_fov>1.3962634</horizontal_fov>  <!-- FOV in radians -->
  <image>
    <width>640</width>    <!-- Image width -->
    <height>480</height>  <!-- Image height -->
    <format>R8G8B8</format>
  </image>
</camera>
<update_rate>30.0</update_rate>  <!-- FPS -->
```

### Adjusting Drive Parameters

Edit `ball_e.urdf.xacro` in the differential drive plugin section:

```xml
<plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
  <!-- Kinematics -->
  <wheel_separation>0.16</wheel_separation>
  <wheel_diameter>0.066</wheel_diameter>

  <!-- Limits -->
  <max_wheel_torque>20</max_wheel_torque>        <!-- Adjust for more power -->
  <max_wheel_acceleration>1.0</max_wheel_acceleration>
</plugin>
```

### Adding Sensors (Future)

To add a new sensor:

1. Create a xacro macro file (e.g., `microphone.xacro`)
2. Include it in `ball_e.urdf.xacro`
3. Instantiate the macro with appropriate parameters

## Published Topics

When running in Gazebo Classic:

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | RGB camera feed |
| `/camera/camera_info` | sensor_msgs/CameraInfo | Camera calibration |
| `/odom` | nav_msgs/Odometry | Wheel odometry |
| `/tf` | tf2_msgs/TFMessage | Transform tree (odom→base_footprint) |

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | geometry_msgs/Twist | Velocity commands for robot |

## Future Enhancements

### Phase 2: Microphone Array
- Add `microphone.xacro` with audio sensor
- Configure 4-mic array for sound localization

### Phase 3: Additional Sensors
- IMU (orientation and acceleration)
- Depth camera (for navigation)
- Touch sensors

## Technical Notes

### Inertial Properties

All links have proper inertial properties (mass, inertia tensors) required for physics simulation. These are calculated based on simple geometric shapes.

### Gazebo Classic Plugins

The robot uses standard Gazebo Classic ROS plugins:
- **Camera**: `libgazebo_ros_camera.so` - Publishes images directly to ROS2 topics
- **Differential Drive**: `libgazebo_ros_diff_drive.so` - Provides wheel control and odometry

### Xacro Benefits

Using xacro provides:
- **Modularity**: Sensors defined in separate files
- **Reusability**: Camera macro can be instantiated multiple times
- **Parameterization**: Easy to adjust properties
- **Maintainability**: Changes in one place affect all instances

## Resources

- **URDF Tutorial**: http://wiki.ros.org/urdf/Tutorials
- **Xacro Documentation**: http://wiki.ros.org/xacro
- **Gazebo Classic Plugins**: http://classic.gazebosim.org/tutorials?tut=ros_gzplugins

---

**Note**: This robot description is designed to work with Gazebo Classic simulation and can be adapted for future real hardware integration.
