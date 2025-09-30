# System Architecture

## Overview

BALL·E is divided into two main units:
- **Mobile Robot (BALL·E)**: Handles mobility, expressions, and local sensing.
- **Docking Station (Smart Base)**: Handles computation-heavy tasks like SLAM, AI, and cloud interaction.

## Architecture Components

### 1. Mobile Unit
- 3-wheel drive system
- Stereo camera
- Infrared sensor
- Microphone array, speaker
- Touch sensors
- Display for animated eyes
- Battery + wireless charging coil

### 2. Docking Station
- Jetson Nano or mini PC
- ROS2 master node
- Handles SLAM, GPT API, face recognition, etc.
- Controls remote behavior and stores memory

### 3. Communication
- Local Wi-Fi or mesh
- Robot sends real-time data; base sends behavior decisions

## Design Philosophy
- Robot handles real-time tasks with low latency
- Base handles intelligence, cloud, long-term memory
- Everything is modular and extensible
