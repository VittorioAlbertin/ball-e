# BALL·E - Environmental Awareness Pipeline

This document outlines how BALL·E can understand a general view of its environment by combining low-level perception and high-level semantic reasoning. The system is modular and designed to grow over time.

---

## 1. Spatial Awareness (Where Things Are)

### Tools:
- **SLAM (Simultaneous Localization and Mapping)**: Builds a 2D or 3D map of the surroundings.
- **Occupancy Grid Mapping**: Differentiates between free and occupied space.
- **LiDAR** or **Stereo Vision + Depth**: Provides real-time distance and structure estimation.

### ROS Packages:
- `rtabmap_ros` or `ORB-SLAM3`: Visual SLAM
- `gmapping` or `cartographer`: 2D SLAM (LiDAR-based)
- `costmap_2d` or `octomap`: Environment modeling

BALL·E uses these tools to perceive and map its environment's physical structure.

---

## 2. Semantic Understanding (What Things Are)

### Object and Scene Understanding:
- **YOLO-World**: Real-time object detection using text prompts (open vocabulary).
- **CLIP** or **OWL-ViT**: Matches text queries to images or regions for reasoning.

### Scene Analysis:
- **Semantic Segmentation** (e.g., DeepLabV3, Segformer): Labels areas like walls, floors, furniture at the pixel level.
- **Semantic Mapping**: Fuse object/scene labels with SLAM map to build a "semantic map".

This enables BALL·E to recognize objects, people, and general room layout.

---

## 3. High-Level Reasoning (Why/How/Who)

With spatial and semantic data, BALL·E can perform contextual reasoning:

- **Who is here?** (Face recognition and memory)
- **What is in front of me?** (Object detection)
- **How can I get there?** (Path planning)
- **Why is someone waving?** (Gesture recognition)
- **Should I respond?** (Personality and emotional state engine)

Example scenario:
> BALL·E sees a person near the door. It recognizes "Alice", recalls she waved yesterday, and chooses to roll toward her and ask: “Hi Alice! Are you heading out again?”

---

## Environmental Awareness Pipeline (Minimal Setup)

| Layer                     | Functionality             | Tool/Package               |
|--------------------------|---------------------------|----------------------------|
| Depth/Structure           | Visual SLAM               | `rtabmap_ros`, `ORB-SLAM3` |
| Occupancy Mapping         | Navigation grid           | `nav2`, `move_base`        |
| Object Detection          | Real-time identification  | YOLOv8, YOLO-World         |
| Scene Summary             | Global reasoning          | CLIP, Segment Anything     |
| Face Recognition          | Person identification     | YOLO + face embeddings     |
| Behavior and Memory       | Interaction logic         | Custom logic + ChatGPT API |

---

## Optional Enhancements

- **Ambient Light Sensor**: Detect lighting changes and mood.
- **Sound Detection**: Identify footsteps, claps, or voices directionally.
- **Touch Sensor**: Detect bumps or taps.
- **Memory and Emotional Model**: Fuse environment data with past experiences to drive personality evolution.

---

## Next Steps

Would you like:
- A pipeline diagram?
- ROS node graph and topic layout?
- A minimal ROS workspace example?

Let me know how you'd like to proceed.
