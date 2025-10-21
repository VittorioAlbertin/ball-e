I'm refactoring Ball-e, a ROS2 Humble social robot running in Docker. Current setup:
- Camera node → YOLO detection → Face detection → Database matching
- Problem: Face recognition runs every frame (~1 FPS), causing bottlenecks
- Goal: Implement person tracking to decouple detection from identification

ARCHITECTURE VISION:
1. Detection Layer: Camera → YOLO → Person Tracker (assigns persistent IDs)
2. Recognition Layer: Face recognition triggered conditionally, not every frame
3. State Manager: Centralized person tracking (ID, identity, bbox, confidence, timestamps)
4. Coordinator: Decides when to trigger recognition based on tracking state

FUTURE INTEGRATION POINTS:
- Audio direction (microphone array) → merge with visual tracking
- Emotion modeling per tracked person
- TTS/STT interaction system
- Multi-frame identity persistence

CURRENT STACK:
- ROS2 Humble, Docker container
- Python nodes for camera, YOLO, face detection, database service
- RViz2 for visualization

Keep the architecture modular and extensible. Each node should have clear 
responsibilities and communicate via well-defined interfaces.