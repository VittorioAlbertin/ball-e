# Person Enrollment Guide

## Quick Start

This guide explains how to enroll new people into Ball-e's identification system using the interactive enrollment CLI.

## Prerequisites

1. Camera connected and functional
2. Microphone connected (for voice enrollment)
3. ROS 2 workspace built and sourced

```bash
cd ~/Documents/ball-e/ros2_ws
colcon build --packages-select msgs_interfaces sensors_pkg perception_pkg interaction_pkg robot_bringup
source install/setup.bash
```

## Starting Enrollment Mode

### Terminal 1: Launch Services
```bash
ros2 launch robot_bringup enrollment_launch.py
```

Wait until you see:
```
  ENROLLMENT MODE READY
  Run enrollment CLI: ros2 run interaction_pkg enrollment_cli
```

### Terminal 2: Run Enrollment CLI
```bash
ros2 run interaction_pkg enrollment_cli
```

## Enrollment Workflow

### Step 1: Enter Person's Name
```
============================================================
       Ball-e Person Enrollment Wizard
============================================================

1. Enter person's name: John Doe

   Enrolling: John Doe
```

### Step 2: Face Enrollment (5 Poses)

The system will guide you through 5 face poses:

```
2. Face Enrollment
   --------------------------------------------------

   [FRONT] Look straight ahead at the camera
   Press Enter when ready...
   Detecting face... OK
   Generating embedding... CAPTURED (quality: 0.85)

   [LEFT] Turn your head 45 degrees to your LEFT
   Press Enter when ready...
   Detecting face... OK
   Generating embedding... CAPTURED (quality: 0.72)

   [RIGHT] Turn your head 45 degrees to your RIGHT
   Press Enter when ready...
   Detecting face... OK
   Generating embedding... CAPTURED (quality: 0.78)

   [UP] Tilt your head slightly UP
   Press Enter when ready...
   Detecting face... OK
   Generating embedding... CAPTURED (quality: 0.68)

   [DOWN] Tilt your head slightly DOWN
   Press Enter when ready...
   Detecting face... OK
   Generating embedding... CAPTURED (quality: 0.75)

   [5/5 poses captured]

   Computing representative embedding from 5 poses...
   Representative embedding computed (dim=512)
```

**Tips for Face Capture:**
- Position yourself 0.5-1m from camera
- Ensure good, even lighting
- Remove glasses if possible (or capture with and without)
- Keep expressions neutral
- Follow pose instructions exactly

### Step 3: Voice Enrollment (3 Phrases)

```
3. Voice Enrollment
   --------------------------------------------------

   Sample 1/3
   Say: "The quick brown fox jumps over the lazy dog"
   Press Enter when ready to speak...
   Listening... CAPTURED (2.45s)
   Generating voice embedding... OK

   Sample 2/3
   Say: "Ball-e, please remember my voice"
   Press Enter when ready to speak...
   Listening... CAPTURED (1.82s)
   Generating voice embedding... OK

   Sample 3/3
   Say: "Hello, my name is [say your name]"
   Press Enter when ready to speak...
   Listening... CAPTURED (1.95s)
   Generating voice embedding... OK

   [3/3 voice samples captured]
```

**Tips for Voice Capture:**
- Speak clearly at normal volume
- Minimize background noise
- Wait for "Listening..." before speaking
- If timeout occurs, try again with louder/clearer speech

### Step 4: Database Save

```
4. Saving to Database
   --------------------------------------------------

   Person ID: 5
   Face embeddings saved (5 poses + average)
   Voice embedding saved

============================================================
   Enrollment complete! John Doe is now registered.
============================================================
```

## Handling Issues

### Face Detection Fails
```
   [FRONT] Look straight ahead at the camera
   Press Enter when ready...
   Detecting face... FAILED - No face detected
   Retrying front...
```

**Solutions:**
- Improve lighting
- Move closer to camera
- Ensure face is fully visible
- Remove obstructions (hat, mask)

### Voice Capture Timeout
```
   Listening... TIMEOUT - No speech detected
```

**Solutions:**
- Speak louder
- Check microphone connection: `arecord -l`
- Adjust VAD threshold in launch file
- Reduce ambient noise

### Insufficient Poses Warning
```
   [2/5 poses captured]
   Warning: Less than 3 poses captured. Consider retrying.
```

**Recommendation:** Re-run enrollment for better recognition accuracy.

## Advanced Options

### Launch with Custom Settings
```bash
ros2 launch robot_bringup enrollment_launch.py \
    camera_index:=2 \
    use_gpu:=true \
    enable_voice:=false
```

### Disable Voice Enrollment
```bash
ros2 launch robot_bringup enrollment_launch.py enable_voice:=false
```

The system will proceed with face-only enrollment.

## Database Management

### View Enrolled People
```bash
ros2 service call /people_db/list_people msgs_interfaces/srv/ListPeople "{}"
```

### Remove a Person
```bash
ros2 service call /people_db/remove_person msgs_interfaces/srv/RemovePerson "{person_id: 5}"
```

### Check Database Location
Default: `/ball-e/ros2_ws/robot_data/people.db`

Can be changed via parameter:
```yaml
people_database_node:
  ros__parameters:
    db_path: '/custom/path/people.db'
```

## Post-Enrollment Verification

After enrollment, verify recognition:

```bash
# Launch full identification pipeline
ros2 launch robot_bringup multimodal_identification_launch.py

# Check identification output
ros2 topic echo /person_states
```

You should see:
```yaml
person_states:
- track_id: 1
  person_id: 5
  name: "John Doe"
  identity_confidence: 0.85
  ...
```

## Batch Enrollment

For enrolling multiple people:

1. Launch enrollment services once
2. Run CLI multiple times:

```bash
# In Terminal 2
ros2 run interaction_pkg enrollment_cli  # Person 1
ros2 run interaction_pkg enrollment_cli  # Person 2
ros2 run interaction_pkg enrollment_cli  # Person 3
```

## Re-Enrollment

To update an existing person's biometrics:

1. Remove old entry:
```bash
ros2 service call /people_db/remove_person msgs_interfaces/srv/RemovePerson "{person_id: 5}"
```

2. Re-enroll with same name:
```bash
ros2 run interaction_pkg enrollment_cli
# Enter same name when prompted
```

## Troubleshooting

### Services Not Available
```
Waiting for services...
[ERROR] Service /face_detection/detect_face not available
```

**Solution:** Ensure enrollment_launch.py is running and all nodes started successfully.

### Permission Denied (Camera/Microphone)
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Add user to audio group
sudo usermod -a -G audio $USER

# Logout and login for changes to take effect
```

### Low Quality Scores
If quality scores are consistently low:
- Improve camera resolution/focus
- Better lighting conditions
- Position face more centrally in frame

## Best Practices

1. **Consistent Environment**: Enroll in similar conditions to recognition environment
2. **Multiple Sessions**: Consider enrolling same person on different days
3. **Verify After Enrollment**: Always test recognition after enrollment
4. **Keep Records**: Note person_id assignments for reference
5. **Regular Updates**: Re-enroll if appearance changes significantly (haircut, beard, glasses)

## Next Steps

After successful enrollment:
- Run full identification system: See `MULTIMODAL_IDENTIFICATION.md`
- Fine-tune parameters: Adjust thresholds based on performance
- Add more people: Repeat enrollment process
