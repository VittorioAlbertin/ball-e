# Face Enrollment Node

## Overview
The face enrollment node handles adding unknown faces to the people database. It monitors face recognition results, detects unknown faces, and provides a simple interface for enrolling them with a name.

## Node Information
- **Package**: `interaction_pkg`
- **Executable**: `face_enrollment_node`
- **Node Name**: `face_enrollment_node`

## Topics

### Subscribed Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/face/recognition` | `msgs_interfaces/FaceRecognition` | Face recognition results from face detection node |

## Services

### Provided Services
| Service | Type | Description |
|---------|------|-------------|
| `enroll_pending_face` | `msgs_interfaces/EnrollPendingFace` | Enroll the currently pending unknown face |

### Service Clients
| Service | Type | Description |
|---------|------|-------------|
| `people_db/add_person` | `msgs_interfaces/AddPerson` | Add new person to database |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cooldown_seconds` | float | `10.0` | Minimum seconds between enrollment prompts |
| `min_confidence` | float | `0.5` | Minimum face confidence to offer enrollment |

## Workflow

### 1. Unknown Face Detection
```
Face Recognition → Unknown (not found) → Check Confidence → Check Cooldown → Store Pending
```

### 2. User Enrollment
```
User Runs CLI → Service Call → Add to Database → Clear Pending
```

### 3. Cooldown Management
Prevents spam by enforcing minimum time between prompts:
- Unknown face detected → Start cooldown timer
- Additional unknown faces within cooldown → Ignored
- After cooldown expires → Accept new unknown face

## Pending Face Storage

### Data Structure
```python
pending_enrollment = {
    'embedding': [512-dim numpy array],
    'confidence': float,
    'timestamp': float  # Unix timestamp
}
```

### Expiration
- Pending faces expire after 60 seconds
- Only one pending face at a time
- New pending face overwrites old one

## Service: enroll_pending_face

### Request
```
string name        # Name for the person (required)
string notes       # Optional notes about the person
```

### Response
```
bool success       # True if enrollment succeeded
string message     # Status or error message
int32 person_id    # Database ID of enrolled person (-1 on failure)
```

### Error Cases

| Error | Cause | Solution |
|-------|-------|----------|
| "No pending face to enroll" | No unknown face detected recently | Wait for unknown face detection |
| "Enrollment data expired" | More than 60 seconds since detection | Detect unknown face again |
| "Service call failed" | Database service unavailable | Check people_database_node is running |

## CLI Tool: enroll_face

### Usage
```bash
ros2 run interaction_pkg enroll_face <name> [notes]
```

### Examples
```bash
# Simple enrollment
ros2 run interaction_pkg enroll_face "John Doe"

# With notes
ros2 run interaction_pkg enroll_face "Jane Smith" "Colleague from IT department"

# Name with spaces (use quotes)
ros2 run interaction_pkg enroll_face "Maria Garcia"
```

### Output
```
✓ SUCCESS: Successfully enrolled John Doe
  Person ID: 5
```

or

```
✗ FAILED: No pending face to enroll
```

## Complete Enrollment Flow

### Step-by-Step

1. **Unknown Face Appears**
   ```
   [face_enrollment_node] ============================================================
   [face_enrollment_node] UNKNOWN FACE DETECTED!
   [face_enrollment_node] Confidence: 0.72
   [face_enrollment_node] ============================================================
   [face_enrollment_node] Would you like to add this person to the database?
   [face_enrollment_node] Use the add_face service or call /enroll_face action
   [face_enrollment_node]
   [face_enrollment_node] Example:
   [face_enrollment_node]   ros2 run interaction_pkg enroll_face "John Doe"
   [face_enrollment_node] ============================================================
   ```

2. **User Enrolls** (within 60 seconds)
   ```bash
   ros2 run interaction_pkg enroll_face "John Doe" "New friend"
   ```

3. **Confirmation**
   ```
   Waiting for enroll_pending_face service...
   Enrolling pending face as: John Doe
   ✓ SUCCESS: Successfully enrolled John Doe
     Person ID: 5
   ```

4. **Next Frame**
   - Face is now recognized
   - Shows "John Doe" in green on bbox
   - No more enrollment prompts

## Anti-Spam Features

### Cooldown Timer
- Default: 10 seconds between prompts
- Prevents console spam with multiple unknowns
- Adjustable via parameter

### Confidence Filter
- Default: Only faces with >0.5 confidence
- Prevents enrollment of poor quality detections
- Adjustable via parameter

### Single Pending Face
- Only stores most recent unknown face
- New unknown overwrites old pending
- Prevents confusion with multiple unknowns

## Integration with Face Detection

### Message Flow
```
Camera → YOLO → Face Detection → Recognition → Enrollment Node
                                     ↓
                                 Database
                                     ↓
                              (if not found)
                                     ↓
                              Enrollment Prompt
```

### Recognition Result Processing
```python
def face_callback(self, msg):
    # Only process unknown faces
    if msg.found:
        return  # Recognized, ignore

    # Check confidence
    if msg.confidence < self.min_confidence:
        return  # Too low, ignore

    # Check cooldown
    if time.time() - self.last_unknown_time < self.cooldown_seconds:
        return  # Too soon, ignore

    # Store pending enrollment
    self.pending_enrollment = {
        'embedding': msg.face_embedding,
        'confidence': msg.confidence,
        'timestamp': time.time()
    }

    # Prompt user
    self.print_enrollment_prompt()
```

## Usage Examples

### Python Client

```python
import rclpy
from rclpy.node import Node
from msgs_interfaces.srv import EnrollPendingFace

class EnrollmentClient(Node):
    def __init__(self):
        super().__init__('enrollment_client')
        self.client = self.create_client(
            EnrollPendingFace,
            'enroll_pending_face'
        )
        self.client.wait_for_service()

    def enroll(self, name, notes=''):
        request = EnrollPendingFace.Request()
        request.name = name
        request.notes = notes

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.success:
            print(f"Enrolled as ID {result.person_id}")
        else:
            print(f"Failed: {result.message}")
```

### Automated Enrollment (Advanced)

```python
# Auto-enroll with generated name (for testing)
class AutoEnroller(Node):
    def __init__(self):
        super().__init__('auto_enroller')
        self.enrollment_sub = self.create_subscription(
            FaceRecognition,
            '/face/recognition',
            self.face_callback,
            10
        )
        self.enroll_client = self.create_client(
            EnrollPendingFace,
            'enroll_pending_face'
        )
        self.counter = 0

    def face_callback(self, msg):
        if not msg.found and msg.confidence > 0.6:
            # Wait a moment for enrollment node to store pending face
            time.sleep(0.5)

            # Auto-enroll with generated name
            self.counter += 1
            request = EnrollPendingFace.Request()
            request.name = f"Person_{self.counter}"
            request.notes = "Auto-enrolled"

            self.enroll_client.call_async(request)
```

## Troubleshooting

### "No pending face to enroll"
**Cause**: No recent unknown face detection
**Solutions**:
1. Ensure face_detection_node is running
2. Check unknown face appears in `/face/debug_image`
3. Wait for enrollment prompt in logs
4. Check face confidence is above threshold

### "Enrollment data expired"
**Cause**: More than 60 seconds since detection
**Solutions**:
1. Re-detect the unknown face
2. Enroll faster (within 60 seconds)
3. Check logs for new enrollment prompt

### Service Not Available
**Cause**: Enrollment node not running
**Solutions**:
1. Launch enrollment node: `ros2 run interaction_pkg face_enrollment_node`
2. Check it's in your launch file
3. Verify no startup errors in logs

### Multiple Prompts for Same Person
**Cause**: Cooldown too short or person moving
**Solutions**:
1. Increase `cooldown_seconds` parameter
2. Enroll the person to stop prompts
3. Check face is being detected consistently

### No Enrollment Prompts
**Cause**: Faces being recognized or low confidence
**Solutions**:
1. Check `/face/recognition` topic shows `found: false`
2. Lower `min_confidence` parameter
3. Verify face detection is working
4. Check database is empty (no enrolled faces)

## Configuration Examples

### Aggressive Enrollment (Frequent Prompts)
```python
Node(
    package='interaction_pkg',
    executable='face_enrollment_node',
    parameters=[
        {'cooldown_seconds': 5.0},    # Prompt every 5 seconds
        {'min_confidence': 0.3}        # Accept lower quality faces
    ]
)
```

### Conservative Enrollment (Rare Prompts)
```python
Node(
    package='interaction_pkg',
    executable='face_enrollment_node',
    parameters=[
        {'cooldown_seconds': 30.0},   # Prompt every 30 seconds
        {'min_confidence': 0.7}        # Only high quality faces
    ]
)
```

## Dependencies
- `rclpy`
- `msgs_interfaces` (custom messages/services)
- `people_database_node` (must be running)

## Notes
- Synchronous service call to database (blocking)
- Only one pending face at a time
- Thread-safe (service callbacks)
- Face embeddings stored from recognition message
- No duplicate detection (relies on database)
