# Audio Pipeline API Reference

## Topics

### `/microphone/audio_raw`
**Type:** `msgs_interfaces/msg/AudioChunk`

Raw audio stream from microphone.

```python
# Subscribe
self.create_subscription(AudioChunk, '/microphone/audio_raw', self.callback, 10)

def callback(self, msg):
    samples = np.array(msg.audio_data, dtype=np.float32)
    rms = msg.rms_level
    # Process audio chunk
```

**Fields:**
- `header`: Standard ROS header with timestamp
- `audio_data`: float32[] - PCM samples normalized to [-1, 1]
- `sample_rate`: uint32 - Typically 16000 Hz
- `num_channels`: uint32 - 1 for mono
- `num_samples`: uint32 - Samples in this chunk
- `rms_level`: float32 - Root Mean Square energy
- `is_clipping`: bool - True if audio is clipping

---

### `/microphone/speech_segment`
**Type:** `msgs_interfaces/msg/SpeechSegment`

Complete speech segments detected by VAD.

```python
self.create_subscription(SpeechSegment, '/microphone/speech_segment', self.callback, 10)

def callback(self, msg):
    audio = np.array(msg.audio_data, dtype=np.float32)
    duration = msg.duration_seconds
    # Process complete speech utterance
```

**Fields:**
- `header`: Standard ROS header
- `audio_data`: float32[] - Complete speech audio
- `sample_rate`: uint32 - Sample rate (16000)
- `duration_seconds`: float32 - Total speech duration
- `start_time`: builtin_interfaces/Time - When speech started
- `end_time`: builtin_interfaces/Time - When speech ended
- `confidence`: float32 - VAD confidence score
- `average_rms`: float32 - Average energy level
- `is_complete`: bool - True if segment is complete

---

### `/speech/transcription`
**Type:** `std_msgs/msg/String`

Speech-to-text transcription output.

```python
self.create_subscription(String, '/speech/transcription', self.callback, 10)

def callback(self, msg):
    text = msg.data
    print(f"Heard: {text}")
```

---

### `/voice/embedding`
**Type:** `std_msgs/msg/Float32MultiArray`

Generated voice embedding from speech segment.

```python
self.create_subscription(Float32MultiArray, '/voice/embedding', self.callback, 10)

def callback(self, msg):
    embedding = np.array(msg.data, dtype=np.float32)  # 256-dim
```

---

## Services

### `/voice_recognition/generate_embedding`
**Type:** `msgs_interfaces/srv/GenerateVoiceEmbedding`

Generate speaker embedding from audio data.

**Request:**
```python
request = GenerateVoiceEmbedding.Request()
request.audio_data = audio_samples.tolist()  # float32 list
request.sample_rate = 16000
```

**Response:**
```python
response.success  # bool - Operation success
response.embedding  # float32[] - 256-dim embedding
response.message  # string - Error message if failed
response.confidence  # float32 - Embedding quality score
```

**Example:**
```python
from msgs_interfaces.srv import GenerateVoiceEmbedding

client = self.create_client(GenerateVoiceEmbedding, '/voice_recognition/generate_embedding')

request = GenerateVoiceEmbedding.Request()
request.audio_data = speech_segment.audio_data
request.sample_rate = speech_segment.sample_rate

future = client.call_async(request)
rclpy.spin_until_future_complete(self, future)

response = future.result()
if response.success:
    embedding = np.array(response.embedding)
    print(f"Generated {len(embedding)}-dim embedding")
```

---

### `/people_db/recognize_voice`
**Type:** `msgs_interfaces/srv/RecognizeVoice`

Match voice embedding against database.

**Request:**
```python
request = RecognizeVoice.Request()
request.voice_embedding = embedding.tolist()
request.confidence_threshold = 0.6
```

**Response:**
```python
response.match_found  # bool - True if match above threshold
response.person_name  # string - Matched person's name
response.person_id  # int32 - Matched person's database ID
response.similarity_score  # float32 - Best match score

# For Bayesian fusion:
response.all_person_ids  # int32[] - All person IDs
response.all_scores  # float32[] - Score for each person
```

**Example:**
```python
from msgs_interfaces.srv import RecognizeVoice

client = self.create_client(RecognizeVoice, '/people_db/recognize_voice')

request = RecognizeVoice.Request()
request.voice_embedding = voice_embedding.tolist()
request.confidence_threshold = 0.6

future = client.call_async(request)
rclpy.spin_until_future_complete(self, future)

response = future.result()
if response.match_found:
    print(f"Matched: {response.person_name} (score: {response.similarity_score:.3f})")

# For Bayesian fusion
scores_dict = {
    pid: score
    for pid, score in zip(response.all_person_ids, response.all_scores)
}
```

---

### `/people_db/add_person`
**Type:** `msgs_interfaces/srv/AddPerson`

Add new person to database.

**Request:**
```python
request = AddPerson.Request()
request.name = "John Doe"
request.face_embedding = face_emb.tolist()  # float32[] - 512-dim
request.preferences_json = '{}'  # JSON string
request.notes = "Enrolled via CLI"
```

**Response:**
```python
response.success  # bool
response.person_id  # int32 - Assigned ID
response.message  # string - Status message
```

---

### `/face_recognition/generate_embedding`
**Type:** `msgs_interfaces/srv/GenerateEmbedding`

Generate face embedding from image region.

**Request:**
```python
request = GenerateEmbedding.Request()
request.header = image_msg.header  # Frame reference
request.track_id = 0
request.face_x = 100  # Face bounding box
request.face_y = 150
request.face_w = 200
request.face_h = 200
```

**Response:**
```python
response.success  # bool
response.embedding  # float32[] - 512-dim face embedding
response.message  # string
```

---

### `/people_db/recognize_face`
**Type:** `msgs_interfaces/srv/RecognizeFace`

Match face embedding against database.

**Request:**
```python
request = RecognizeFace.Request()
request.embedding = face_embedding.tolist()
request.confidence_threshold = 0.6
```

**Response:**
```python
response.match_found  # bool
response.person_name  # string
response.person_id  # int32
response.similarity_score  # float32

# For Bayesian fusion:
response.all_person_ids  # int32[]
response.all_scores  # float32[]
```

---

## Temporal Bayesian Tracker API

### Class: `TemporalBayesianIdentity`

```python
from perception_pkg.temporal_bayesian_tracker import TemporalBayesianIdentity

# Initialize
tracker = TemporalBayesianIdentity(
    known_person_ids=[1, 2, 3, 4],  # Database person IDs
    prior_unknown=0.1,  # Prior P(unknown)
    decay_rate_per_second=0.01  # Confidence decay rate
)
```

#### Methods

**`update_face(face_scores, timestamp=None)`**

Update belief with face recognition scores.

```python
face_scores = {
    1: 0.85,  # Person 1: 85% similarity
    2: 0.32,  # Person 2: 32% similarity
    3: 0.45,  # Person 3: 45% similarity
    4: 0.28   # Person 4: 28% similarity
}
tracker.update_face(face_scores)
```

**`update_voice(voice_scores, timestamp=None)`**

Update belief with voice recognition scores.

```python
voice_scores = {
    1: 0.78,  # Person 1: 78% similarity
    2: 0.41,  # Person 2: 41% similarity
    3: 0.35,  # Person 3: 35% similarity
    4: 0.22   # Person 4: 22% similarity
}
tracker.update_voice(voice_scores)
```

**`get_identity()`**

Get most likely identity and confidence.

```python
identity, confidence = tracker.get_identity()
# identity: int (person_id) or 'unknown'
# confidence: float [0, 1]

if identity == 'unknown':
    print(f"Unknown person (conf: {confidence:.3f})")
else:
    print(f"Person ID {identity} (conf: {confidence:.3f})")
```

**`get_identity_with_decay()`**

Get identity with time-based decay applied.

```python
identity, decayed_conf = tracker.get_identity_with_decay()
# Confidence decreased based on time since last update
```

**`get_all_beliefs()`**

Get full probability distribution.

```python
beliefs = tracker.get_all_beliefs()
# {1: 0.45, 2: 0.12, 3: 0.08, 4: 0.05, 'unknown': 0.30}

for person_id, prob in sorted(beliefs.items(), key=lambda x: x[1], reverse=True):
    print(f"{person_id}: {prob:.3f}")
```

**`predict(time_elapsed=None)`**

Apply time-based decay (prediction step of Bayes filter).

```python
tracker.predict(time_elapsed=5.0)  # 5 seconds elapsed
# Beliefs decay towards uniform distribution
```

**`add_person(person_id)`**

Add new person to tracker (after enrollment).

```python
tracker.add_person(person_id=5)
# Redistributes probability mass
```

**`remove_person(person_id)`**

Remove person from tracker.

```python
tracker.remove_person(person_id=3)
# Re-normalizes remaining beliefs
```

**`reset()`**

Reset to initial uniform distribution.

```python
tracker.reset()
# Clears all accumulated evidence
```

**`get_time_since_last_update()`**

Get timing information.

```python
times = tracker.get_time_since_last_update()
# {
#   'face': 2.5,  # seconds since last face update
#   'voice': 10.3,  # seconds since last voice update
#   'any': 2.5  # seconds since any update
# }
```

---

### Class: `MultiTrackIdentityManager`

Manages multiple trackers for multiple visual tracks.

```python
from perception_pkg.temporal_bayesian_tracker import MultiTrackIdentityManager

manager = MultiTrackIdentityManager(known_person_ids=[1, 2, 3, 4])
```

**`get_tracker(track_id)`**

Get or create tracker for a track.

```python
tracker = manager.get_tracker(track_id=5)
tracker.update_face(scores)
```

**`remove_tracker(track_id)`**

Remove tracker when track is lost.

```python
manager.remove_tracker(track_id=5)
```

**`add_known_person(person_id)`**

Add new person to all trackers.

```python
manager.add_known_person(person_id=6)
```

**`get_all_identities()`**

Get identities for all tracks.

```python
identities = manager.get_all_identities()
# {
#   5: (1, 0.85),  # track 5 → person 1, 85% confidence
#   7: (2, 0.72),  # track 7 → person 2, 72% confidence
#   9: ('unknown', 0.45)  # track 9 → unknown
# }
```

**`cleanup_old_trackers(max_age_seconds=60.0)`**

Remove stale trackers.

```python
manager.cleanup_old_trackers(max_age_seconds=120.0)
# Removes trackers not updated for 2 minutes
```

---

## Usage Examples

### Complete Voice Recognition Pipeline

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from msgs_interfaces.msg import SpeechSegment
from msgs_interfaces.srv import GenerateVoiceEmbedding, RecognizeVoice


class VoiceIdentifier(Node):
    def __init__(self):
        super().__init__('voice_identifier')

        # Service clients
        self.gen_emb_client = self.create_client(
            GenerateVoiceEmbedding, '/voice_recognition/generate_embedding'
        )
        self.recognize_client = self.create_client(
            RecognizeVoice, '/people_db/recognize_voice'
        )

        # Subscribe to speech
        self.speech_sub = self.create_subscription(
            SpeechSegment, '/microphone/speech_segment', self.speech_callback, 10
        )

    def speech_callback(self, msg):
        # Generate embedding
        emb_request = GenerateVoiceEmbedding.Request()
        emb_request.audio_data = list(msg.audio_data)
        emb_request.sample_rate = msg.sample_rate

        emb_future = self.gen_emb_client.call_async(emb_request)
        emb_future.add_done_callback(self.embedding_callback)

    def embedding_callback(self, future):
        emb_response = future.result()
        if not emb_response.success:
            return

        # Recognize voice
        rec_request = RecognizeVoice.Request()
        rec_request.voice_embedding = list(emb_response.embedding)
        rec_request.confidence_threshold = 0.6

        rec_future = self.recognize_client.call_async(rec_request)
        rec_future.add_done_callback(self.recognition_callback)

    def recognition_callback(self, future):
        response = future.result()
        if response.match_found:
            self.get_logger().info(
                f"Identified: {response.person_name} "
                f"(score: {response.similarity_score:.3f})"
            )
        else:
            self.get_logger().info("Unknown speaker")


def main():
    rclpy.init()
    node = VoiceIdentifier()
    rclpy.spin(node)
    rclpy.shutdown()
```

### Bayesian Fusion Example

```python
from perception_pkg.temporal_bayesian_tracker import TemporalBayesianIdentity

# Create tracker
tracker = TemporalBayesianIdentity([1, 2, 3])

# Initial belief (uniform)
print("Initial:", tracker.get_all_beliefs())
# {1: 0.3, 2: 0.3, 3: 0.3, 'unknown': 0.1}

# Face recognition suggests person 1
face_scores = {1: 0.75, 2: 0.35, 3: 0.40}
tracker.update_face(face_scores)
print("After face:", tracker.get_identity())
# (1, 0.65)

# Voice recognition also suggests person 1
voice_scores = {1: 0.80, 2: 0.30, 3: 0.25}
tracker.update_voice(voice_scores)
print("After voice:", tracker.get_identity())
# (1, 0.89) - Confidence increased with corroborating evidence

# Voice recognition suggests person 2 (conflicting)
voice_scores = {1: 0.30, 2: 0.85, 3: 0.20}
tracker.update_voice(voice_scores)
print("After conflicting voice:", tracker.get_identity())
# (1, 0.62) or (2, 0.55) - Depends on prior strength
```
