# Multi-Modal Person Identification System

## Overview

Ball-e's multi-modal identification system combines **face recognition** and **voice recognition** using **Bayesian fusion** to achieve robust person identification. This document describes the architecture, algorithms, and usage of the system.

## Table of Contents

1. [Architecture](#architecture)
2. [Three-Layer Tracking](#three-layer-tracking)
3. [Bayesian Identity Fusion](#bayesian-identity-fusion)
4. [Components](#components)
5. [Message Interfaces](#message-interfaces)
6. [Launch Files](#launch-files)
7. [Enrollment Process](#enrollment-process)
8. [Configuration](#configuration)
9. [Dependencies](#dependencies)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISUAL PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│  Camera → YOLO → ByteTrack → Face Detector → Face Recognizer   │
│                      ↓                              ↓           │
│                  track_id                    face_scores        │
└──────────────────────┬──────────────────────────────┬───────────┘
                       │                              │
                       ▼                              ▼
              ┌─────────────────────────────────────────┐
              │      PERSON STATE MANAGER               │
              │  ┌───────────────────────────────┐     │
              │  │  Temporal Bayesian Tracker    │←────┼─── SINGLE SOURCE OF TRUTH
              │  │  P(identity | face, voice)    │     │
              │  │  Returns: person_id (int)     │     │
              │  └───────────────────────────────┘     │
              │                                         │
              │  Internal State: person_id (int)       │
              │  Published: person_name (string)       │
              │        ↓                               │
              │  GetPerson Service → DB name lookup    │
              │                                         │
              │  Frame Cache Control:                   │
              │  → /frame_cache/pin_timestamp          │
              │  → /frame_cache/release_timestamp      │
              └─────────────────────────────────────────┘
                       ▲                              ▲
                       │                              │
                  track_id                    voice_scores
┌──────────────────────┴──────────────────────────────┴───────────┐
│                    AUDIO PIPELINE                               │
├─────────────────────────────────────────────────────────────────┤
│  Microphone → VAD → Speech Segment → Voice Recognizer → STT    │
└─────────────────────────────────────────────────────────────────┘
```

### Frame Cache Management (NEW)

The system uses explicit frame pinning to prevent cache misses during identification:

1. **Rolling Cache**: Both face_detector (5s) and face_recognizer (3s) maintain time-based caches
2. **Pinned Cache**: When identification starts, PSM pins the frame via topic
3. **Auto-expire**: Pinned frames auto-expire after 30 seconds (safety net)
4. **Explicit Release**: PSM releases frames after identification completes

---

## Three-Layer Tracking

The system employs three distinct tracking layers, each serving a different purpose:

### Layer 1: Visual Tracking (ByteTrack)
- **Purpose**: Spatial tracking - "Where is this person in the frame?"
- **Output**: `track_id`, bounding box, tracking confidence
- **Updates**: Every frame (30 FPS)
- **Note**: `tracking_confidence` reflects visual tracking quality, NOT identity certainty

### Layer 2: Biometric Recognition (Face + Voice)
- **Purpose**: Single-observation identity - "Who does this person look/sound like?"
- **Output**: Similarity scores for all known persons
- **Updates**: On-demand (new track, low confidence due to decay, speech event)

### Layer 3: Temporal Bayesian Tracking
- **Purpose**: Smoothed identity over time - "Who is this person, considering all evidence?"
- **Output**: Probability distribution over identities
- **Updates**: After each biometric observation
- **Decay**: Confidence decays towards uniform distribution over time (1% per second)

**Key Insight**: The three layers are independent. ByteTrack may lose and re-acquire a track (causing track_id change), but the Bayesian tracker maintains identity continuity through biometric evidence. Re-identification is triggered by confidence decay, not periodic timers.

---

## Bayesian Identity Fusion

### Mathematical Foundation

The system uses **recursive Bayesian filtering** with **Naive Bayes** assumption (conditional independence of modalities).

**Bayes' Theorem**:
```
P(identity | evidence) ∝ P(evidence | identity) × P(identity)
```

**Multi-modal fusion** (assuming independence):
```
P(identity | face, voice) ∝ P(face | identity) × P(voice | identity) × P(identity)
```

### Update Process

1. **Prior**: Current belief distribution over all known persons + "unknown"
2. **Likelihood**: Convert similarity scores to probabilities
3. **Posterior**: Multiply prior × likelihood, then normalize

```python
def update_with_evidence(scores, modality):
    for person_id in belief:
        prior = belief[person_id]
        likelihood = scores_to_likelihood(scores[person_id])
        new_belief[person_id] = prior * likelihood
    normalize(new_belief)
```

### Temporal Decay

Without new evidence, confidence decays towards uniform distribution:
```python
belief[id] = (1 - decay) * belief[id] + decay * (1/N)
```

This models the intuition that identity certainty decreases over time without reinforcement.

**Re-identification Trigger**: When decayed confidence drops below `identity_confidence_threshold` (default 0.5), the system automatically triggers re-identification. This replaces periodic timer-based re-identification with a principled uncertainty-driven approach.

### Identity Storage Architecture (NEW)

The system uses a **two-layer identity representation**:

1. **Internal State**: Stores `person_id` (int), -1 for unknown
   - Face recognition → Bayesian tracker → returns person_id
   - Voice recognition → Bayesian tracker → state unchanged (just tracker)
   - State synchronized from Bayesian tracker every 1 second

2. **Published Messages**: Contains person_name (string)
   - PersonStateArray publishes actual names from database
   - GetPerson service client fetches names with caching
   - Names resolved only at publish time (10 Hz)

**Data Flow**:
```
Face Recognition → all_scores → tracker.update_face()
                                         ↓
                              Bayesian Tracker (single source)
                                         ↓
                              person_id + Bayesian confidence
                                         ↓
                              state['identity'] = person_id (int)
                                         ↓
                              publish_state_array()
                                         ↓
                              get_person_name(person_id) → DB lookup
                                         ↓
                              PersonState.identity = "Alice" (string)
```

This design ensures:
- Both face and voice use the same Bayesian tracker
- Confidence values are always comparable (Bayesian, not raw similarity)
- No type mismatches between modalities
- Database names fetched only when needed (with caching)

### Unknown Person Handling

The system maintains probability for "unknown" category:
- If all biometric scores are low → more likely unknown
- If any score is high → less likely unknown

```python
unknown_likelihood = 1.0 - max(scores.values())
```

---

## Components

### Sensors Package

#### `microphone_node.py`
- **Function**: Audio capture with Voice Activity Detection (VAD)
- **Input**: System audio device
- **Output**:
  - `/microphone/audio_raw` (AudioChunk) - raw audio stream
  - `/microphone/speech_segment` (SpeechSegment) - detected speech
- **VAD Algorithm**: Energy-based detection with smoothing
  - Speech starts when RMS > threshold for min_speech_ms
  - Speech ends when RMS < threshold for max_silence_ms

### Perception Package

#### `voice_recognizer_node.py`
- **Function**: Speaker embedding generation using Resemblyzer
- **Model**: Resemblyzer (pretrained speaker encoder)
- **Output**: 256-dimensional L2-normalized embedding
- **Service**: `/voice_recognition/generate_embedding`

#### `speech_to_text_node.py`
- **Function**: Speech transcription using OpenAI Whisper
- **Models**: tiny, base, small, medium (configurable)
- **Output**: `/speech/transcription` (String)

#### `temporal_bayesian_tracker.py`
- **Function**: Bayesian identity tracking over time
- **Classes**:
  - `TemporalBayesianIdentity`: Single-track tracker
  - `MultiTrackIdentityManager`: Multi-track management
- **Features**:
  - Asynchronous face/voice updates
  - Time-based confidence decay
  - Unknown person probability

#### `person_state_manager_node.py` (Modified)
- **Function**: World state orchestrator with Bayesian fusion
- **Key Architecture**: Bayesian tracker is the single source of truth for identity
- **Features**:
  - Maintains `identity_trackers` dict (track_id → Bayesian tracker)
  - Subscribes to speech segments for voice recognition
  - Uses identity confidence (not tracking confidence) for decisions
  - Fuses face and voice scores into unified identity
  - **Stores person_id (int) internally**, not person names
  - **Names fetched from database** via GetPerson service with caching
  - Face recognition updates tracker, returns Bayesian output
  - Voice recognition updates tracker only, no direct state modification
  - State synchronized from Bayesian tracker via 1 Hz decay timer

### Interaction Package

#### `people_database_node.py`
- **Extended Schema**:
  - `voice_embedding` column in people table
  - `face_embeddings` table for multi-pose storage
- **New Services**:
  - `RecognizeVoice`: Match voice against database
  - Methods for multi-pose face embeddings

#### `enrollment_cli.py`
- **Function**: Interactive terminal-based enrollment wizard
- **Features**:
  - Multi-pose face capture (front, left, right, up, down)
  - Quality-weighted embedding averaging
  - Multi-phrase voice capture
  - Progress feedback and error handling

---

## Message Interfaces

### New Messages

#### `AudioChunk.msg`
```
std_msgs/Header header
float32[] audio_data      # Raw PCM samples
uint32 sample_rate        # Hz (typically 16000)
uint32 num_channels       # 1 for mono
uint32 num_samples        # Samples in chunk
float32 rms_level         # Root Mean Square energy
bool is_clipping          # Audio clipping detected
```

#### `SpeechSegment.msg`
```
std_msgs/Header header
float32[] audio_data          # Complete speech audio
uint32 sample_rate            # Hz
float32 duration_seconds      # Speech duration
builtin_interfaces/Time start_time
builtin_interfaces/Time end_time
float32 confidence            # VAD confidence
float32 average_rms           # Average energy
bool is_complete              # Segment complete flag
```

### New Services

#### `GenerateVoiceEmbedding.srv`
```
# Request
float32[] audio_data
uint32 sample_rate
---
# Response
bool success
float32[] embedding    # 256-dim normalized vector
string message
float32 confidence
```

#### `RecognizeVoice.srv`
```
# Request
float32[] voice_embedding
float32 confidence_threshold
---
# Response
bool match_found
string person_name
int32 person_id
float32 similarity_score
int32[] all_person_ids     # For Bayesian fusion
float32[] all_scores       # Scores for all persons
```

### Modified Services

#### `RecognizeFace.srv` (Updated)
Added fields for Bayesian fusion:
```
# Response (new fields)
int32[] all_person_ids
float32[] all_scores
```

#### `GetPerson.srv` (Used for name lookup)
Fetches person details from database:
```
# Request
int32 person_id
string name  # Optional: query by name
---
# Response
bool found
int32 person_id
string name
string last_seen
string created_at
int32 interaction_count
string preferences_json
string notes
string message
```

PersonStateManager uses this service with caching to resolve person_id → person_name for publishing.

---

## Launch Files

### `audio_pipeline_launch.py`
Standalone audio processing:
```bash
ros2 launch robot_bringup audio_pipeline_launch.py \
    sample_rate:=16000 \
    vad_threshold:=0.01 \
    whisper_model:=base \
    enable_stt:=true
```

### `multimodal_identification_launch.py`
Complete system with Bayesian fusion:
```bash
ros2 launch robot_bringup multimodal_identification_launch.py \
    camera_index:=0 \
    use_gpu:=true \
    enable_voice:=true \
    enable_stt:=true \
    identity_confidence_threshold:=0.6
```

### `enrollment_launch.py`
Person enrollment mode:
```bash
# Terminal 1
ros2 launch robot_bringup enrollment_launch.py

# Terminal 2
ros2 run interaction_pkg enrollment_cli
```

---

## Enrollment Process

### Multi-Pose Face Capture

The enrollment CLI captures 5 face poses:
1. **Front**: Looking straight ahead
2. **Left**: Head turned 45° left
3. **Right**: Head turned 45° right
4. **Up**: Head tilted slightly up
5. **Down**: Head tilted slightly down

Each pose is assigned a quality score based on face size:
```python
quality_score = min(1.0, face_area / 10000.0)
```

The representative embedding is a **quality-weighted average**:
```python
weighted_sum = Σ(embedding_i × quality_i)
representative = weighted_sum / Σ(quality_i)
representative = normalize(representative)  # L2 normalization
```

### Multi-Phrase Voice Capture

Three phrases are captured:
1. "The quick brown fox jumps over the lazy dog" (phonetically diverse)
2. "Ball-e, please remember my voice" (natural speech)
3. "Hello, my name is [name]" (personalized)

Voice embeddings are averaged and normalized:
```python
avg_embedding = mean(embeddings, axis=0)
avg_embedding = normalize(avg_embedding)
```

---

## Configuration

### PersonStateManager Parameters

```yaml
identity_confidence_threshold: 0.5  # Min Bayesian confidence for identity
known_person_reidentify_interval: 15.0  # Cooldown between requests (known persons)
unknown_person_reidentify_interval: 5.0  # Cooldown between requests (unknown persons)
enable_voice_recognition: true
# NOTE: These intervals are cooldowns to prevent spam, NOT periodic triggers
# Re-identification is triggered by confidence decay below threshold, not timers
```

### ByteTrack Person Tracker Parameters

```yaml
max_age: 30  # Frames to keep lost track
min_hits: 20  # Consecutive detections before confirming (was 3, now 20 for stable faces)
iou_threshold: 0.3  # IoU for matching
high_conf_threshold: 0.6  # High quality detection threshold
```

### Temporal Bayesian Tracker Parameters

```python
TemporalBayesianIdentity(
    known_person_ids=[1, 2, 3],
    prior_unknown=0.1,  # 10% prior for unknown person
    decay_rate_per_second=0.01  # Confidence decay rate
)
# Note: History limited to 100 entries to prevent memory bloat
```

### Microphone Node Parameters

```yaml
sample_rate: 16000  # Hz
chunk_duration_ms: 100  # Audio chunk size
vad_threshold: 0.01  # RMS threshold for speech
vad_min_speech_ms: 300  # Min speech duration
vad_max_silence_ms: 500  # Max silence before end
```

---

## Dependencies

### Python Packages

```bash
pip install resemblyzer  # Speaker embeddings
pip install openai-whisper  # Speech-to-text
pip install sounddevice  # Audio capture
pip install scipy  # Signal processing
pip install numpy  # Numerical operations
```

### System Dependencies

```bash
sudo apt install portaudio19-dev  # For sounddevice
sudo apt install ffmpeg  # For Whisper
```

### ROS 2 Dependencies

- sensor_msgs
- std_msgs
- cv_bridge
- msgs_interfaces (custom)

---

## Performance Considerations

### Computational Load

| Component | CPU Load | GPU Load | Memory |
|-----------|----------|----------|--------|
| Microphone Node | Low | None | ~10 MB |
| Voice Recognizer | Medium | None | ~200 MB |
| Speech-to-Text (base) | High | Optional | ~500 MB |
| Face Detector | Medium | Optional | ~100 MB |
| Face Recognizer | High | Recommended | ~2 GB |
| Bayesian Tracker | Low | None | ~1 MB |

### Latency

- Face recognition: ~50-100ms per query
- Voice embedding: ~100-200ms per segment
- Speech-to-text: ~500ms-2s (model dependent)
- Bayesian update: <1ms

### Recommendations

1. Use GPU for face recognition (InsightFace)
2. Use "tiny" or "base" Whisper model for real-time STT
3. Adjust VAD threshold based on environment noise
4. Set appropriate decay rate for your use case

---

## Troubleshooting

### Common Issues

**No speech detected**:
- Check microphone permissions: `arecord -l`
- Adjust `vad_threshold` (lower for quieter environments)
- Verify sample rate matches device capability

**Face recognition not matching**:
- Ensure good lighting during enrollment
- Capture diverse poses (especially front)
- Check recognition threshold (default 0.6)

**Bayesian confidence too low**:
- Increase evidence sources (both face and voice)
- Reduce decay rate for persistent identities
- Lower `identity_confidence_threshold` if needed

**Service timeout**:
- Increase service timeout in clients
- Check if all required nodes are running
- Verify topic/service namespaces

**Frame cache miss during identification**:
- Check if frame pinning is working: look for `[PIN] Pinned frame` logs
- Verify face_detector and face_recognizer subscribe to `/frame_cache/pin_timestamp`
- Ensure timestamps match between camera and tracker messages
- Check for excessive processing delay in pipeline

**Identification triggers too often / too rarely**:
- Adjust `identity_confidence_threshold` (default 0.5) - lower = less frequent re-ID
- Adjust `decay_rate_per_second` in Bayesian tracker (default 0.01) - lower = slower decay
- Cooldown intervals prevent spam but don't trigger re-ID
- Re-identification is driven by confidence decay, not periodic timers

**First identification always fails (unknown)**:
- Increase `min_hits` in tracker (default 20 for stable faces)
- Check lighting and camera quality
- Ensure person is fully visible and stable before identification triggers

---

## Future Enhancements

1. **Sound Source Localization**: GCC-PHAT with microphone array
2. **Speaker Diarization**: Distinguish multiple simultaneous speakers
3. **Online Learning**: Update embeddings with new samples
4. **Gait Recognition**: Additional biometric modality
5. **Emotion Recognition**: Affect from voice and face
6. **Active Enrollment**: Robot-initiated enrollment suggestions

---

## References

1. **Resemblyzer**: Wan et al., "Generalized End-to-End Loss for Speaker Verification" (2018)
2. **InsightFace**: Deng et al., "ArcFace: Additive Angular Margin Loss" (2019)
3. **ByteTrack**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (2022)
4. **Whisper**: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (2022)
5. **Bayesian Filtering**: Thrun et al., "Probabilistic Robotics" (2005)
