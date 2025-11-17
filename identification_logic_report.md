# Report: Face and Voice Identification Logic

**Last Updated**: After Bayesian tracker as single source of truth (2025-11-17)

## 1. FACE IDENTIFICATION - Trigger Conditions

Face identification is triggered in `requires_identification()` (line 281) when:

1. **NEW TRACK** (line 296): `track.is_new_track` AND `last_identification_request_time is None`
   - Track must have 20 consecutive frames before `is_new_track` becomes true (min_hits=20)
2. **LOW IDENTITY CONFIDENCE** (line 320): Bayesian tracker confidence < `identity_confidence_threshold` (default 0.5)
   - Confidence decays automatically over time via `apply_tracker_decay()` (1 Hz)
   - When decayed confidence drops below threshold, re-identification triggers naturally

**Key Features**:
- **No periodic re-identification**: Relies on confidence decay to trigger re-ID
- Uses `last_identification_request_time` for cooldown (prevents spam)
- Cooldown periods: 15s for known persons, 5s for unknown
- min_hits increased to 20 for more stable face images

---

## 2. FRAME PINNING MECHANISM (NEW)

### Problem Solved
Frame cache misses occurred when identification processing took longer than cache duration.

### Solution
Explicit frame pinning via ROS topics:

```
PersonStateManager publishes to:
  /frame_cache/pin_timestamp (Header)    - Pin frame before identification
  /frame_cache/release_timestamp (Header) - Release after completion

FaceDetector and FaceRecognizer subscribe and maintain:
  self.pinned_frames = {timestamp_ns: (Image, pin_time)}
```

### Flow
1. PSM decides to identify track T with timestamp X
2. PSM publishes X to `/frame_cache/pin_timestamp`
3. Both face_detector and face_recognizer pin that frame
4. PSM calls detect → embed → recognize services
5. PSM publishes X to `/frame_cache/release_timestamp`
6. Safety net: pinned frames auto-expire after 30 seconds

---

## 3. VOICE IDENTIFICATION - Trigger Conditions

Voice identification is triggered in `speech_segment_callback()` (line 724) whenever:
- A `SpeechSegment` message is received on `/microphone/speech_segment`
- Voice recognition is enabled (parameter check)
- At least one person is being tracked

**No gating/throttling mechanisms** - voice recognition runs on every speech segment.

---

## 4. HOW THEY ARE COMBINED - Bayesian Fusion

Both modalities update a shared `TemporalBayesianIdentity` tracker per track_id:

```python
Face: tracker.update_face(face_scores)
Voice: tracker.update_voice(voice_scores)
```

The Bayesian tracker applies:
```
P(identity | evidence) ∝ P(evidence | identity) × P(identity)
```

History is now limited to last 100 entries to prevent memory bloat.

---

## 5. REMAINING PROBLEMS TO ADDRESS

### Problem 1: Voice Evidence Applied to ALL Tracks (lines 757-783)
```python
# For single speaker scenario: Update all tracked persons' Bayesian trackers
for track_id in list(self.identity_trackers.keys()):
    tracker = self.identity_trackers[track_id]
    tracker.update_voice(voice_scores)  # SAME voice evidence for ALL tracks!
```

**Issue**: When voice is detected, the same voice scores are applied to every tracked person's Bayesian belief. If you have 3 people tracked, they all get updated with the same voice evidence, which is wrong.

### ~~Problem 2: Bayesian Identity NOT Actually Used for Final State~~ FIXED
```python
# OLD (BROKEN):
if identity_result['match_found']:
    return {
        'success': True,
        'identity': identity_result['person_name'],           # Uses raw face result
        'confidence': identity_result['similarity_score'],    # NOT Bayesian confidence
        'person_id': identity_result['person_id']
    }

# NEW (FIXED):
bayesian_identity, bayesian_conf = tracker.get_identity()
if bayesian_identity != 'unknown':
    return {
        'success': True,
        'identity': person_id,          # person_id (int) from Bayesian tracker
        'confidence': bayesian_conf,    # Bayesian confidence
        'person_id': person_id
    }
```

**FIXED**: Now returns Bayesian tracker identity (person_id as int) and Bayesian confidence. State stores person_id internally, names are fetched from database only for display.

### Problem 3: Known Person IDs List Starts Empty (line 104)
```python
self.known_person_ids = []  # Will be populated from database
```

**Issue**: The comment says "Will be populated from database" but there's no code that actually populates it. New person IDs are added reactively when they appear in face recognition results.

### ~~Problem 4: Voice Updates Don't Affect Main Identity Reliably~~ FIXED
```python
# OLD (BROKEN):
# Update main identity from fused Bayesian belief
if identity != 'unknown' and confidence > state['identity_confidence']:
    state['identity'] = str(identity) if isinstance(identity, str) else self.get_person_name(identity)

# NEW (FIXED):
# Voice callback only updates Bayesian tracker, not main state directly
tracker.update_voice(voice_scores)
identity, confidence = tracker.get_identity()
state['voice_identity'] = identity if isinstance(identity, int) else -1  # For monitoring only

# Main identity is synced from Bayesian tracker in apply_tracker_decay() (1 Hz):
identity, confidence = tracker.get_identity()
if identity != 'unknown':
    state['identity'] = identity  # person_id (int)
    state['person_id'] = identity
state['identity_confidence'] = confidence
```

**FIXED**: Voice recognition only updates the Bayesian tracker. The `apply_tracker_decay()` function (1 Hz timer) synchronizes state with the Bayesian tracker, ensuring single source of truth. Both face and voice confidences are now Bayesian confidences, making them comparable.

### ~~Problem 5: Type Mismatch in Identity Storage~~ FIXED
```python
# OLD (BROKEN):
state['voice_identity'] = str(identity) if identity != 'unknown' else 'unknown'  # Could be int like 42
state['identity'] = str(identity) if isinstance(identity, str) else self.get_person_name(identity)  # get_person_name returns "person_42"

# NEW (FIXED):
# State now stores person_id (int) consistently
state['identity'] = -1  # person_id (int) or -1 for unknown
state['voice_identity'] = identity if isinstance(identity, int) else -1  # person_id (int) or -1

# Names are fetched from DB only for display (in publish_state_array):
person_id = state['identity']
if person_id != -1:
    person_state.identity = self.get_person_name(person_id)  # Now queries DB with caching
else:
    person_state.identity = 'unknown'
```

**FIXED**: State stores `person_id` (int) everywhere, with `-1` meaning unknown. `get_person_name()` now queries the database via `people_db/get_person` service and caches results. Names are resolved only when publishing PersonStateArray messages.

### ~~Problem 6: Decay Timer Updates Confidence Incorrectly~~ FIXED
```python
# OLD (PROBLEMATIC):
def apply_tracker_decay(self):
    for track_id, tracker in self.identity_trackers.items():
        tracker.predict()  # Apply decay
        if track_id in self.person_states:
            identity, confidence = tracker.get_identity()
            state['identity_confidence'] = confidence  # Overwrites face recognition confidence

# NEW (FIXED):
def apply_tracker_decay(self):
    for track_id, tracker in self.identity_trackers.items():
        tracker.predict()  # Apply decay
        if track_id in self.person_states:
            identity, confidence = tracker.get_identity()
            # Synchronize entire identity from Bayesian tracker
            if identity != 'unknown':
                state['identity'] = identity  # person_id (int)
                state['person_id'] = identity
            else:
                state['identity'] = -1
                state['person_id'] = -1
            state['identity_confidence'] = confidence
```

**FIXED**: This is now the DESIRED behavior. Since both face and voice recognition now update the Bayesian tracker (not state directly), this decay timer is the single synchronization point that ensures state always reflects the Bayesian tracker. The confidence is now always Bayesian confidence, making comparisons consistent.

### ~~Problem 7: No Synchronization Between Face and Voice Person IDs~~ FIXED
**OLD Issue**: Face recognition returns person names as strings, while Bayesian tracker uses person_id (int). When storing identity in state:
- Face result stores `person_name` (string) in `state['identity']`
- Voice result tries to convert `person_id` (int) to name via broken `get_person_name()`

**FIXED**: Both face and voice recognition now:
1. Update the Bayesian tracker with similarity scores
2. Bayesian tracker returns person_id (int) or 'unknown'
3. State stores person_id (int) internally, -1 for unknown
4. Names are fetched from database only for publishing
5. GetPerson service client with caching provides actual database names

---

## 6. SUMMARY OF DATA FLOW

```
Face Pipeline (synchronous, on-demand):
PersonTrack → requires_identification() → PIN FRAME → detect_face → generate_embedding → recognize_face → RELEASE FRAME
            → tracker.update_face(all_scores)  ✓ Updates Bayesian
            → Returns Bayesian identity (person_id) and Bayesian confidence  ✓ FIXED
            → State stores person_id (int), -1 for unknown  ✓ FIXED

Voice Pipeline (async, on every speech):
SpeechSegment → generate_voice_embedding → recognize_voice
              → tracker.update_voice() for ALL tracks  ✗ Still needs DoA fix (Problem 1)
              → Only updates Bayesian tracker, NOT state directly  ✓ FIXED

State Synchronization (1 Hz timer):
apply_tracker_decay() → For each track:
                      → tracker.predict() (apply decay)
                      → identity, confidence = tracker.get_identity()
                      → state['identity'] = person_id (int) or -1  ✓ Single source of truth
                      → state['identity_confidence'] = Bayesian confidence

Publishing (10 Hz):
publish_state_array() → For each state:
                      → person_name = get_person_name(person_id)  ✓ DB lookup with caching
                      → PersonState.identity = person_name (string for ROS message)
```

---

## 7. RECENT IMPROVEMENTS (2025-11-17)

### Phase 1: Frame pinning and cooldown improvements
1. **min_hits increased to 20**: More stable face before first identification (~0.7s at 30fps)
2. **Frame pinning mechanism**: Explicit cache control prevents frame misses
3. **Request-based cooldown**: Uses request time, not result time, to prevent spam
4. **Removed max_attempts limit**: Continuous re-identification with cooldown
5. **History size limit**: Bayesian tracker history capped at 100 entries
6. **Removed periodic re-identification**: Relies on Bayesian confidence decay instead of timer-based triggers

### Phase 2: Bayesian tracker as single source of truth (LATEST)
7. **Face recognition uses Bayesian output**: Returns person_id and Bayesian confidence, not raw face result
8. **State stores person_id (int)**: No more string names internally, -1 for unknown
9. **Database name lookup with caching**: GetPerson service client fetches actual names from DB
10. **Voice updates tracker only**: No direct state modification, ensures single source of truth
11. **State synchronized from tracker**: apply_tracker_decay() syncs identity and confidence from Bayesian tracker
12. **Names resolved at publish time**: PersonStateArray contains actual database names, not internal IDs
13. **Enrollment updates tracker**: New persons get strong belief (0.9) in their Bayesian tracker immediately

---

## 8. RECOMMENDATIONS FOR REMAINING ISSUES

1. **Fix voice-to-track association** (Problem 1): Use sound localization (DoA) or single-speaker assumption with explicit handling
2. ~~**Use Bayesian identity as the source of truth**~~: DONE - Returns fused Bayesian belief (person_id + Bayesian confidence)
3. **Initialize known_person_ids from database** on startup (Problem 3): Query all person IDs from DB at node startup
4. ~~**Consistent identity representation**~~: DONE - Stores person_id (int) everywhere, resolves names only for display
5. ~~**Fix confidence comparison**~~: DONE - Both face and voice use Bayesian confidence, making them comparable

### Remaining Work:
- **Problem 1**: Voice evidence applied to all tracks - needs DoA or speaker identification
- **Problem 3**: Known persons not loaded from DB at startup - needs service call in `__init__()`
