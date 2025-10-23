# Complete Pipeline Analysis: Camera → Annotated Image

**Date:** 2025-10-23
**Purpose:** Step-by-step trace of every function, transformation, and data flow to identify bugs causing false face recognition matches.

---

## 1. CAMERA → YOLO NODE

### Input
- **Topic:** `/camera/image_raw`
- **Message Type:** `sensor_msgs/Image`
- **Format:** BGR8, variable resolution (e.g., 4K)

### YoloNode.image_callback() [`yolo_node.py:57-59`]
```python
def image_callback(self, msg):
    with self.lock:
        self.latest_msg = msg
```
- **Thread Safety:** Uses lock to protect shared state
- **Frame Dropping:** ⚠️ **ISSUE**: Latest frame overwrites previous, drops frames if processing is slow

### YoloNode.process_frame() [`yolo_node.py:61-79`]
```python
cv_image = rnp.numpify(msg)              # Convert ROS msg to numpy
img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # BGR→RGB
results = self.model(img_rgb, size=640)   # YOLO inference at 640px
```
- **Color Space:** BGR → RGB (YOLO expects RGB)
- **Resize:** YOLO internally resizes to 640px (maintains aspect ratio with padding)
- **Output:** Results in original image coordinates

### YoloNode.publish_results() [`yolo_node.py:81-143`]
**Critical Transformation:** YOLO outputs [xmin, ymin, xmax, ymax] but publishes in CENTER+SIZE format:

```python
# For each detection in YOLO results:
det.bbox.center.position.x = float((row.xmin + row.xmax) / 2.0)  # CENTER
det.bbox.center.position.y = float((row.ymin + row.ymax) / 2.0)  # CENTER
det.bbox.size_x = float(row.xmax - row.xmin)                     # WIDTH
det.bbox.size_y = float(row.ymax - row.ymin)                     # HEIGHT

# Class ID stored as STRING
hyp.hypothesis.class_id = str(int(row['class']))  # "0" for person
hyp.hypothesis.score = float(row['confidence'])
```

### Output
- **Topic:** `/yolo/detections`
- **Message Type:** `Detection2DArray`
- **Format:** Center+Size, class_id as string, confidence scores

### Coordinate Verification
✅ **CORRECT**: Center+size format matches ROS standard

---

## 2. YOLO NODE → PERSON TRACKER

### Input
- **Topic:** `/yolo/detections`
- **Message Type:** `Detection2DArray`
- **Format:** Center+size, class_id as string

### PersonTrackerNode.detection_callback() [`person_tracker.py:150-183`]

**Step 1: Filter for persons**
```python
for detection in msg.detections:
    for result in detection.results:
        if result.hypothesis.class_id == 'person' or result.hypothesis.class_id == '0':
```
✅ **CORRECT**: Checks both "person" and "0" string formats

**Step 2: Transform Center+Size → TopLeft+Size**
```python
center_x = detection.bbox.center.position.x
center_y = detection.bbox.center.position.y
size_x = detection.bbox.size_x
size_y = detection.bbox.size_y

# Convert to [x, y, w, h] format (top-left corner)
x = center_x - size_x / 2.0  # TOP-LEFT X
y = center_y - size_y / 2.0  # TOP-LEFT Y
w = size_x                    # WIDTH
h = size_y                    # HEIGHT
```

**Verification:**
- If center_x=300, size_x=100: x = 300 - 50 = 250 ✅
- If center_y=200, size_y=80: y = 200 - 40 = 160 ✅

**Step 3: Create detection list**
```python
person_detections.append({
    'bbox': [x, y, w, h],
    'confidence': conf
})
```

### PersonTrackerNode.update_tracks() [`person_tracker.py:185-238`]

**ByteTrack Two-Stage Association:**

**Stage 1: High-confidence detections (≥0.6)**
```python
high_conf_dets = [d for d in detections if d['confidence'] >= 0.6]
unmatched_tracks, unmatched_dets = self.associate_detections_to_tracks(
    high_conf_dets, self.tracks, 0.3)  # IoU threshold = 0.3
```

**Stage 2: Low-confidence detections (0.1-0.6)**
```python
low_conf_dets = [d for d in detections if 0.1 <= d['confidence'] < 0.6]
unmatched_tracks_low, _ = self.associate_detections_to_tracks(
    low_conf_dets, remaining_tracks, 0.3)
```

**Track Management:**
- Create new tracks for unmatched high-confidence detections
- Mark unmatched tracks as missed
- Remove tracks with `frames_since_update > 30`
- Confirm tracks after `hits >= 3`

### compute_iou() [`person_tracker.py:69-99`]
```python
def compute_iou(bbox1, bbox2):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    box1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    box2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]

    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersection
    return intersection / union if union > 0 else 0.0
```
✅ **CORRECT**: Standard IoU calculation

### PersonTrackerNode.publish_tracks() [`person_tracker.py:286-319`]
```python
# Only publish CONFIRMED tracks (hits >= 3)
if not track.is_confirmed:
    continue

track_msg.track_id = track.track_id  # PERSISTENT ID
track_msg.bbox_x = float(track.bbox[0])  # TOP-LEFT X
track_msg.bbox_y = float(track.bbox[1])  # TOP-LEFT Y
track_msg.bbox_w = float(track.bbox[2])  # WIDTH
track_msg.bbox_h = float(track.bbox[3])  # HEIGHT
track_msg.tracking_confidence = float(track.tracking_conf)
track_msg.is_new_track = (track.hits == self.min_hits)  # True when hits==3
```

### Output
- **Topic:** `/person_tracker/tracks`
- **Message Type:** `PersonTrackArray`
- **Format:** [x, y, w, h] with persistent track_id
- **Timestamp:** Inherits header from original YOLO detections

### Coordinate Verification
✅ **CORRECT**: Coordinates properly converted back to top-left format

---

## 3. PERSON TRACKER → PERSON STATE MANAGER

### Input
- **Topic:** `/person_tracker/tracks`
- **Message Type:** `PersonTrackArray`

### PersonStateManager.track_callback() [`person_state_manager.py:99-134`]

**For NEW tracks:**
```python
if track_id not in self.persons:
    self.persons[track_id] = {
        'track_id': track_id,
        'identity': '',  # EMPTY - not yet identified
        'identity_confidence': 0.0,
        'bbox_x': track.bbox_x,  # Store bbox
        'bbox_y': track.bbox_y,
        'bbox_w': track.bbox_w,
        'bbox_h': track.bbox_h,
        'first_seen': current_time,
        'last_seen': current_time,
        'requires_identification': True,  # TRIGGER FACE RECOGNITION
        'tracking_confidence': track.tracking_confidence,
        'frames_since_last_seen': track.frames_since_last_seen
    }
```

**For EXISTING tracks:**
```python
person['bbox_x'] = track.bbox_x  # UPDATE bbox
person['bbox_y'] = track.bbox_y
person['bbox_w'] = track.bbox_w
person['bbox_h'] = track.bbox_h
person['last_seen'] = current_time
person['tracking_confidence'] = track.tracking_confidence
# NOTE: Does NOT reset 'identity' or 'requires_identification'
```

### PersonStateManager.publish_states() [`person_state_manager.py:223-250`]
- **Timer:** Publishes at 10 Hz
- **Content:** All person states with statistics

### Output
- **Topic:** `/person_state/all`
- **Message Type:** `PersonStateArray`
- **Update Rate:** 10 Hz (independent of frame rate)

---

## 4. FACE RECOGNITION NODE - FRAME CACHING

### Input (Parallel Streams)
1. `/camera/image_raw` - Full resolution images
2. `/person_tracker/tracks` - Track updates
3. `/person_state/all` - State updates

### FaceRecognitionConditional.image_callback() [`face_recognition_conditional.py:180-187`]
```python
def image_callback(self, msg):
    image = rnp.numpify(msg)  # Convert to numpy
    timestamp = self.get_clock().now()  # GET CURRENT TIME
    self.frame_cache.append((timestamp, image, msg.header))
```

⚠️ **POTENTIAL ISSUE**: Timestamp is current time, not from msg.header.stamp
- Frame cache stores last 10 frames with timestamps
- **FULL RESOLUTION** images stored (e.g., 4K)

### FaceRecognitionConditional.track_callback() [`face_recognition_conditional.py:189-211`]

**Trigger 1: Auto-identify new tracks**
```python
if self.auto_identify_new_tracks:
    for track in msg.tracks:
        if track.is_new_track and track_id not in self.identification_queue:
            self.identification_queue[track_id] = time.time()
```

**Trigger 2: Process queued identifications**
```python
for track_id in list(self.identification_queue.keys()):
    if track_id in tracks_by_id:
        track = tracks_by_id[track_id]
        queue_time = self.identification_queue.get(track_id)
        if queue_time and time.time() - queue_time >= 0.5:  # 0.5s debounce
            self._process_identification(track_id, track)
```

### FaceRecognitionConditional.state_callback() [`face_recognition_conditional.py:213-221`]
```python
for person in msg.persons:
    if person.requires_identification and track_id not in self.identification_queue:
        self.identification_queue[track_id] = time.time()
```

---

## 5. FACE RECOGNITION - FACE DETECTION

### FaceRecognitionConditional._process_identification() [`face_recognition_conditional.py:336-467`]

**Step 1: Get frame**
```python
timestamp, image, header = self.frame_cache[-1]  # LATEST FRAME
```

🚨 **CRITICAL BUG FOUND!**
- Uses LATEST frame from cache (`[-1]`)
- But track bbox is from OLDER frame (when track was detected)
- If person moved between frames, bbox won't match person's current position!

**Example:**
1. Frame N: Person at position (100, 100), YOLO detects, tracker publishes bbox
2. Frame N+1, N+2, N+3 arrive, person moves to (150, 100)
3. Face recognition gets track with bbox (100, 100) but uses Frame N+3
4. Face detection looks at wrong position!

**Step 2: Extract person bbox**
```python
person_bbox = [track.bbox_x, track.bbox_y, track.bbox_w, track.bbox_h]
```

**Step 3: Detect face in ROI**
```python
face_bbox = self._detect_face_in_roi(image, person_bbox)
```

### FaceRecognitionConditional._detect_face_in_roi() [`face_recognition_conditional.py:233-334`]

**Step 1: Extract person ROI**
```python
x, y, w, h = bbox  # person_bbox in [x, y, w, h] format
x_min = max(0, int(x))
y_min = max(0, int(y))
x_max = min(image.shape[1], int(x + w))
y_max = min(image.shape[0], int(y + h))

person_roi = image[y_min:y_max, x_min:x_max]  # Extract ROI
```
✅ **CORRECT**: Proper ROI extraction with bounds checking

**Step 2: Prepare for UltraFace detector**
```python
detector_h, detector_w = 240, 320
resized_roi = cv2.resize(person_roi, (detector_w, detector_h))
resized_roi_rgb = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
```

⚠️ **POTENTIAL ISSUE**: Aspect ratio distortion
- Person ROI has arbitrary aspect ratio (e.g., 200x400 = 1:2)
- Resized to fixed 320x240 (4:3 aspect ratio)
- Image is STRETCHED/SQUASHED if aspect ratios don't match
- However, normalized coordinates should still map correctly

**Step 3: Normalize and run detection**
```python
input_data = np.expand_dims(resized_roi_rgb.transpose(2, 0, 1), 0).astype(np.float32)
input_data = (input_data - 127.0) / 128.0  # Normalize to [-1, 1]

outputs = self.face_detector_session.run(None, {self.face_detector_input_name: input_data})
scores = outputs[0]  # Shape: (1, num_boxes, 2)
boxes = outputs[1]   # Shape: (1, num_boxes, 4)
```

**Step 4: Find best face**
```python
face_scores = scores[:, 1]  # Get face scores (not background)
best_idx = np.argmax(face_scores)
best_score = float(face_scores[best_idx])

if best_score < self.face_detection_threshold:  # 0.7
    return None
```

**Step 5: Transform coordinates**
```python
face_box_norm = boxes[best_idx]  # Normalized [x1, y1, x2, y2] in [0, 1]
face_x1_norm, face_y1_norm, face_x2_norm, face_y2_norm = face_box_norm

# Scale normalized coords by ORIGINAL ROI dimensions (NOT 320x240)
face_x1_roi = int(face_x1_norm * (x_max - x_min))
face_y1_roi = int(face_y1_norm * (y_max - y_min))
face_x2_roi = int(face_x2_norm * (x_max - x_min))
face_y2_roi = int(face_y2_norm * (y_max - y_min))

# Convert to full image coordinates
face_x1 = x_min + face_x1_roi
face_y1 = y_min + face_y1_roi
face_x2 = x_min + face_x2_roi
face_y2 = y_min + face_y2_roi

face_bbox = [face_x1, face_y1, face_x2 - face_x1, face_y2 - face_y1]
```

**Coordinate Transform Verification:**
- UltraFace outputs normalized [0, 1] coordinates relative to 320x240 input
- Multiplying by original ROI dimensions correctly scales back
- Example:
  - Person ROI: 100x200 pixels (in full image)
  - Face detected at normalized [0.25, 0.5, 0.75, 0.8]
  - face_x1_roi = 0.25 * 100 = 25
  - face_y1_roi = 0.5 * 200 = 100
  - face_x2_roi = 0.75 * 100 = 75
  - face_y2_roi = 0.8 * 200 = 160
  - Face in ROI: [25, 100, 50, 60]
  - If ROI starts at (50, 30) in full image:
    - Face in full image: [75, 130, 50, 60] ✅

✅ **CORRECT**: Coordinate transformation is mathematically sound

**Step 6: Validate face size**
```python
face_width = face_bbox[2]
face_height = face_bbox[3]
face_size = min(face_width, face_height)  # Minimum dimension

if face_size < self.min_face_size:  # 80 pixels
    return None
```
✅ **CORRECT**: Rejects faces < 80px (minimum dimension)

---

## 6. FACE RECOGNITION - EMBEDDING EXTRACTION

### FaceRecognitionConditional._process_identification() (continued)

**Step 4: Extract face crop from FULL RESOLUTION image**
```python
face_x, face_y, face_w, face_h = face_bbox
face_x_min = max(0, int(face_x))
face_y_min = max(0, int(face_y))
face_x_max = min(image.shape[1], int(face_x + face_w))
face_y_max = min(image.shape[0], int(face_y + face_h))

face_crop = image[face_y_min:face_y_max, face_x_min:face_x_max]
```
✅ **CORRECT**: Extracts face from full resolution image

**Step 5: Get embedding**
```python
embedding = self._get_face_embedding(face_crop)
```

### FaceRecognitionConditional._get_face_embedding() [`face_recognition_conditional.py:469-506`]

```python
h, w = self.embedding_input_shape[2], self.embedding_input_shape[3]  # 112, 112

# Convert BGR to RGB
face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

# Resize to 112x112
resized = cv2.resize(face_rgb, (w, h))

# ArcFace normalization: (pixel - 127.5) / 127.5
input_data = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)
input_data = (input_data - 127.5) / 127.5  # Range: [-1, 1]

# Run inference
outputs = self.embedding_session.run(None, {self.embedding_input_name: input_data})
embedding = outputs[0][0].flatten()  # 512-dimensional vector

# Normalize embedding (L2 normalization)
embedding = embedding / np.linalg.norm(embedding)
return embedding
```

**Preprocessing Verification:**
- Input range: [0, 255] (uint8)
- After normalization: [-1, 1] (float32)
- This matches ArcFace requirements ✅

**Embedding Verification:**
- Output: 512-dimensional vector
- L2 normalized (norm = 1.0)
- Standard for face recognition ✅

---

## 7. DATABASE RECOGNITION

### Service Call: RecognizeFace

**Request:**
```python
request.face_embedding = embedding.tolist()  # 512 floats
request.threshold = 0.75
```

### PeopleDatabaseNode.recognize_face_callback() [`people_database_node.py:94-129`]

```python
embedding = np.array(request.face_embedding, dtype=np.float32)
threshold = request.threshold if request.threshold > 0 else 0.6

result = self.db.find_similar_face(embedding, threshold, logger=self.get_logger())

if result:
    match, similarity = result
    response.found = True
    response.person_id = match['id']
    response.name = match['name']
    response.similarity = float(similarity)  # ACTUAL SIMILARITY
    # Auto-update last_seen
    self.db.update_last_seen(match['id'])
```

### PeopleDatabase.find_similar_face() [`people_database.py:151-219`]

```python
# Get all stored embeddings
embeddings = self.get_all_embeddings()  # Returns [(person_id, embedding), ...]

# Normalize query embedding
query_norm = query_embedding / np.linalg.norm(query_embedding)

best_similarity = -1

for person_id, stored_embedding in embeddings:
    # Normalize stored embedding
    stored_norm = stored_embedding / np.linalg.norm(stored_embedding)

    # Cosine similarity
    similarity = np.dot(query_norm, stored_norm)

    if similarity > best_similarity:
        best_similarity = similarity
        best_match_id = person_id

if best_similarity >= threshold:
    return (self.get_person_by_id(best_match_id), best_similarity)
else:
    return None
```

**Similarity Calculation Verification:**
- Cosine similarity = dot product of normalized vectors
- Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
- For face embeddings: typically 0.3-0.6 for different people, 0.85+ for same person
- ✅ **CORRECT**: Standard cosine similarity

**Double Normalization Check:**
- Embedding normalized in `_get_face_embedding()`: `embedding / norm(embedding)`
- Then normalized again in database: `query_norm = query_embedding / norm(query_embedding)`
- ✅ **NOT A BUG**: Normalizing a normalized vector returns the same vector (norm=1, so divide by 1)

---

## 8. VISUALIZATION

### VisualizationNode.state_callback() [`visualization_node.py:89-110`]

```python
annotated = self.current_image.copy()

for person in msg.persons:
    self._draw_person(annotated, person)
```

### VisualizationNode._draw_person() [`visualization_node.py:112-208`]

```python
x = int(person.bbox_x)
y = int(person.bbox_y)
w = int(person.bbox_w)
h = int(person.bbox_h)

x_max = x + w
y_max = y + h

# Draw bounding box
cv2.rectangle(image, (x, y), (x_max, y_max), color, thickness)

# Draw labels (track_id, identity, confidence)
# Draw confidence bar
```

✅ **CORRECT**: Draws person bbox properly

**Note:** Visualization does NOT draw face bounding boxes (only person boxes)

---

## SUMMARY OF BUGS FOUND

### 🚨 CRITICAL BUG #1: Frame/BBox Temporal Mismatch

**Location:** `face_recognition_conditional.py:345`

**Problem:**
```python
timestamp, image, header = self.frame_cache[-1]  # Uses LATEST frame
person_bbox = [track.bbox_x, track.bbox_y, track.bbox_w, track.bbox_h]  # BBox from OLD frame
```

**Impact:**
- If person is moving, bbox from old frame doesn't match person's position in new frame
- Face detection searches in wrong location
- May detect faces from DIFFERENT people in the frame
- Explains high false match rates!

**Example Scenario:**
1. Frame N (t=0.0s): Person A at x=100, YOLO detects, tracker creates track with bbox [100, 50, 80, 200]
2. Frame N+1 (t=0.033s): Person A moves to x=120, YOLO updates
3. Frame N+2 (t=0.066s): Tracker publishes updated bbox [120, 50, 80, 200]
4. Frame N+3 (t=0.100s): In cache, person now at x=140
5. Face recognition gets track with bbox [120, 50, 80, 200] but applies to Frame N+3
6. Face detection searches at x=120 but person is at x=140!
7. If another person is at x=120, their face gets detected instead!

**Fix Required:**
Either:
1. Match frame timestamp with track timestamp
2. OR use timestamp-matched frame from cache
3. OR implement motion prediction in tracker

### ⚠️ POTENTIAL BUG #2: No Motion Prediction

**Location:** `person_tracker.py:56-61`

**Problem:**
```python
def predict(self):
    """Predict next position (for ByteTrack, just keep current position)."""
    self.age += 1
    self.frames_since_update += 1
    self.tracking_conf *= 0.95
    # No bbox update!
```

**Impact:**
- When detection is missed for a few frames, bbox doesn't update
- Bbox lags behind actual person position
- Compounds the temporal mismatch issue

**Standard Solution:**
- Implement Kalman filter or simple motion model
- Predict bbox based on velocity

### ⚠️ WARNING: Aspect Ratio Distortion

**Location:** `face_recognition_conditional.py:259`

**Problem:**
```python
resized_roi = cv2.resize(person_roi, (320, 240))  # May distort
```

**Impact:**
- Person ROI (e.g., 200x400) stretched to 320x240
- Face detector sees distorted image
- May affect detection accuracy, but coordinates should map correctly

**Better Solution:**
- Resize with padding to maintain aspect ratio
- Or use letterboxing

---

## ROOT CAUSE ANALYSIS: Why 0.99+ Similarity for Different People?

### Hypothesis 1: Small Face Crops ✅ CONFIRMED
- 28x41 pixel faces lack detail for discriminative embeddings
- All tiny faces look similar when resized to 112x112
- **Status:** FIXED with 80px minimum threshold

### Hypothesis 2: Frame/BBox Mismatch 🚨 HIGHLY LIKELY
- Wrong person's face extracted due to temporal mismatch
- Face from person B matched against person A's database entry
- **Status:** NEEDS FIX

### Hypothesis 3: Background Contamination ⚠️ POSSIBLE
- If face detection fails, returns bbox including background
- Embedding captures background features instead of face
- **Status:** Should be mitigated by face detection confidence threshold (0.7)

---

## RECOMMENDED FIXES

### Fix #1: Timestamp-Matched Frame Selection (HIGH PRIORITY)
```python
def _get_synchronized_frame(self, track_header):
    """Get frame with matching or closest timestamp to track."""
    track_time_ns = track_header.stamp.sec * 1e9 + track_header.stamp.nanosec

    best_match = None
    min_time_diff = float('inf')

    for cached_timestamp, cached_image, cached_header in self.frame_cache:
        cached_time_ns = cached_header.stamp.sec * 1e9 + cached_header.stamp.nanosec
        time_diff = abs(cached_time_ns - track_time_ns)

        if time_diff < min_time_diff:
            min_time_diff = time_diff
            best_match = (cached_timestamp, cached_image, cached_header)

    if best_match and min_time_diff < 200e6:  # Within 200ms
        return best_match
    else:
        self.get_logger().warning(f'No matching frame found (time_diff={min_time_diff/1e6:.1f}ms)')
        return self.frame_cache[-1]  # Fallback to latest
```

### Fix #2: Add Motion Prediction to Tracker (MEDIUM PRIORITY)
Implement simple constant velocity model in `Track.predict()`:
```python
def predict(self):
    if hasattr(self, 'velocity'):
        # Predict next position
        self.bbox[0] += self.velocity[0]
        self.bbox[1] += self.velocity[1]
    self.age += 1
    self.frames_since_update += 1
    self.tracking_conf *= 0.95

def update(self, bbox, detection_conf, frame_id):
    # Calculate velocity
    if hasattr(self, 'bbox'):
        self.velocity = [
            bbox[0] - self.bbox[0],
            bbox[1] - self.bbox[1]
        ]
    self.bbox = np.array(bbox, dtype=np.float32)
    # ... rest of update
```

### Fix #3: Aspect-Ratio Preserving Resize (LOW PRIORITY)
```python
def _resize_with_padding(image, target_size):
    """Resize image maintaining aspect ratio with padding."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Create canvas with padding
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas, (x_offset, y_offset, scale)
```

---

## CONCLUSION

The primary issue causing high false match rates is **temporal mismatch between track bboxes and face recognition frames**. When a person moves, their old bbox is applied to a newer frame, causing face detection to extract faces from the wrong location (potentially from different people in the scene).

**Priority:**
1. **HIGH**: Fix frame synchronization (Fix #1)
2. **MEDIUM**: Verify 80px minimum face size is working
3. **MEDIUM**: Add motion prediction to tracker (Fix #2)
4. **LOW**: Aspect ratio preservation (Fix #3)

After implementing Fix #1, the system should correctly match faces with their corresponding person bboxes, dramatically improving recognition accuracy.
