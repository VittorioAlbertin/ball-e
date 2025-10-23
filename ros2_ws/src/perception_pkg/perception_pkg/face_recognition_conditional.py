#!/usr/bin/env python3
"""
Conditional Face Recognition Node for Ball-e Robot

This node performs face recognition ON-DEMAND instead of every frame.
It processes faces only when triggered by:
1. New track appears (is_new_track == true)
2. Explicit identification request
3. Re-identification timer expires

Uses lightweight UltraFace detector for accurate face localization within person ROI.
Optimized for <200ms processing time per identification.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import ros2_numpy as rnp
from sensor_msgs.msg import Image
from msgs_interfaces.msg import PersonTrackArray, PersonStateArray, IdentityUpdate
from msgs_interfaces.srv import RecognizeFace, UpdateIdentity
from collections import deque
import time
import onnxruntime as ort
import os
import urllib.request
from pathlib import Path


class FaceRecognitionConditional(Node):
    """On-demand face recognition node with proper face detection."""

    def __init__(self):
        super().__init__('face_recognition_conditional')

        # Declare parameters
        self.declare_parameter('face_detector_model_path', '/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/version-RFB-320.onnx')
        self.declare_parameter('face_detector_model_url', 'https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx')
        self.declare_parameter('embedding_model_path', '/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/facenet.onnx')
        self.declare_parameter('embedding_model_url', 'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx')
        self.declare_parameter('face_detection_threshold', 0.7)  # Face detection confidence threshold
        self.declare_parameter('recognition_threshold', 0.75)  # Face recognition threshold
        self.declare_parameter('min_face_size', 80)  # Minimum face pixels (changed from 40 to 80)
        self.declare_parameter('frame_cache_size', 10)  # Cache last N frames
        self.declare_parameter('reidentification_interval', 30.0)  # Re-ID every 30 seconds
        self.declare_parameter('auto_identify_new_tracks', True)  # Auto-identify new tracks

        self.face_detector_model_path = self.get_parameter('face_detector_model_path').value
        self.face_detector_model_url = self.get_parameter('face_detector_model_url').value
        self.embedding_model_path = self.get_parameter('embedding_model_path').value
        self.embedding_model_url = self.get_parameter('embedding_model_url').value
        self.face_detection_threshold = self.get_parameter('face_detection_threshold').value
        self.recognition_threshold = self.get_parameter('recognition_threshold').value
        self.min_face_size = self.get_parameter('min_face_size').value
        self.frame_cache_size = self.get_parameter('frame_cache_size').value
        self.reidentification_interval = self.get_parameter('reidentification_interval').value
        self.auto_identify_new_tracks = self.get_parameter('auto_identify_new_tracks').value

        # Logging
        self.get_logger().info('Conditional Face Recognition Node initializing...')
        self.get_logger().info(f'  face_detection_threshold: {self.face_detection_threshold}')
        self.get_logger().info(f'  recognition_threshold: {self.recognition_threshold}')
        self.get_logger().info(f'  min_face_size: {self.min_face_size}px')
        self.get_logger().info(f'  frame_cache_size: {self.frame_cache_size}')
        self.get_logger().info(f'  reidentification_interval: {self.reidentification_interval}s')
        self.get_logger().info(f'  auto_identify_new_tracks: {self.auto_identify_new_tracks}')

        # Download models if needed
        self._ensure_model_exists(self.face_detector_model_path, self.face_detector_model_url, "face detector")
        self._ensure_model_exists(self.embedding_model_path, self.embedding_model_url, "face embedding")

        # Load face detector model (UltraFace)
        try:
            self.get_logger().info("Loading face detector model (UltraFace)...")
            self.face_detector_session = ort.InferenceSession(self.face_detector_model_path)
            self.face_detector_input_name = self.face_detector_session.get_inputs()[0].name
            self.face_detector_input_shape = self.face_detector_session.get_inputs()[0].shape
            self.get_logger().info(f'✓ Loaded face detector: {self.face_detector_model_path}')
            self.get_logger().info(f'  Input shape: {self.face_detector_input_shape}')
        except Exception as e:
            self.get_logger().error(f"Failed to load face detector model: {e}")
            raise

        # Load embedding model
        try:
            self.get_logger().info("Loading face embedding model...")
            self.embedding_session = ort.InferenceSession(self.embedding_model_path)
            self.embedding_input_name = self.embedding_session.get_inputs()[0].name
            self.embedding_input_shape = self.embedding_session.get_inputs()[0].shape
            self.get_logger().info(f'✓ Loaded embedding model: {self.embedding_model_path}')
            self.get_logger().info(f'  Input shape: {self.embedding_input_shape}')
        except Exception as e:
            self.get_logger().error(f"Failed to load embedding model: {e}")
            raise

        # Frame cache for async processing (stores full resolution images)
        self.frame_cache = deque(maxlen=self.frame_cache_size)

        # Track last identification times for re-identification
        self.last_identification_time = {}  # {track_id: timestamp}

        # Pending identification requests (use dict to prevent duplicates)
        self.identification_queue = {}  # {track_id: timestamp}

        # Service clients
        self.face_recognition_client = self.create_client(RecognizeFace, 'people_db/recognize_face')
        self.update_identity_client = self.create_client(UpdateIdentity, '/person_state/update_identity')

        # Wait for services
        self.get_logger().info('Waiting for face recognition service...')
        while not self.face_recognition_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('  Still waiting for people_db/recognize_face...')

        self.get_logger().info('Waiting for update identity service...')
        while not self.update_identity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('  Still waiting for person_state/update_identity...')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.track_sub = self.create_subscription(
            PersonTrackArray,
            '/person_tracker/tracks',
            self.track_callback,
            10
        )

        self.state_sub = self.create_subscription(
            PersonStateArray,
            '/person_state/all',
            self.state_callback,
            10
        )

        # Publishers
        self.identity_pub = self.create_publisher(
            IdentityUpdate,
            '/face_recognition/identity_update',
            10
        )

        # Timer for periodic re-identification checks
        self.reidentification_timer = self.create_timer(
            5.0,  # Check every 5 seconds
            self.check_reidentification
        )

        self.get_logger().info('Conditional Face Recognition Node started')

    def _ensure_model_exists(self, model_path, model_url, model_name):
        """Download model if it doesn't exist locally."""
        if os.path.exists(model_path):
            self.get_logger().info(f"✓ {model_name} model found: {model_path}")
            return

        self.get_logger().info(f"⚠ {model_name} model not found, downloading from {model_url}")

        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)

            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 50 == 0:
                    self.get_logger().info(f"  Downloading {model_name}: {percent:.1f}%")

            urllib.request.urlretrieve(model_url, model_path, reporthook=report_progress)
            self.get_logger().info(f"✓ Successfully downloaded {model_name} model")

        except Exception as e:
            self.get_logger().error(f"Failed to download {model_name} model: {e}")
            raise

    def image_callback(self, msg):
        """Cache incoming camera frames (FULL RESOLUTION)."""
        try:
            image = rnp.numpify(msg)
            timestamp = self.get_clock().now()
            self.frame_cache.append((timestamp, image, msg.header))
        except Exception as e:
            self.get_logger().error(f"Failed to cache image: {e}")

    def track_callback(self, msg):
        """Process track updates and trigger identification for new tracks or queued tracks."""
        # Process new tracks if auto-identify is enabled
        if self.auto_identify_new_tracks:
            for track in msg.tracks:
                if track.is_new_track and track.track_id not in self.identification_queue:
                    track_id = track.track_id
                    self.get_logger().info(f'New track detected: {track_id}, queuing for identification')
                    self.identification_queue[track_id] = time.time()

        # Process queued identifications (only if not already processing)
        tracks_by_id = {track.track_id: track for track in msg.tracks}
        for track_id in list(self.identification_queue.keys()):
            if track_id in tracks_by_id:
                track = tracks_by_id[track_id]
                # Check if we're not already processing this track recently
                queue_time = self.identification_queue.get(track_id)
                if queue_time and time.time() - queue_time >= 0.5:  # Wait at least 0.5s between requests
                    self.get_logger().info(f'Processing queued identification for track_id={track_id}')
                    self._process_identification(track_id, track)
                    # Remove from queue after processing (if still exists)
                    if track_id in self.identification_queue:
                        del self.identification_queue[track_id]

    def state_callback(self, msg):
        """Monitor person states for identification requests."""
        for person in msg.persons:
            track_id = person.track_id

            # Check if identification is requested and not already queued
            if person.requires_identification and track_id not in self.identification_queue:
                self.get_logger().info(f'Identification requested for track_id={track_id}')
                self.identification_queue[track_id] = time.time()

    def check_reidentification(self):
        """Check if any known persons need re-identification."""
        current_time = time.time()

        for track_id, last_time in list(self.last_identification_time.items()):
            if current_time - last_time > self.reidentification_interval:
                if track_id not in self.identification_queue:
                    self.get_logger().info(f'Re-identification interval expired for track_id={track_id}')
                    self.identification_queue[track_id] = current_time

    def _detect_face_in_roi(self, image, bbox):
        """
        Detect face within person bounding box using UltraFace detector.

        Args:
            image: Full resolution image
            bbox: Person bounding box [x, y, w, h]

        Returns:
            Face bounding box [x, y, w, h] in image coordinates, or None if no face detected
        """
        try:
            # Extract person ROI
            x, y, w, h = bbox
            x_min = max(0, int(x))
            y_min = max(0, int(y))
            x_max = min(image.shape[1], int(x + w))
            y_max = min(image.shape[0], int(y + h))

            if x_max <= x_min or y_max <= y_min:
                return None

            person_roi = image[y_min:y_max, x_min:x_max]

            # Prepare input for face detector (UltraFace expects 320x240 RGB)
            detector_h, detector_w = 240, 320
            resized_roi = cv2.resize(person_roi, (detector_w, detector_h))
            resized_roi_rgb = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)

            # Normalize and prepare input
            input_data = np.expand_dims(resized_roi_rgb.transpose(2, 0, 1), 0).astype(np.float32)
            input_data = (input_data - 127.0) / 128.0

            # Run face detection
            outputs = self.face_detector_session.run(None, {self.face_detector_input_name: input_data})

            # Parse outputs (UltraFace model outputs: scores, boxes)
            # Output 0: scores shape (1, num_boxes, 2) - [background_score, face_score]
            # Output 1: boxes shape (1, num_boxes, 4) - [x1, y1, x2, y2] normalized
            scores = outputs[0]  # Shape: (1, num_boxes, 2)
            boxes = outputs[1]   # Shape: (1, num_boxes, 4)

            # Remove batch dimension
            if len(scores.shape) == 3:
                scores = scores[0]  # Shape: (num_boxes, 2)
            if len(boxes.shape) == 3:
                boxes = boxes[0]    # Shape: (num_boxes, 4)

            self.get_logger().debug(f'UltraFace: detected {len(boxes)} candidate regions')

            # Check if any faces detected
            if len(boxes) == 0:
                self.get_logger().debug('No faces detected by UltraFace')
                return None

            # Get best face detection
            # scores shape: (num_boxes, 2) where [:, 0] = background, [:, 1] = face
            face_scores = scores[:, 1]
            best_idx = np.argmax(face_scores)
            best_score = float(face_scores[best_idx])

            self.get_logger().debug(f'Best face: idx={best_idx}, score={best_score:.3f}')

            if best_score < self.face_detection_threshold:
                self.get_logger().debug(f'No face detected (best score: {best_score:.3f} < threshold: {self.face_detection_threshold})')
                return None

            # Extract face box (normalized coordinates)
            # boxes shape: (num_boxes, 4) where each box is [x1, y1, x2, y2]
            face_box_norm = boxes[best_idx]
            face_x1_norm, face_y1_norm, face_x2_norm, face_y2_norm = face_box_norm

            # Convert to ROI coordinates
            face_x1_roi = int(face_x1_norm * (x_max - x_min))
            face_y1_roi = int(face_y1_norm * (y_max - y_min))
            face_x2_roi = int(face_x2_norm * (x_max - x_min))
            face_y2_roi = int(face_y2_norm * (y_max - y_min))

            # Convert to image coordinates
            face_x1 = x_min + face_x1_roi
            face_y1 = y_min + face_y1_roi
            face_x2 = x_min + face_x2_roi
            face_y2 = y_min + face_y2_roi

            # Convert to [x, y, w, h]
            face_bbox = [face_x1, face_y1, face_x2 - face_x1, face_y2 - face_y1]

            # Validate face size (both width AND height must be sufficient)
            face_width = face_bbox[2]
            face_height = face_bbox[3]
            face_size = min(face_width, face_height)  # Use minimum dimension, not maximum

            if face_size < self.min_face_size:
                self.get_logger().info(f'Face too small: {face_width}x{face_height}px (min dimension: {face_size}px < {self.min_face_size}px threshold)')
                return None

            self.get_logger().debug(f'Face detected: bbox={face_bbox} ({face_width}x{face_height}px), score={best_score:.3f}')
            return face_bbox

        except Exception as e:
            self.get_logger().error(f"Face detection failed: {e}")
            return None

    def _process_identification(self, track_id, track):
        """Process face recognition for a specific track."""
        start_time = time.time()

        # Get latest frame from cache (FULL RESOLUTION)
        if len(self.frame_cache) == 0:
            self.get_logger().warning(f'No frames in cache for track_id={track_id}')
            return

        timestamp, image, header = self.frame_cache[-1]

        # Get person bounding box
        person_bbox = [track.bbox_x, track.bbox_y, track.bbox_w, track.bbox_h]

        # Detect face in person ROI
        face_detection_start = time.time()
        face_bbox = self._detect_face_in_roi(image, person_bbox)
        face_detection_time = (time.time() - face_detection_start) * 1000

        if face_bbox is None:
            self.get_logger().warning(
                f'No face detected in track_id={track_id} '
                f'(person_bbox=[{int(person_bbox[0])},{int(person_bbox[1])},{int(person_bbox[2])},{int(person_bbox[3])}])'
            )
            # Remove from queue so we can retry later
            if track_id in self.identification_queue:
                del self.identification_queue[track_id]
            return

        # Extract face crop from FULL RESOLUTION image
        face_x, face_y, face_w, face_h = face_bbox
        face_x_min = max(0, int(face_x))
        face_y_min = max(0, int(face_y))
        face_x_max = min(image.shape[1], int(face_x + face_w))
        face_y_max = min(image.shape[0], int(face_y + face_h))

        face_crop = image[face_y_min:face_y_max, face_x_min:face_x_max]

        # TEMP: Save face crop for debugging
        try:
            debug_path = f'/tmp/face_debug_track{track_id}.jpg'
            cv2.imwrite(debug_path, face_crop)
            self.get_logger().info(f'DEBUG: Saved face crop to {debug_path}')
        except Exception as e:
            self.get_logger().warning(f'Failed to save debug face crop: {e}')

        # Log extraction details
        self.get_logger().info(
            f'Track {track_id}: Person box=[{int(person_bbox[0])},{int(person_bbox[1])},{int(person_bbox[2])},{int(person_bbox[3])}] '
            f'({int(person_bbox[2])}x{int(person_bbox[3])}px), '
            f'Face detected at=[{face_x_min},{face_y_min},{face_x_max},{face_y_max}] '
            f'({face_x_max - face_x_min}x{face_y_max - face_y_min}px), '
            f'detection_time={face_detection_time:.1f}ms'
        )

        # Extract face embedding
        embedding_start = time.time()
        embedding = self._get_face_embedding(face_crop)
        embedding_time = (time.time() - embedding_start) * 1000

        if embedding is None:
            self.get_logger().error(f'Failed to extract embedding for track_id={track_id}')
            if track_id in self.identification_queue:
                del self.identification_queue[track_id]
            return

        self.get_logger().info(f'Embedding extraction took {embedding_time:.1f}ms')

        # Call face recognition service
        recognition_start = time.time()
        request = RecognizeFace.Request()
        request.face_embedding = embedding.tolist()
        request.threshold = self.recognition_threshold

        try:
            # Asynchronous call with callback (non-blocking)
            future = self.face_recognition_client.call_async(request)

            # Add callback to handle response when it arrives
            def handle_recognition_response(future_result):
                try:
                    response = future_result.result()
                    recognition_time = (time.time() - recognition_start) * 1000

                    # Total processing time
                    total_time = (time.time() - start_time) * 1000

                    # Create identity update message
                    identity_msg = IdentityUpdate()
                    identity_msg.header = header
                    identity_msg.track_id = track_id
                    identity_msg.identity = response.name if response.found else ''
                    identity_msg.confidence = response.similarity  # USE ACTUAL SIMILARITY
                    identity_msg.face_embedding = embedding.tolist()
                    identity_msg.processing_time_ms = float(total_time)
                    identity_msg.face_size_pixels = float(max(face_w, face_h))
                    identity_msg.face_quality_ok = True  # Passed face detection
                    identity_msg.found_in_database = response.found
                    identity_msg.person_id = response.person_id
                    identity_msg.message = response.message

                    # Publish identity update
                    self.identity_pub.publish(identity_msg)

                    # Update person state via service
                    self._update_person_state(track_id, identity_msg.identity, identity_msg.confidence)

                    # Log results
                    self.get_logger().info(
                        f'Identification complete for track_id={track_id}: '
                        f'identity={identity_msg.identity}, '
                        f'confidence={response.similarity:.4f}, '
                        f'found={response.found}, '
                        f'total_time={total_time:.1f}ms '
                        f'(face_detection={face_detection_time:.1f}ms, '
                        f'embedding={embedding_time:.1f}ms, '
                        f'recognition={recognition_time:.1f}ms)'
                    )

                    # Update last identification time
                    self.last_identification_time[track_id] = time.time()

                except Exception as e:
                    self.get_logger().error(f'Recognition response handling failed for track_id={track_id}: {e}')

            # Attach callback to future
            future.add_done_callback(handle_recognition_response)

        except Exception as e:
            self.get_logger().error(f'Recognition service call failed for track_id={track_id}: {e}')
            if track_id in self.identification_queue:
                del self.identification_queue[track_id]

    def _get_face_embedding(self, face_image):
        """Extract face embedding using ONNX model."""
        try:
            h, w = self.embedding_input_shape[2], self.embedding_input_shape[3]

            # ArcFace expects RGB input (OpenCV uses BGR)
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(face_rgb, (w, h))

            # ArcFace normalization: scale to [0, 1] then normalize to [-1, 1]
            # Standard preprocessing: (pixel / 255.0 - 0.5) / 0.5 = (pixel - 127.5) / 127.5
            input_data = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)
            input_data = (input_data - 127.5) / 127.5

            # DEBUG: Log input stats
            self.get_logger().info(f'Input stats: min={input_data.min():.4f}, max={input_data.max():.4f}, mean={input_data.mean():.4f}')

            # Run inference
            outputs = self.embedding_session.run(None, {self.embedding_input_name: input_data})

            # Extract embedding (usually first output)
            embedding = outputs[0][0].flatten()

            # DEBUG: Check raw embedding statistics
            self.get_logger().info(f'Raw embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}, std={embedding.std():.4f}, norm={np.linalg.norm(embedding):.4f}')
            self.get_logger().info(f'First 10 values: {embedding[:10]}')

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            # DEBUG: Check normalized embedding
            self.get_logger().info(f'Normalized embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}, std={embedding.std():.4f}, norm={np.linalg.norm(embedding):.4f}')

            return embedding

        except Exception as e:
            self.get_logger().error(f"Embedding extraction failed: {e}")
            return None

    def _update_person_state(self, track_id, identity, confidence):
        """Update person state via service."""
        request = UpdateIdentity.Request()
        request.track_id = track_id
        request.identity = identity
        request.confidence = confidence

        try:
            future = self.update_identity_client.call_async(request)
            # Don't wait for response, fire and forget
        except Exception as e:
            self.get_logger().error(f'Failed to update person state for track_id={track_id}: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionConditional()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Conditional Face Recognition Node')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
