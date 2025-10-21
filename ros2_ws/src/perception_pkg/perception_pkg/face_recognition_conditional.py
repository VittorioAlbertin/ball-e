#!/usr/bin/env python3
"""
Conditional Face Recognition Node for Ball-e Robot

This node performs face recognition ON-DEMAND instead of every frame.
It processes faces only when triggered by:
1. New track appears (is_new_track == true)
2. Explicit identification request
3. Re-identification timer expires

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
    """On-demand face recognition node with performance optimization."""

    def __init__(self):
        super().__init__('face_recognition_conditional')

        # Declare parameters
        self.declare_parameter('embedding_model_path', '/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/facenet.onnx')
        self.declare_parameter('embedding_model_url', 'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx')
        self.declare_parameter('recognition_threshold', 0.6)
        self.declare_parameter('min_face_size', 20)  # Minimum face pixels
        self.declare_parameter('max_face_size', 400)  # Maximum face pixels
        self.declare_parameter('frame_cache_size', 10)  # Cache last N frames
        self.declare_parameter('reidentification_interval', 30.0)  # Re-ID every 30 seconds
        self.declare_parameter('auto_identify_new_tracks', True)  # Auto-identify new tracks

        self.embedding_model_path = self.get_parameter('embedding_model_path').value
        self.embedding_model_url = self.get_parameter('embedding_model_url').value
        self.recognition_threshold = self.get_parameter('recognition_threshold').value
        self.min_face_size = self.get_parameter('min_face_size').value
        self.max_face_size = self.get_parameter('max_face_size').value
        self.frame_cache_size = self.get_parameter('frame_cache_size').value
        self.reidentification_interval = self.get_parameter('reidentification_interval').value
        self.auto_identify_new_tracks = self.get_parameter('auto_identify_new_tracks').value

        # Logging
        self.get_logger().info('Conditional Face Recognition Node initializing...')
        self.get_logger().info(f'  recognition_threshold: {self.recognition_threshold}')
        self.get_logger().info(f'  min_face_size: {self.min_face_size}px')
        self.get_logger().info(f'  max_face_size: {self.max_face_size}px')
        self.get_logger().info(f'  frame_cache_size: {self.frame_cache_size}')
        self.get_logger().info(f'  reidentification_interval: {self.reidentification_interval}s')
        self.get_logger().info(f'  auto_identify_new_tracks: {self.auto_identify_new_tracks}')

        # Download model if needed
        self._ensure_model_exists(self.embedding_model_path, self.embedding_model_url, "face embedding")

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

        # Frame cache for async processing
        self.frame_cache = deque(maxlen=self.frame_cache_size)

        # Track last identification times for re-identification
        self.last_identification_time = {}  # {track_id: timestamp}

        # Pending identification requests
        self.identification_queue = set()

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
        """Cache incoming camera frames."""
        try:
            image = rnp.numpify(msg)
            timestamp = self.get_clock().now()
            self.frame_cache.append((timestamp, image, msg.header))
        except Exception as e:
            self.get_logger().error(f"Failed to cache image: {e}")

    def track_callback(self, msg):
        """Process track updates and trigger identification for new tracks."""
        if not self.auto_identify_new_tracks:
            return

        for track in msg.tracks:
            if track.is_new_track:
                track_id = track.track_id
                self.get_logger().info(f'New track detected: {track_id}, queuing for identification')
                self.identification_queue.add(track_id)
                # Process immediately
                self._process_identification(track_id, track)

    def state_callback(self, msg):
        """Monitor person states for identification requests."""
        for person in msg.persons:
            track_id = person.track_id

            # Check if identification is requested
            if person.requires_identification and track_id not in self.identification_queue:
                self.get_logger().info(f'Identification requested for track_id={track_id}')
                self.identification_queue.add(track_id)

    def check_reidentification(self):
        """Check if any known persons need re-identification."""
        # This will be triggered by the person state manager
        # For now, just check based on last identification time
        current_time = time.time()

        for track_id, last_time in list(self.last_identification_time.items()):
            if current_time - last_time > self.reidentification_interval:
                self.get_logger().info(f'Re-identification interval expired for track_id={track_id}')
                self.identification_queue.add(track_id)

    def _process_identification(self, track_id, track):
        """Process face recognition for a specific track."""
        start_time = time.time()

        # Get latest frame from cache
        if len(self.frame_cache) == 0:
            self.get_logger().warning(f'No frames in cache for track_id={track_id}')
            return

        timestamp, image, header = self.frame_cache[-1]

        # Extract bbox
        bbox_x = int(track.bbox_x)
        bbox_y = int(track.bbox_y)
        bbox_w = int(track.bbox_w)
        bbox_h = int(track.bbox_h)

        # Quality check: face size
        face_size = max(bbox_w, bbox_h)
        quality_ok = self.min_face_size <= face_size <= self.max_face_size

        if not quality_ok:
            self.get_logger().warning(
                f'Face quality check failed for track_id={track_id}: '
                f'size={face_size}px (min={self.min_face_size}, max={self.max_face_size})'
            )
            # Still process but log the warning
            # Optionally return here if quality is too poor

        # Crop face from image
        x_min = max(0, bbox_x)
        y_min = max(0, bbox_y)
        x_max = min(image.shape[1], bbox_x + bbox_w)
        y_max = min(image.shape[0], bbox_y + bbox_h)

        if x_max <= x_min or y_max <= y_min:
            self.get_logger().error(f'Invalid bbox for track_id={track_id}')
            return

        face_crop = image[y_min:y_max, x_min:x_max]

        if face_crop.size == 0:
            self.get_logger().error(f'Empty face crop for track_id={track_id}')
            return

        # Extract face embedding
        embedding_start = time.time()
        embedding = self._get_face_embedding(face_crop)
        embedding_time = (time.time() - embedding_start) * 1000

        if embedding is None:
            self.get_logger().error(f'Failed to extract embedding for track_id={track_id}')
            return

        self.get_logger().info(f'Embedding extraction took {embedding_time:.1f}ms')

        # Call face recognition service
        recognition_start = time.time()
        request = RecognizeFace.Request()
        request.face_embedding = embedding.tolist()
        request.threshold = self.recognition_threshold

        try:
            # Synchronous call with timeout
            future = self.face_recognition_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

            if future.done():
                response = future.result()
                recognition_time = (time.time() - recognition_start) * 1000

                # Total processing time
                total_time = (time.time() - start_time) * 1000

                # Create identity update message
                identity_msg = IdentityUpdate()
                identity_msg.header = header
                identity_msg.track_id = track_id
                identity_msg.identity = response.name if response.found else ''
                identity_msg.confidence = self.recognition_threshold if response.found else 0.0
                identity_msg.processing_time_ms = float(total_time)
                identity_msg.face_size_pixels = float(face_size)
                identity_msg.face_quality_ok = quality_ok
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
                    f'found={response.found}, '
                    f'total_time={total_time:.1f}ms '
                    f'(embedding={embedding_time:.1f}ms, recognition={recognition_time:.1f}ms)'
                )

                # Update last identification time
                self.last_identification_time[track_id] = time.time()

                # Remove from queue
                self.identification_queue.discard(track_id)

            else:
                self.get_logger().error(f'Recognition service timeout for track_id={track_id}')

        except Exception as e:
            self.get_logger().error(f'Recognition service call failed for track_id={track_id}: {e}')

    def _get_face_embedding(self, face_image):
        """Extract face embedding using ONNX model."""
        try:
            h, w = self.embedding_input_shape[2], self.embedding_input_shape[3]
            resized = cv2.resize(face_image, (w, h))

            # Normalize
            input_data = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

            # Run inference
            outputs = self.embedding_session.run(None, {self.embedding_input_name: input_data})

            # Extract embedding (usually first output)
            embedding = outputs[0][0].flatten()

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

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
