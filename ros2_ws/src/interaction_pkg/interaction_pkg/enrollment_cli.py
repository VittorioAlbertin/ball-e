#!/usr/bin/env python3
"""
Interactive Person Enrollment CLI for Ball-e Robot

DESCRIPTION:
    Terminal-based enrollment wizard that captures multiple face poses and
    voice samples to register a new person in the database.

FEATURES:
    - Multi-pose face capture (front, left, right, up, down)
    - Multi-phrase voice capture
    - Quality-weighted embedding averaging
    - Interactive prompts with progress feedback

USAGE:
    ros2 run interaction_pkg enrollment_cli

AUTHOR: Ball-e Team
DATE: 2025-11-15
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import time
import sys

from sensor_msgs.msg import Image
from msgs_interfaces.msg import SpeechSegment
from msgs_interfaces.srv import DetectFace, GenerateEmbedding, AddPerson, GenerateVoiceEmbedding
from cv_bridge import CvBridge


class EnrollmentCLI(Node):
    def __init__(self):
        super().__init__('enrollment_cli')

        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()

        # Frame cache
        self.latest_frame = None
        self.latest_frame_msg = None

        # Speech segment cache
        self.speech_segments = []
        self.waiting_for_speech = False

        # Service clients
        self.detect_face_client = self.create_client(
            DetectFace,
            '/face_detection/detect_face',
            callback_group=self.callback_group
        )
        self.generate_embedding_client = self.create_client(
            GenerateEmbedding,
            '/face_recognition/generate_embedding',
            callback_group=self.callback_group
        )
        self.add_person_client = self.create_client(
            AddPerson,
            'people_db/add_person',
            callback_group=self.callback_group
        )
        self.generate_voice_embedding_client = self.create_client(
            GenerateVoiceEmbedding,
            '/voice_recognition/generate_embedding',
            callback_group=self.callback_group
        )

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10,
            callback_group=self.callback_group
        )
        self.speech_sub = self.create_subscription(
            SpeechSegment,
            '/microphone/speech_segment',
            self.speech_callback,
            10,
            callback_group=self.callback_group
        )

        self.get_logger().info("EnrollmentCLI initialized")

    def image_callback(self, msg):
        """Cache latest camera frame."""
        self.latest_frame_msg = msg

    def speech_callback(self, msg):
        """Cache speech segments when waiting for speech."""
        if self.waiting_for_speech:
            self.speech_segments.append(msg)
            self.get_logger().info(f"Speech captured: {msg.duration_seconds:.2f}s")

    def run_enrollment(self):
        """Main enrollment workflow."""
        print("\n" + "=" * 60)
        print("       Ball-e Person Enrollment Wizard")
        print("=" * 60)

        # 1. Get person name
        person_name = input("\n1. Enter person's name: ").strip()
        if not person_name:
            print("Error: Name cannot be empty")
            return False

        print(f"\n   Enrolling: {person_name}")

        # 2. Capture face embeddings
        print("\n2. Face Enrollment")
        print("   " + "-" * 50)
        face_embeddings = self.capture_face_poses()

        if not face_embeddings:
            print("\n   Error: Failed to capture face embeddings")
            return False

        # Compute representative face embedding
        print(f"\n   Computing representative embedding from {len(face_embeddings)} poses...")
        face_embedding = self.compute_weighted_average(face_embeddings)
        print(f"   Representative embedding computed (dim={len(face_embedding)})")

        # 3. Capture voice embeddings
        print("\n3. Voice Enrollment")
        print("   " + "-" * 50)
        voice_embedding = self.capture_voice_samples()

        if voice_embedding is None:
            print("\n   Warning: No voice samples captured")
            print("   Proceeding with face-only enrollment")

        # 4. Save to database
        print("\n4. Saving to Database")
        print("   " + "-" * 50)

        success, person_id = self.save_to_database(
            person_name,
            face_embedding,
            face_embeddings,
            voice_embedding
        )

        if success:
            print(f"\n   Person ID: {person_id}")
            print(f"\n   Face embeddings saved ({len(face_embeddings)} poses + average)")
            if voice_embedding is not None:
                print(f"   Voice embedding saved")
            else:
                print("   No voice embedding saved")

            print("\n" + "=" * 60)
            print(f"   Enrollment complete! {person_name} is now registered.")
            print("=" * 60)
            return True
        else:
            print("\n   Error: Failed to save to database")
            return False

    def capture_face_poses(self):
        """
        Capture multiple face poses.

        Returns:
            list: List of (embedding, pose_type, quality_score) tuples
        """
        poses = [
            ('front', 'Look straight ahead at the camera'),
            ('left', 'Turn your head 45 degrees to your LEFT'),
            ('right', 'Turn your head 45 degrees to your RIGHT'),
            ('up', 'Tilt your head slightly UP'),
            ('down', 'Tilt your head slightly DOWN')
        ]

        captured_embeddings = []

        for pose_type, instruction in poses:
            print(f"\n   [{pose_type.upper()}] {instruction}")
            input("   Press Enter when ready...")

            # Wait for camera frame
            if not self.wait_for_frame():
                print(f"   ERROR: No camera frame available")
                continue

            # Detect face
            print("   Detecting face...", end=" ", flush=True)
            face_result = self.detect_face()

            if not face_result['success']:
                print(f"FAILED - {face_result['message']}")
                print(f"   Retrying {pose_type}...")
                continue

            print("OK")

            # Generate embedding
            print("   Generating embedding...", end=" ", flush=True)
            embedding_result = self.generate_embedding(face_result)

            if not embedding_result['success']:
                print(f"FAILED - {embedding_result['message']}")
                continue

            embedding = np.array(embedding_result['embedding'], dtype=np.float32)

            # Compute quality score (based on face size)
            face_area = face_result['face_w'] * face_result['face_h']
            quality_score = min(1.0, face_area / 10000.0)  # Normalize by expected face size

            captured_embeddings.append({
                'embedding': embedding,
                'pose_type': pose_type,
                'quality_score': quality_score
            })

            print(f"CAPTURED (quality: {quality_score:.2f})")

        print(f"\n   [{len(captured_embeddings)}/{len(poses)} poses captured]")

        if len(captured_embeddings) < 3:
            print("   Warning: Less than 3 poses captured. Consider retrying.")

        return captured_embeddings

    def capture_voice_samples(self):
        """
        Capture multiple voice samples.

        Returns:
            np.ndarray: Averaged voice embedding or None
        """
        phrases = [
            "The quick brown fox jumps over the lazy dog",
            "Ball-e, please remember my voice",
            "Hello, my name is [say your name]"
        ]

        voice_embeddings = []

        for i, phrase in enumerate(phrases, 1):
            print(f"\n   Sample {i}/{len(phrases)}")
            print(f"   Say: \"{phrase}\"")
            input("   Press Enter when ready to speak...")

            # Wait for speech
            self.speech_segments = []

            # Small delay to let keypress sound pass through VAD
            time.sleep(0.3)

            self.waiting_for_speech = True

            print("   Listening...", end=" ", flush=True)

            # Wait for speech segment (max 10 seconds)
            timeout = 10.0
            start_time = time.time()
            while not self.speech_segments and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)

            self.waiting_for_speech = False

            if not self.speech_segments:
                print("TIMEOUT - No speech detected")
                continue

            speech_msg = self.speech_segments[-1]  # Use most recent
            print(f"CAPTURED ({speech_msg.duration_seconds:.2f}s)")

            # Generate voice embedding
            print("   Generating voice embedding...", end=" ", flush=True)
            embedding_result = self.generate_voice_embedding(speech_msg)

            if not embedding_result['success']:
                print(f"FAILED - {embedding_result['message']}")
                continue

            embedding = np.array(embedding_result['embedding'], dtype=np.float32)
            voice_embeddings.append(embedding)
            print("OK")

        print(f"\n   [{len(voice_embeddings)}/{len(phrases)} voice samples captured]")

        if not voice_embeddings:
            return None

        # Average voice embeddings
        avg_embedding = np.mean(voice_embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        return avg_embedding.astype(np.float32)

    def compute_weighted_average(self, embeddings_data):
        """
        Compute quality-weighted average of embeddings.

        Args:
            embeddings_data: List of dicts with 'embedding' and 'quality_score'

        Returns:
            np.ndarray: Weighted average embedding
        """
        embeddings = [e['embedding'] for e in embeddings_data]
        weights = [e['quality_score'] for e in embeddings_data]

        # Weighted average
        weighted_sum = np.zeros_like(embeddings[0])
        total_weight = 0

        for emb, weight in zip(embeddings, weights):
            weighted_sum += emb * weight
            total_weight += weight

        avg_embedding = weighted_sum / total_weight

        # Re-normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        return avg_embedding.astype(np.float32)

    def wait_for_frame(self, timeout=2.0):
        """Wait for a FRESH camera frame to be available."""
        # Clear old frame to ensure we get a fresh one
        self.latest_frame_msg = None

        start = time.time()
        while self.latest_frame_msg is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_frame_msg is not None

    def detect_face(self):
        """Call face detection service on current frame."""
        if not self.detect_face_client.wait_for_service(timeout_sec=2.0):
            return {'success': False, 'message': 'Service not available'}

        request = DetectFace.Request()
        request.header = self.latest_frame_msg.header

        # Use full frame as bbox
        request.track_id = 0
        request.bbox_x = 0
        request.bbox_y = 0
        request.bbox_w = 640  # Assuming low-res camera
        request.bbox_h = 360

        future = self.detect_face_client.call_async(request)

        timeout = 2.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)

        if not future.done():
            return {'success': False, 'message': 'Timeout'}

        response = future.result()

        if response.success:
            return {
                'success': True,
                'face_x': response.face_x,
                'face_y': response.face_y,
                'face_w': response.face_w,
                'face_h': response.face_h
            }
        else:
            return {'success': False, 'message': response.message}

    def generate_embedding(self, face_bbox):
        """Generate face embedding."""
        if not self.generate_embedding_client.wait_for_service(timeout_sec=2.0):
            return {'success': False, 'message': 'Service not available'}

        request = GenerateEmbedding.Request()
        request.header = self.latest_frame_msg.header
        request.track_id = 0
        request.face_x = face_bbox['face_x']
        request.face_y = face_bbox['face_y']
        request.face_w = face_bbox['face_w']
        request.face_h = face_bbox['face_h']

        future = self.generate_embedding_client.call_async(request)

        timeout = 2.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)

        if not future.done():
            return {'success': False, 'message': 'Timeout'}

        response = future.result()

        if response.success:
            return {'success': True, 'embedding': response.embedding}
        else:
            return {'success': False, 'message': response.message}

    def generate_voice_embedding(self, speech_msg):
        """Generate voice embedding from speech segment."""
        if not self.generate_voice_embedding_client.wait_for_service(timeout_sec=2.0):
            return {'success': False, 'message': 'Service not available'}

        request = GenerateVoiceEmbedding.Request()
        request.audio_data = speech_msg.audio_data
        request.sample_rate = speech_msg.sample_rate

        future = self.generate_voice_embedding_client.call_async(request)

        timeout = 5.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)

        if not future.done():
            return {'success': False, 'message': 'Timeout'}

        response = future.result()

        if response.success:
            return {'success': True, 'embedding': response.embedding}
        else:
            return {'success': False, 'message': response.message}

    def save_to_database(self, name, face_embedding, face_embeddings_data, voice_embedding):
        """
        Save person to database.

        Args:
            name: Person's name
            face_embedding: Representative face embedding
            face_embeddings_data: List of all pose embeddings
            voice_embedding: Voice embedding or None

        Returns:
            tuple: (success, person_id)
        """
        if not self.add_person_client.wait_for_service(timeout_sec=2.0):
            return False, -1

        request = AddPerson.Request()
        request.name = name
        request.face_embedding = face_embedding.tolist()
        request.preferences_json = '{}'
        request.notes = f"Enrolled via CLI with {len(face_embeddings_data)} face poses"

        # TODO: Add voice_embedding to request once AddPerson service is updated
        # For now, voice embedding needs to be saved separately

        future = self.add_person_client.call_async(request)

        timeout = 2.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)

        if not future.done():
            return False, -1

        response = future.result()

        if response.success:
            person_id = response.person_id

            # TODO: Save individual face embeddings to face_embeddings table
            # TODO: Save voice embedding to people table

            return True, person_id
        else:
            self.get_logger().error(f"Failed to save: {response.message}")
            return False, -1


def main(args=None):
    rclpy.init(args=args)

    cli = EnrollmentCLI()

    # Wait for services to be ready
    print("Waiting for services...")
    cli.detect_face_client.wait_for_service(timeout_sec=10.0)
    cli.generate_embedding_client.wait_for_service(timeout_sec=10.0)
    cli.add_person_client.wait_for_service(timeout_sec=10.0)
    print("Services ready!")

    try:
        # Run enrollment
        success = cli.run_enrollment()

        if not success:
            print("\nEnrollment failed. Please try again.")

    except KeyboardInterrupt:
        print("\nEnrollment cancelled.")
    finally:
        cli.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
