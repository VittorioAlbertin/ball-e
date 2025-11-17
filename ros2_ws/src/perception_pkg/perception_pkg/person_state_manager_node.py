"""
Person State Manager Node (Service-Based Orchestrator)

DESCRIPTION:
    The PersonStateManager is the central orchestrator and single source of truth
    for all tracked persons in the system. It maintains the complete state of each
    tracked person including their identity, tracking confidence, and identification
    status. This node decides when identification is required and orchestrates
    the identification pipeline by making synchronous service calls.

RESPONSIBILITIES:
    1. Maintain world state of all tracked persons
    2. Decide when identification/re-identification is required
    3. Orchestrate identification pipeline via service calls
    4. Manage identification timers and retry logic
    5. Publish complete PersonStateArray for downstream consumers
    6. Update database with interaction timestamps
    7. Provide enrollment service for new person registration

IDENTIFICATION LOGIC:
    A person requires identification when any of these conditions are met:
    - NEW TRACK: First time track appears (is_new_track flag)
    - LOW CONFIDENCE: Tracking confidence drops below threshold
    - TIME ELAPSED: Periodic re-verification based on identity status
        * Known persons: Re-identify every known_person_reidentify_interval
        * Unknown persons: Re-identify every unknown_person_reidentify_interval

SUBSCRIPTIONS:
    - /person_tracker/tracks (PersonTrackArray): Track updates from ByteTrack

PUBLICATIONS:
    - /person_state/array (PersonStateArray): Complete world state with identities

SERVICE CLIENTS:
    - face_detection/detect_face: Detect face in person ROI
    - face_recognition/generate_embedding: Generate face embedding
    - people_db/recognize_face: Recognize person from embedding
    - people_db/add_person: Add new person to database
    - people_db/update_last_seen: Update interaction timestamp

SERVICE SERVERS:
    - person_state/enroll_person: Enroll new person from track_id

PARAMETERS:
    - reidentification_confidence_threshold (float): Tracking confidence below which
      re-identification is triggered [default: 0.4]
    - known_person_reidentify_interval (float): Seconds between re-identification
      for known persons [default: 60.0]
    - unknown_person_reidentify_interval (float): Seconds between re-identification
      for unknown persons [default: 15.0]
    - max_identification_attempts (int): Maximum attempts before marking as "unknown"
      [default: 5]

DATA FLOW:
    1. Receive PersonTrack from ByteTrack
    2. Check if identification required (new track, low confidence, time elapsed)
    3. If yes: Synchronously call detection → embedding → recognition services
    4. Update internal state with identity
    5. Call update_last_seen service for identified persons
    6. Publish PersonStateArray with updated identities

AUTHOR: Vittorio Albertin
DATE: 2025-10-29
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from msgs_interfaces.msg import PersonTrackArray, PersonStateArray, PersonState, PersonTrack, SpeechSegment
from msgs_interfaces.srv import DetectFace, GenerateEmbedding, RecognizeFace, AddPerson, UpdateLastSeen, EnrollPerson, GenerateVoiceEmbedding, RecognizeVoice, GetPerson
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
import time
import numpy as np

from perception_pkg.temporal_bayesian_tracker import TemporalBayesianIdentity, MultiTrackIdentityManager


class PersonStateManager(Node):
    def __init__(self):
        super().__init__('person_state_manager')

        # Declare parameters
        self.declare_parameter('reidentification_confidence_threshold', 0.4)
        self.declare_parameter('known_person_reidentify_interval', 5.0)
        self.declare_parameter('unknown_person_reidentify_interval', 5.0)
        self.declare_parameter('identity_confidence_threshold', 0.65)
        self.declare_parameter('enable_voice_recognition', True)

        # Get parameters
        self.reidentification_threshold = self.get_parameter('reidentification_confidence_threshold').value
        self.known_interval = self.get_parameter('known_person_reidentify_interval').value
        self.unknown_interval = self.get_parameter('unknown_person_reidentify_interval').value
        self.identity_confidence_threshold = self.get_parameter('identity_confidence_threshold').value
        self.enable_voice_recognition = self.get_parameter('enable_voice_recognition').value

        # Internal state: {track_id: PersonStateData}
        self.person_states = {}

        # Bayesian identity trackers: {track_id: TemporalBayesianIdentity}
        self.identity_trackers = {}
        self.known_person_ids = []  # Will be populated from database

        # Cache for person names (to avoid repeated DB lookups)
        self.person_name_cache = {}  # {person_id: person_name}

        # Create callback groups to allow concurrent execution
        self.reentrant_group = ReentrantCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()

        # Subscriptions (use reentrant group to allow concurrent processing)
        self.track_sub = self.create_subscription(
            PersonTrackArray,
            '/person_tracker/tracks',
            self.track_callback,
            10,
            callback_group=self.reentrant_group
        )

        # Publications
        self.state_pub = self.create_publisher(PersonStateArray, '/person_state/array', 10)

        # Frame cache pin/release publishers
        self.pin_frame_pub = self.create_publisher(Header, '/frame_cache/pin_timestamp', 10)
        self.release_frame_pub = self.create_publisher(Header, '/frame_cache/release_timestamp', 10)

        # Service clients (use reentrant group)
        self.detect_face_client = self.create_client(
            DetectFace,
            '/face_detection/detect_face',
            callback_group=self.reentrant_group
        )
        self.generate_embedding_client = self.create_client(
            GenerateEmbedding,
            '/face_recognition/generate_embedding',
            callback_group=self.reentrant_group
        )
        self.recognize_face_client = self.create_client(
            RecognizeFace,
            'people_db/recognize_face',
            callback_group=self.reentrant_group
        )
        self.add_person_client = self.create_client(
            AddPerson,
            'people_db/add_person',
            callback_group=self.reentrant_group
        )
        self.update_last_seen_client = self.create_client(
            UpdateLastSeen,
            'people_db/update_last_seen',
            callback_group=self.reentrant_group
        )
        self.get_person_client = self.create_client(
            GetPerson,
            'people_db/get_person',
            callback_group=self.reentrant_group
        )

        # Voice recognition service clients
        if self.enable_voice_recognition:
            self.generate_voice_embedding_client = self.create_client(
                GenerateVoiceEmbedding,
                '/voice_recognition/generate_embedding',
                callback_group=self.reentrant_group
            )
            self.recognize_voice_client = self.create_client(
                RecognizeVoice,
                'people_db/recognize_voice',
                callback_group=self.reentrant_group
            )

            # Subscribe to speech segments for voice recognition
            self.speech_sub = self.create_subscription(
                SpeechSegment,
                '/microphone/speech_segment',
                self.speech_segment_callback,
                10,
                callback_group=self.reentrant_group
            )
            self.get_logger().info("Voice recognition enabled")

        # Service server (use service group)
        self.enroll_service = self.create_service(
            EnrollPerson,
            '/person_state/enroll_person',
            self.enroll_person_callback,
            callback_group=self.service_group
        )

        # Publishing timer for state array (use reentrant group)
        self.create_timer(0.1, self.publish_state_array, callback_group=self.reentrant_group)  # 10 Hz

        # Timer for Bayesian tracker decay (apply time-based confidence decay)
        self.create_timer(1.0, self.apply_tracker_decay, callback_group=self.reentrant_group)  # 1 Hz

        self.get_logger().info("PersonStateManager initialized (Service-Based with MultiThreadedExecutor + Bayesian Fusion)")
        self.get_logger().info(f"  Reidentification threshold: {self.reidentification_threshold}")
        self.get_logger().info(f"  Known person interval: {self.known_interval}s")
        self.get_logger().info(f"  Unknown person interval: {self.unknown_interval}s")
        self.get_logger().info(f"  Identity confidence threshold: {self.identity_confidence_threshold}")
        self.get_logger().info(f"  Voice recognition: {self.enable_voice_recognition}")

    def track_callback(self, msg: PersonTrackArray):
        """
        Process track updates from ByteTrack and perform identification if needed.

        Args:
            msg: PersonTrackArray containing current tracks
        """
        current_track_ids = set()

        for track in msg.tracks:
            current_track_ids.add(track.track_id)

            # Initialize state if new track
            if track.track_id not in self.person_states:
                self.person_states[track.track_id] = {
                    'identity': -1,  # person_id (int) or -1 for unknown
                    'identity_confidence': 0.0,
                    'person_id': -1,  # Redundant with identity, kept for compatibility
                    'last_identification_time': None,
                    'last_identification_request_time': None,  # When request was made (not result)
                    'identification_pending': False,
                    'first_seen': track.header.stamp,
                    'last_seen': track.header.stamp,
                    'bbox': (track.bbox_x, track.bbox_y, track.bbox_w, track.bbox_h),
                    'tracking_confidence': track.tracking_confidence,
                    'frames_since_last_seen': track.frames_since_last_seen,
                    'requires_identification': False,  # External flag for manual enrollment
                    # Voice-specific fields (kept for compatibility, but identity comes from Bayesian tracker)
                    'voice_identity': -1,  # person_id from voice (int) or -1
                    'voice_confidence': 0.0,
                    'voice_person_id': -1,
                    'last_voice_identification_time': None,
                    'is_speaking': False
                }
                # Initialize Bayesian tracker for this track
                self.identity_trackers[track.track_id] = TemporalBayesianIdentity(
                    self.known_person_ids
                )
                self.get_logger().info(f"New track initialized: {track.track_id} (with Bayesian tracker)")

            # Update state from track
            state = self.person_states[track.track_id]
            state['last_seen'] = track.header.stamp
            state['bbox'] = (track.bbox_x, track.bbox_y, track.bbox_w, track.bbox_h)
            state['tracking_confidence'] = track.tracking_confidence
            state['frames_since_last_seen'] = track.frames_since_last_seen

            # Check if identification needed
            needs_id_internal = self.requires_identification(track, state)
            needs_id_external = state['requires_identification']

            # Combine: set requires_identification if internal logic says yes
            if needs_id_internal and not needs_id_external:
                state['requires_identification'] = True

            # Process identification if needed and not pending
            if state['requires_identification'] and not state['identification_pending']:
                state['identification_pending'] = True
                state['last_identification_request_time'] = self.get_clock().now().to_msg()

                # Call identification pipeline synchronously
                identity_result = self.perform_identification(track, state)

                if identity_result['success']:
                    state['identity'] = identity_result['identity']
                    state['identity_confidence'] = identity_result['confidence']
                    state['person_id'] = identity_result['person_id']
                    state['last_identification_time'] = track.header.stamp

                    # Update database last_seen for identified persons
                    if identity_result['person_id'] != -1:
                        self.update_database_last_seen(identity_result['person_id'], track.header.stamp)

                state['requires_identification'] = False
                state['identification_pending'] = False

        # Remove tracks that are no longer present (cleanup)
        tracks_to_remove = [tid for tid in self.person_states.keys() if tid not in current_track_ids]
        for track_id in tracks_to_remove:
            self.get_logger().info(f"Removing lost track: {track_id}")
            del self.person_states[track_id]
            if track_id in self.identity_trackers:
                del self.identity_trackers[track_id]

    def requires_identification(self, track, state) -> bool:
        """
        Determine if a track requires (re-)identification.

        Uses Bayesian identity confidence (not tracking confidence) to decide.
        Re-identification is triggered by confidence decay, not periodic timers.
        Uses request time (not result time) for cooldown to prevent spam.

        Args:
            track: PersonTrack message
            state: Internal state dictionary for this track

        Returns:
            bool: True if identification is required
        """
        # Condition 1: NEW TRACK (first time this track appears)
        if track.is_new_track and state['last_identification_request_time'] is None:
            self.get_logger().info(f"Track {track.track_id}: NEW TRACK - requires identification")
            return True

        # Calculate time since last identification REQUEST (not result)
        # Used for cooldown to prevent spam, not for periodic triggering
        time_since_request = None
        if state['last_identification_request_time'] is not None:
            now = self.get_clock().now()
            last_request_time = rclpy.time.Time.from_msg(state['last_identification_request_time'])
            time_since_request = (now - last_request_time).nanoseconds / 1e9  # Convert to seconds

        # Determine cooldown period based on current identity
        if state['identity'] != -1:
            cooldown = self.known_interval  # 15 seconds for known persons
        else:
            cooldown = self.unknown_interval  # 5 seconds for unknown persons

        # Condition 2: LOW IDENTITY CONFIDENCE (from Bayesian tracker)
        # Check this BEFORE cooldown to allow emergency re-identification
        emergency_reidentification = False
        critical_confidence_threshold = 0.4  # Below this, ignore cooldown

        if track.track_id in self.identity_trackers:
            identity, identity_conf = self.identity_trackers[track.track_id].get_identity_with_decay()
            if identity_conf < critical_confidence_threshold:
                # Critically low confidence - allow re-identification even during cooldown
                self.get_logger().info(
                    f"Track {track.track_id}: CRITICAL LOW CONFIDENCE ({identity_conf:.2f}) - emergency re-identification"
                )
                emergency_reidentification = True
            elif identity_conf < self.identity_confidence_threshold:
                self.get_logger().info(
                    f"Track {track.track_id}: LOW IDENTITY CONFIDENCE ({identity_conf:.2f}) - requires re-identification"
                )
                # Will check cooldown below

        # If we recently REQUESTED identification, respect cooldown period
        # UNLESS confidence is critically low (emergency re-identification)
        if time_since_request is not None and time_since_request < cooldown and not emergency_reidentification:
            return False

        # Return True if confidence is below threshold (either critical or normal)
        if track.track_id in self.identity_trackers:
            identity, identity_conf = self.identity_trackers[track.track_id].get_identity_with_decay()
            if identity_conf < self.identity_confidence_threshold:
                return True

        return False

    def perform_identification(self, track, state) -> dict:
        """
        Orchestrate identification pipeline by calling services synchronously.

        Args:
            track: PersonTrack message
            state: Internal state dictionary

        Returns:
            dict: {'success': bool, 'identity': str, 'confidence': float, 'person_id': int}
        """
        # Pin frame in cache before starting identification pipeline
        pin_msg = Header()
        pin_msg.stamp = track.header.stamp
        pin_msg.frame_id = 'identification_request'
        self.pin_frame_pub.publish(pin_msg)
        self.get_logger().debug(f"Track {track.track_id}: Pinned frame for identification")

        try:
            # 1. Detect face
            face_result = self.call_detect_face(track)
            if not face_result['success']:
                self.get_logger().info(f"Track {track.track_id}: Face detection failed - {face_result['message']}")
                return {'success': False, 'identity': 'unknown', 'confidence': 0.0, 'person_id': -1}

            # 2. Generate embedding
            embedding_result = self.call_generate_embedding(track.header, face_result, track.track_id)
            if not embedding_result['success']:
                self.get_logger().info(f"Track {track.track_id}: Embedding generation failed - {embedding_result['message']}")
                return {'success': False, 'identity': 'unknown', 'confidence': 0.0, 'person_id': -1}

            # 3. Recognize face
            identity_result = self.call_recognize_face(embedding_result['embedding'])

            # 4. Update Bayesian tracker with face scores
            if track.track_id in self.identity_trackers and identity_result.get('all_scores'):
                tracker = self.identity_trackers[track.track_id]

                # Ensure all persons from scores are known to the tracker
                for person_id in identity_result['all_scores'].keys():
                    if person_id not in tracker.known_person_ids:
                        tracker.add_person(person_id)
                        self.get_logger().info(f"Track {track.track_id}: Added person_id {person_id} to Bayesian tracker")

                tracker.update_face(identity_result['all_scores'])

                # Get fused identity from Bayesian tracker - THIS IS THE SOURCE OF TRUTH
                bayesian_identity, bayesian_conf = tracker.get_identity()
                self.get_logger().info(
                    f"Track {track.track_id}: Bayesian identity after face update: {bayesian_identity} ({bayesian_conf:.3f})"
                )

                # Return Bayesian tracker result, NOT raw face recognition result
                # bayesian_identity is either person_id (int) or 'unknown' (str)
                if bayesian_identity != 'unknown':
                    person_id = bayesian_identity
                    self.get_logger().info(
                        f"Track {track.track_id} identified as person_id {person_id} " +
                        f"(Bayesian confidence: {bayesian_conf:.3f})"
                    )
                    return {
                        'success': True,
                        'identity': person_id,  # Store person_id (int), not name
                        'confidence': bayesian_conf,  # Use Bayesian confidence
                        'person_id': person_id
                    }
                else:
                    self.get_logger().info(f"Track {track.track_id}: Bayesian identity is unknown")
                    return {'success': True, 'identity': -1, 'confidence': bayesian_conf, 'person_id': -1}
            else:
                self.get_logger().info(f"Track {track.track_id}: No Bayesian tracker or no scores available")
                return {'success': True, 'identity': -1, 'confidence': 0.0, 'person_id': -1}
        finally:
            # Release pinned frame after identification completes (success or failure)
            self.release_frame_pub.publish(pin_msg)
            self.get_logger().debug(f"Track {track.track_id}: Released pinned frame")

    def call_detect_face(self, track) -> dict:
        """Call face detection service."""
        if not self.detect_face_client.wait_for_service(timeout_sec=1.0):
            return {'success': False, 'message': 'detect_face service not available'}

        request = DetectFace.Request()
        request.header = track.header
        request.track_id = int(track.track_id)
        request.bbox_x = int(track.bbox_x)
        request.bbox_y = int(track.bbox_y)
        request.bbox_w = int(track.bbox_w)
        request.bbox_h = int(track.bbox_h)

        try:
            future = self.detect_face_client.call_async(request)

            # Wait for future with timeout
            timeout = 2.0
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    return {'success': False, 'message': 'Service call timeout'}
                time.sleep(0.01)  # Small sleep to avoid busy waiting

            response = future.result()

            if response.success:
                return {
                    'success': True,
                    'face_x': response.face_x,
                    'face_y': response.face_y,
                    'face_w': response.face_w,
                    'face_h': response.face_h,
                    'message': response.message
                }
            else:
                return {'success': False, 'message': response.message}

        except Exception as e:
            return {'success': False, 'message': f'Service call failed: {str(e)}'}

    def call_generate_embedding(self, header, face_bbox, track_id) -> dict:
        """Call embedding generation service."""
        if not self.generate_embedding_client.wait_for_service(timeout_sec=1.0):
            return {'success': False, 'message': 'generate_embedding service not available'}

        request = GenerateEmbedding.Request()
        request.header = header
        request.track_id = int(track_id)
        request.face_x = int(face_bbox['face_x'])
        request.face_y = int(face_bbox['face_y'])
        request.face_w = int(face_bbox['face_w'])
        request.face_h = int(face_bbox['face_h'])

        try:
            future = self.generate_embedding_client.call_async(request)

            # Wait for future with timeout
            timeout = 2.0
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    return {'success': False, 'message': 'Service call timeout'}
                time.sleep(0.01)

            response = future.result()

            if response.success:
                return {
                    'success': True,
                    'embedding': response.embedding,
                    'message': response.message
                }
            else:
                return {'success': False, 'message': response.message}

        except Exception as e:
            return {'success': False, 'message': f'Service call failed: {str(e)}'}

    def call_recognize_face(self, embedding) -> dict:
        """Call face recognition service."""
        if not self.recognize_face_client.wait_for_service(timeout_sec=1.0):
            return {'match_found': False, 'person_name': '', 'person_id': -1, 'similarity_score': 0.0, 'all_scores': {}}

        request = RecognizeFace.Request()
        request.face_embedding = embedding
        request.confidence_threshold = 0.6

        try:
            future = self.recognize_face_client.call_async(request)

            # Wait for future with timeout
            timeout = 2.0
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    return {'match_found': False, 'person_name': '', 'person_id': -1, 'similarity_score': 0.0, 'all_scores': {}}
                time.sleep(0.01)

            response = future.result()

            # Convert all_scores to dict for Bayesian fusion
            all_scores = {}
            for i, person_id in enumerate(response.all_person_ids):
                all_scores[person_id] = float(response.all_scores[i])

            return {
                'match_found': response.match_found,
                'person_name': response.person_name,
                'person_id': response.person_id,
                'similarity_score': response.similarity_score,
                'all_scores': all_scores  # For Bayesian tracker
            }

        except Exception as e:
            self.get_logger().error(f"recognize_face service call failed: {e}")
            return {'match_found': False, 'person_name': '', 'person_id': -1, 'similarity_score': 0.0, 'all_scores': {}}

    def call_add_person(self, name, embedding, notes) -> dict:
        """Call add person service."""
        if not self.add_person_client.wait_for_service(timeout_sec=1.0):
            return {'success': False, 'person_id': -1, 'message': 'add_person service not available'}

        request = AddPerson.Request()
        request.name = name
        request.face_embedding = embedding
        request.preferences_json = '{}'
        request.notes = notes

        try:
            future = self.add_person_client.call_async(request)

            # Wait for future with timeout
            timeout = 2.0
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    return {'success': False, 'person_id': -1, 'message': 'Service call timeout'}
                time.sleep(0.01)

            response = future.result()

            return {
                'success': response.success,
                'person_id': response.person_id,
                'message': response.message
            }

        except Exception as e:
            return {'success': False, 'person_id': -1, 'message': f'Service call failed: {str(e)}'}

    def enroll_person_callback(self, request, response):
        """
        Service callback for enrolling a new person from track_id.

        Args:
            request: EnrollPerson.Request
            response: EnrollPerson.Response

        Returns:
            EnrollPerson.Response
        """
        track_id = request.track_id

        # Get current state
        if track_id not in self.person_states:
            response.success = False
            response.message = f"Track {track_id} not found"
            return response

        state = self.person_states[track_id]

        # Check if track is currently being tracked (recent update)
        now = self.get_clock().now()
        last_seen_time = rclpy.time.Time.from_msg(state['last_seen'])
        time_since_update = (now - last_seen_time).nanoseconds / 1e9

        if time_since_update > 2.0:
            response.success = False
            response.message = (
                f"Track {track_id} hasn't been updated in {time_since_update:.1f} seconds. "
                f"Make sure the person is visible and being tracked, then try again."
            )
            self.get_logger().warning(f"[ENROLL] {response.message}")
            return response

        # Force fresh identification
        state['requires_identification'] = True
        state['identification_pending'] = True

        # Create pseudo-track from current state using the MOST RECENT timestamp
        # This should still be in the frame caches (1 second cache duration)
        track = PersonTrack()
        track.header.stamp = state['last_seen']  # Use most recent track timestamp
        track.header.frame_id = 'camera_low_res'
        track.track_id = track_id
        track.bbox_x, track.bbox_y, track.bbox_w, track.bbox_h = state['bbox']
        track.tracking_confidence = state['tracking_confidence']
        track.frames_since_last_seen = state['frames_since_last_seen']
        track.is_new_track = False

        self.get_logger().info(
            f"[ENROLL] Track {track_id}: Enrolling with timestamp from {time_since_update:.2f}s ago"
        )

        # 1. Detect face
        face_result = self.call_detect_face(track)
        if not face_result['success']:
            response.success = False
            response.message = f"Face detection failed: {face_result['message']}"
            state['identification_pending'] = False
            state['requires_identification'] = False
            return response

        # 2. Generate embedding
        embedding_result = self.call_generate_embedding(track.header, face_result, track_id)
        if not embedding_result['success']:
            response.success = False
            response.message = f"Embedding generation failed: {embedding_result['message']}"
            state['identification_pending'] = False
            state['requires_identification'] = False
            return response

        # 3. Add to database
        add_result = self.call_add_person(
            request.person_name,
            embedding_result['embedding'],
            request.notes
        )

        if not add_result['success']:
            response.success = False
            response.message = add_result['message']
            state['identification_pending'] = False
            state['requires_identification'] = False
            return response

        # 4. Update state directly with person_id (not name)
        new_person_id = add_result['person_id']
        state['identity'] = new_person_id
        state['person_id'] = new_person_id
        state['identity_confidence'] = 1.0
        state['last_identification_time'] = self.get_clock().now().to_msg()
        state['identification_pending'] = False
        state['requires_identification'] = False

        # 5. Cache the person name
        self.person_name_cache[new_person_id] = request.person_name

        # 6. Add to known persons and update Bayesian tracker
        if new_person_id not in self.known_person_ids:
            self.known_person_ids.append(new_person_id)
        if track_id in self.identity_trackers:
            tracker = self.identity_trackers[track_id]
            if new_person_id not in tracker.known_person_ids:
                tracker.add_person(new_person_id)
            # Set strong belief for newly enrolled person
            # This ensures the tracker immediately recognizes the enrollment
            tracker.belief = {pid: 0.01 for pid in tracker.known_person_ids}
            tracker.belief['unknown'] = 0.01
            tracker.belief[new_person_id] = 0.9
            tracker._normalize_belief(tracker.belief)

        response.success = True
        response.person_id = new_person_id
        response.message = f"Successfully enrolled '{request.person_name}'"

        self.get_logger().info(f"Enrolled track {track_id} as '{request.person_name}' (person_id: {new_person_id})")

        return response

    def update_database_last_seen(self, person_id: int, timestamp: Time):
        """
        Update the last_seen timestamp in the database for an identified person.

        Args:
            person_id: Database person ID
            timestamp: Timestamp of the identification
        """
        if not self.update_last_seen_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning("update_last_seen service not available")
            return

        request = UpdateLastSeen.Request()
        request.person_id = person_id
        request.last_seen = timestamp

        future = self.update_last_seen_client.call_async(request)
        future.add_done_callback(
            lambda f: self.get_logger().debug(
                f"Updated last_seen for person_id {person_id}: {f.result().message}"
            )
        )

    def publish_state_array(self):
        """
        Publish PersonStateArray with complete world state.

        This is called periodically (10 Hz) to provide downstream nodes with
        the current state of all tracked persons including their identities.
        """
        if not self.person_states:
            return

        msg = PersonStateArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        identified_count = 0
        unidentified_count = 0
        pending_count = 0

        for track_id, state in self.person_states.items():
            person_state = PersonState()
            person_state.header.stamp = state['last_seen']
            person_state.header.frame_id = 'camera_low_res'
            person_state.track_id = track_id
            # Convert person_id to person_name for publishing
            person_id = state['identity']
            if person_id != -1:
                person_state.identity = self.get_person_name(person_id)
            else:
                person_state.identity = 'unknown'
            person_state.identity_confidence = state['identity_confidence']
            person_state.bbox_x, person_state.bbox_y, person_state.bbox_w, person_state.bbox_h = state['bbox']
            person_state.first_seen = state['first_seen']
            person_state.last_seen = state['last_seen']
            person_state.requires_identification = state['identification_pending']  # Maps to identification_pending
            person_state.tracking_confidence = state['tracking_confidence']
            person_state.frames_since_last_seen = state['frames_since_last_seen']

            msg.persons.append(person_state)

            # Count statistics
            if state['identification_pending']:
                pending_count += 1
            elif state['identity'] != -1:
                identified_count += 1
            else:
                unidentified_count += 1

        msg.total_tracked = len(self.person_states)
        msg.identified_count = identified_count
        msg.unidentified_count = unidentified_count
        msg.pending_identification_count = pending_count

        self.state_pub.publish(msg)

    def speech_segment_callback(self, msg: SpeechSegment):
        """
        Process speech segment for voice recognition.

        This is triggered when speech is detected by the microphone node.
        For single speaker scenario, we update all tracked persons with voice evidence.

        Args:
            msg: SpeechSegment with detected speech audio
        """
        if not self.enable_voice_recognition:
            return

        if not self.person_states:
            self.get_logger().debug("Speech detected but no persons being tracked")
            return

        self.get_logger().info(
            f"Processing speech segment: {msg.duration_seconds:.2f}s for voice recognition"
        )

        # Generate voice embedding
        voice_embedding = self.call_generate_voice_embedding(msg)
        if voice_embedding is None:
            self.get_logger().warning("Failed to generate voice embedding")
            return

        # Get voice scores for all known persons
        voice_scores = self.call_recognize_voice(voice_embedding)
        if not voice_scores:
            self.get_logger().warning("Failed to get voice recognition scores")
            return

        # Filter out low-quality voice evidence (likely noise or invalid speech)
        # If max score is too low, don't apply voice update as it would boost "unknown" probability
        max_voice_score = max(voice_scores.values()) if voice_scores else 0.0
        min_voice_confidence_threshold = 0.4  # Minimum score to consider valid speech

        if max_voice_score < min_voice_confidence_threshold:
            self.get_logger().info(
                f"Ignoring voice evidence: max score {max_voice_score:.3f} < threshold {min_voice_confidence_threshold} "
                f"(likely noise or invalid speech)"
            )
            return

        self.get_logger().info(f"Applying voice evidence with max score {max_voice_score:.3f}")

        # For single speaker scenario: Update all tracked persons' Bayesian trackers
        # In multi-speaker scenario with DoA, we would match voice to specific person
        for track_id in list(self.identity_trackers.keys()):
            if track_id in self.person_states:
                tracker = self.identity_trackers[track_id]
                tracker.update_voice(voice_scores)

                # Get updated identity from Bayesian tracker
                identity, confidence = tracker.get_identity()

                # Update voice-specific fields only (for debugging/monitoring)
                state = self.person_states[track_id]
                state['voice_identity'] = identity if isinstance(identity, int) else -1
                state['voice_confidence'] = float(confidence)
                state['last_voice_identification_time'] = msg.header.stamp
                state['is_speaking'] = True

                # Main identity is updated by apply_tracker_decay() from Bayesian tracker
                # This ensures single source of truth from the tracker

                self.get_logger().info(
                    f"Track {track_id}: Voice evidence updated - "
                    f"Bayesian identity: {identity} ({confidence:.3f})"
                )

    def call_generate_voice_embedding(self, speech_msg: SpeechSegment) -> list:
        """
        Call voice embedding generation service.

        Args:
            speech_msg: SpeechSegment with audio data

        Returns:
            list: Voice embedding or None on failure
        """
        if not self.generate_voice_embedding_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("generate_voice_embedding service not available")
            return None

        request = GenerateVoiceEmbedding.Request()
        request.audio_data = speech_msg.audio_data
        request.sample_rate = speech_msg.sample_rate

        try:
            future = self.generate_voice_embedding_client.call_async(request)

            timeout = 5.0  # Voice embedding can take longer
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    self.get_logger().warning("Voice embedding service call timeout")
                    return None
                time.sleep(0.01)

            response = future.result()

            if response.success:
                return list(response.embedding)
            else:
                self.get_logger().warning(f"Voice embedding failed: {response.message}")
                return None

        except Exception as e:
            self.get_logger().error(f"Voice embedding service call failed: {e}")
            return None

    def call_recognize_voice(self, voice_embedding: list) -> dict:
        """
        Call voice recognition service to get scores for all known persons.

        Args:
            voice_embedding: Voice embedding vector

        Returns:
            dict: {person_id: score} for all known persons
        """
        if not self.recognize_voice_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("recognize_voice service not available")
            return {}

        request = RecognizeVoice.Request()
        request.voice_embedding = voice_embedding
        request.confidence_threshold = 0.0  # We want all scores for Bayesian fusion

        try:
            future = self.recognize_voice_client.call_async(request)

            timeout = 2.0
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    return {}
                time.sleep(0.01)

            response = future.result()

            # Convert to dict
            scores = {}
            for i, person_id in enumerate(response.all_person_ids):
                scores[person_id] = float(response.all_scores[i])

            return scores

        except Exception as e:
            self.get_logger().error(f"Voice recognition service call failed: {e}")
            return {}

    def apply_tracker_decay(self):
        """
        Apply time-based confidence decay to all Bayesian trackers.

        Called periodically (1 Hz) to model uncertainty increase over time.
        Also synchronizes state with Bayesian tracker (single source of truth).
        """
        for track_id, tracker in self.identity_trackers.items():
            tracker.predict()  # Apply decay

            # Synchronize state with Bayesian tracker - this is the single source of truth
            if track_id in self.person_states:
                identity, confidence = tracker.get_identity()
                state = self.person_states[track_id]

                # Update identity from Bayesian tracker
                # identity is either person_id (int) or 'unknown' (str)
                if identity != 'unknown':
                    state['identity'] = identity  # person_id (int)
                    state['person_id'] = identity
                else:
                    state['identity'] = -1
                    state['person_id'] = -1

                state['identity_confidence'] = confidence

    def get_person_name(self, person_id: int) -> str:
        """
        Get person name from database by ID.

        Args:
            person_id: Database person ID

        Returns:
            str: Person name or 'unknown'
        """
        if person_id == -1:
            return 'unknown'

        # Check cache first
        if person_id in self.person_name_cache:
            return self.person_name_cache[person_id]

        # Query database
        if not self.get_person_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning(f"get_person service not available, using fallback name for {person_id}")
            return f"person_{person_id}"

        request = GetPerson.Request()
        request.person_id = person_id

        try:
            future = self.get_person_client.call_async(request)

            timeout = 1.0
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    self.get_logger().warning(f"get_person service timeout for person_id {person_id}")
                    return f"person_{person_id}"
                time.sleep(0.01)

            response = future.result()

            if response.found:
                # Cache the result
                self.person_name_cache[person_id] = response.name
                self.get_logger().debug(f"Cached name for person_id {person_id}: {response.name}")
                return response.name
            else:
                self.get_logger().warning(f"Person {person_id} not found in database")
                return f"person_{person_id}"

        except Exception as e:
            self.get_logger().error(f"get_person service call failed: {e}")
            return f"person_{person_id}"

    def update_known_persons(self, person_ids: list):
        """
        Update the list of known person IDs (e.g., after enrollment).

        Args:
            person_ids: List of known person IDs
        """
        self.known_person_ids = list(person_ids)

        # Update all existing trackers
        for tracker in self.identity_trackers.values():
            for pid in person_ids:
                if pid not in tracker.known_person_ids:
                    tracker.add_person(pid)


def main(args=None):
    rclpy.init(args=args)
    node = PersonStateManager()

    # Use MultiThreadedExecutor to allow concurrent callback execution
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
