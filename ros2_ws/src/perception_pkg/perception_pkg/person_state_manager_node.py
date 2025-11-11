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
from msgs_interfaces.msg import PersonTrackArray, PersonStateArray, PersonState, PersonTrack
from msgs_interfaces.srv import DetectFace, GenerateEmbedding, RecognizeFace, AddPerson, UpdateLastSeen, EnrollPerson
from builtin_interfaces.msg import Time
import time


class PersonStateManager(Node):
    def __init__(self):
        super().__init__('person_state_manager')

        # Declare parameters
        self.declare_parameter('reidentification_confidence_threshold', 0.4)
        self.declare_parameter('known_person_reidentify_interval', 15.0)
        self.declare_parameter('unknown_person_reidentify_interval', 5.0)
        self.declare_parameter('max_identification_attempts', 5)

        # Get parameters
        self.reidentification_threshold = self.get_parameter('reidentification_confidence_threshold').value
        self.known_interval = self.get_parameter('known_person_reidentify_interval').value
        self.unknown_interval = self.get_parameter('unknown_person_reidentify_interval').value
        self.max_attempts = self.get_parameter('max_identification_attempts').value

        # Internal state: {track_id: PersonStateData}
        self.person_states = {}

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

        # Service server (use service group)
        self.enroll_service = self.create_service(
            EnrollPerson,
            '/person_state/enroll_person',
            self.enroll_person_callback,
            callback_group=self.service_group
        )

        # Publishing timer for state array (use reentrant group)
        self.create_timer(0.1, self.publish_state_array, callback_group=self.reentrant_group)  # 10 Hz

        self.get_logger().info("PersonStateManager initialized (Service-Based with MultiThreadedExecutor)")
        self.get_logger().info(f"  Reidentification threshold: {self.reidentification_threshold}")
        self.get_logger().info(f"  Known person interval: {self.known_interval}s")
        self.get_logger().info(f"  Unknown person interval: {self.unknown_interval}s")
        self.get_logger().info(f"  Max attempts: {self.max_attempts}")

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
                    'identity': 'unknown',
                    'identity_confidence': 0.0,
                    'person_id': -1,
                    'last_identification_time': None,
                    'identification_pending': False,
                    'identification_attempts': 0,
                    'first_seen': track.header.stamp,
                    'last_seen': track.header.stamp,
                    'bbox': (track.bbox_x, track.bbox_y, track.bbox_w, track.bbox_h),
                    'tracking_confidence': track.tracking_confidence,
                    'frames_since_last_seen': track.frames_since_last_seen,
                    'requires_identification': False  # External flag for manual enrollment
                }
                self.get_logger().info(f"New track initialized: {track.track_id}")

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
                state['identification_attempts'] += 1

        # Remove tracks that are no longer present (cleanup)
        tracks_to_remove = [tid for tid in self.person_states.keys() if tid not in current_track_ids]
        for track_id in tracks_to_remove:
            self.get_logger().info(f"Removing lost track: {track_id}")
            del self.person_states[track_id]

    def requires_identification(self, track, state) -> bool:
        """
        Determine if a track requires (re-)identification.

        Args:
            track: PersonTrack message
            state: Internal state dictionary for this track

        Returns:
            bool: True if identification is required
        """
        # Check max attempts first
        if state['identification_attempts'] >= self.max_attempts:
            return False

        # Condition 1: NEW TRACK
        if track.is_new_track and state['identification_attempts'] == 0:
            self.get_logger().info(f"Track {track.track_id}: NEW TRACK - requires identification")
            return True

        # Calculate time since last identification attempt
        time_since_last = None
        if state['last_identification_time'] is not None:
            now = self.get_clock().now()
            last_id_time = rclpy.time.Time.from_msg(state['last_identification_time'])
            time_since_last = (now - last_id_time).nanoseconds / 1e9  # Convert to seconds

        # Determine cooldown period based on current identity
        if state['identity'] != 'unknown':
            cooldown = self.known_interval  # 60 seconds for known persons
        else:
            cooldown = self.unknown_interval  # 15 seconds for unknown persons

        # If we recently tried identification, respect cooldown period
        if time_since_last is not None and time_since_last < cooldown:
            return False

        # Condition 2: LOW CONFIDENCE (but only after cooldown)
        if track.tracking_confidence < self.reidentification_threshold:
            if time_since_last is None or time_since_last >= cooldown:
                self.get_logger().info(
                    f"Track {track.track_id}: LOW CONFIDENCE ({track.tracking_confidence:.2f}) - requires re-identification"
                )
                return True

        # Condition 3: TIME ELAPSED (periodic re-verification)
        if time_since_last is not None and time_since_last >= cooldown:
            self.get_logger().info(
                f"Track {track.track_id} ('{state['identity']}'): " +
                f"{time_since_last:.1f}s elapsed - requires re-verification"
            )
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

        if identity_result['match_found']:
            self.get_logger().info(
                f"Track {track.track_id} identified as '{identity_result['person_name']}' " +
                f"(confidence: {identity_result['similarity_score']:.3f})"
            )
            return {
                'success': True,
                'identity': identity_result['person_name'],
                'confidence': identity_result['similarity_score'],
                'person_id': identity_result['person_id']
            }
        else:
            self.get_logger().info(f"Track {track.track_id}: No match found - marked as unknown")
            return {'success': True, 'identity': 'unknown', 'confidence': 0.0, 'person_id': -1}

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
            return {'match_found': False, 'person_name': '', 'person_id': -1, 'similarity_score': 0.0}

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
                    return {'match_found': False, 'person_name': '', 'person_id': -1, 'similarity_score': 0.0}
                time.sleep(0.01)

            response = future.result()

            return {
                'match_found': response.match_found,
                'person_name': response.person_name,
                'person_id': response.person_id,
                'similarity_score': response.similarity_score
            }

        except Exception as e:
            self.get_logger().error(f"recognize_face service call failed: {e}")
            return {'match_found': False, 'person_name': '', 'person_id': -1, 'similarity_score': 0.0}

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

        # 4. Update state directly
        state['identity'] = request.person_name
        state['person_id'] = add_result['person_id']
        state['identity_confidence'] = 1.0
        state['last_identification_time'] = self.get_clock().now().to_msg()
        state['identification_pending'] = False
        state['requires_identification'] = False

        response.success = True
        response.person_id = add_result['person_id']
        response.message = f"Successfully enrolled '{request.person_name}'"

        self.get_logger().info(f"Enrolled track {track_id} as '{request.person_name}' (person_id: {add_result['person_id']})")

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
            person_state.identity = state['identity']
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
            elif state['identity'] != 'unknown':
                identified_count += 1
            else:
                unidentified_count += 1

        msg.total_tracked = len(self.person_states)
        msg.identified_count = identified_count
        msg.unidentified_count = unidentified_count
        msg.pending_identification_count = pending_count

        self.state_pub.publish(msg)


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
