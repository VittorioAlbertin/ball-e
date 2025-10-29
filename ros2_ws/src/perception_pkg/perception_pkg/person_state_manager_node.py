"""
Person State Manager Node

DESCRIPTION:
    The PersonStateManager is the central orchestrator and single source of truth
    for all tracked persons in the system. It maintains the complete state of each
    tracked person including their identity, tracking confidence, and identification
    status. This node decides when identification is required based on configurable
    logic and coordinates the face recognition pipeline.

RESPONSIBILITIES:
    1. Maintain world state of all tracked persons
    2. Decide when identification/re-identification is required
    3. Manage identification timers and retry logic
    4. Coordinate between person tracker and face recognition system
    5. Publish complete PersonStateArray for downstream consumers
    6. Update database with interaction timestamps

IDENTIFICATION LOGIC:
    A person requires identification when any of these conditions are met:
    - NEW TRACK: First time track appears (is_new_track flag)
    - LOW CONFIDENCE: Tracking confidence drops below threshold
    - TIME ELAPSED: Periodic re-verification based on identity status
        * Known persons: Re-identify every known_person_reidentify_interval
        * Unknown persons: Re-identify every unknown_person_reidentify_interval

SUBSCRIPTIONS:
    - /person_tracker/tracks (PersonTrackArray): Track updates from ByteTrack
    - /face_recognizer/results (IdentityUpdate): Identity results from face recognition

PUBLICATIONS:
    - /person_state/array (PersonStateArray): Complete world state with identities
    - /person_state/identification_request (RequestIdentification): Triggers face detection

SERVICE CLIENTS:
    - people_db/update_last_seen: Update interaction timestamp for identified persons

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
    3. If yes: Publish identification request to face detection pipeline
    4. Receive IdentityUpdate from face recognition
    5. Update internal state with identity
    6. Call update_last_seen service for identified persons
    7. Publish PersonStateArray with updated identities

AUTHOR: Vittorio Albertin
DATE: 2025-10-29
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from msgs_interfaces.msg import PersonTrackArray, PersonStateArray, PersonState, IdentityUpdate, IdentificationRequest
from msgs_interfaces.srv import UpdateLastSeen
from builtin_interfaces.msg import Time


class PersonStateManager(Node):
    def __init__(self):
        super().__init__('person_state_manager')

        # Declare parameters
        self.declare_parameter('reidentification_confidence_threshold', 0.4)
        self.declare_parameter('known_person_reidentify_interval', 60.0)
        self.declare_parameter('unknown_person_reidentify_interval', 15.0)
        self.declare_parameter('max_identification_attempts', 5)

        # Get parameters
        self.reidentification_threshold = self.get_parameter('reidentification_confidence_threshold').value
        self.known_interval = self.get_parameter('known_person_reidentify_interval').value
        self.unknown_interval = self.get_parameter('unknown_person_reidentify_interval').value
        self.max_attempts = self.get_parameter('max_identification_attempts').value

        # Internal state: {track_id: PersonStateData}
        self.person_states = {}

        # Subscriptions
        self.track_sub = self.create_subscription(
            PersonTrackArray,
            '/person_tracker/tracks',
            self.track_callback,
            10
        )

        self.identity_sub = self.create_subscription(
            IdentityUpdate,
            '/face_recognizer/results',
            self.identity_callback,
            10
        )

        # Publications
        self.state_pub = self.create_publisher(PersonStateArray, '/person_state/array', 10)
        self.identification_request_pub = self.create_publisher(
            IdentificationRequest,
            '/person_state/identification_request',
            10
        )

        # Service client for database updates
        self.update_last_seen_client = self.create_client(UpdateLastSeen, 'people_db/update_last_seen')

        # Publishing timer for state array
        self.create_timer(0.1, self.publish_state_array)  # 10 Hz

        self.get_logger().info("PersonStateManager initialized")
        self.get_logger().info(f"  Reidentification threshold: {self.reidentification_threshold}")
        self.get_logger().info(f"  Known person interval: {self.known_interval}s")
        self.get_logger().info(f"  Unknown person interval: {self.unknown_interval}s")
        self.get_logger().info(f"  Max attempts: {self.max_attempts}")

    def track_callback(self, msg: PersonTrackArray):
        """
        Process track updates from ByteTrack and decide if identification is needed.

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
                    'frames_since_last_seen': track.frames_since_last_seen
                }
                self.get_logger().info(f"New track initialized: {track.track_id}")

            # Update state from track
            state = self.person_states[track.track_id]
            state['last_seen'] = track.header.stamp
            state['bbox'] = (track.bbox_x, track.bbox_y, track.bbox_w, track.bbox_h)
            state['tracking_confidence'] = track.tracking_confidence
            state['frames_since_last_seen'] = track.frames_since_last_seen

            # Check if identification is required (and not already pending)
            if not state['identification_pending'] and self.requires_identification(track, state):
                self.request_identification(track)
                state['identification_pending'] = True
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

    def request_identification(self, track):
        """
        Publish identification request to trigger face detection pipeline.

        Args:
            track: PersonTrack message
        """
        # Create identification request message
        request = IdentificationRequest()
        request.header = track.header
        request.track_id = int(track.track_id)
        request.bbox_x = int(track.bbox_x)
        request.bbox_y = int(track.bbox_y)
        request.bbox_w = int(track.bbox_w)
        request.bbox_h = int(track.bbox_h)

        # Publish request
        self.identification_request_pub.publish(request)

        self.get_logger().info(f"Identification requested for track {track.track_id}")

    def identity_callback(self, msg: IdentityUpdate):
        """
        Process identity update from face recognition system.

        Args:
            msg: IdentityUpdate containing recognition results
        """
        track_id = msg.track_id

        if track_id not in self.person_states:
            self.get_logger().warning(f"Received identity for unknown track {track_id}")
            return

        state = self.person_states[track_id]

        # Update state with identity
        state['identity'] = msg.identity
        state['identity_confidence'] = msg.confidence
        state['person_id'] = msg.person_id
        state['last_identification_time'] = msg.header.stamp
        state['identification_pending'] = False

        self.get_logger().info(
            f"Track {track_id} identified as '{msg.identity}' " +
            f"(confidence: {msg.confidence:.2f}, person_id: {msg.person_id})"
        )

        # Update database last_seen for identified persons
        if msg.person_id != -1:
            self.update_database_last_seen(msg.person_id, msg.header.stamp)

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
            person_state.requires_identification = state['identification_pending']
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
