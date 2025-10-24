#!/usr/bin/env python3
"""
Person State Manager Node for Ball-e Robot

This node maintains the "world model" of all tracked people.
It is the single source of truth for person states, combining:
- Tracking information from person_tracker
- Identity information from face recognition
- Temporal state management

It provides services for querying and updating person states.
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from msgs_interfaces.msg import PersonTrackArray, PersonState, PersonStateArray
from msgs_interfaces.srv import GetPersonInfo, RequestIdentification, UpdateIdentity
from builtin_interfaces.msg import Time as TimeMsg
import time


class PersonStateManager(Node):
    """
    Centralized manager for person states.

    Maintains a dictionary of all tracked persons with their complete state
    including tracking, identity, and temporal information.
    """

    def __init__(self):
        super().__init__('person_state_manager')

        # Declare parameters
        self.declare_parameter('cleanup_timeout', 5.0)  # seconds
        self.declare_parameter('publish_rate', 10.0)    # Hz

        self.cleanup_timeout = self.get_parameter('cleanup_timeout').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # Person state storage: {track_id: PersonStateData}
        self.persons = {}

        # Queue for identification requests
        self.identification_queue = []

        # Logging
        self.get_logger().info(f'Person State Manager initialized with:')
        self.get_logger().info(f'  cleanup_timeout: {self.cleanup_timeout}s')
        self.get_logger().info(f'  publish_rate: {self.publish_rate}Hz')

        # Subscribers
        self.track_sub = self.create_subscription(
            PersonTrackArray,
            '/person_tracker/tracks',
            self.track_callback,
            10
        )

        # Publishers
        self.state_pub = self.create_publisher(
            PersonStateArray,
            '/person_state/all',
            10
        )

        # Services
        self.get_info_srv = self.create_service(
            GetPersonInfo,
            '/person_state/get_info',
            self.get_info_callback
        )

        self.request_id_srv = self.create_service(
            RequestIdentification,
            '/person_state/request_identification',
            self.request_identification_callback
        )

        self.update_identity_srv = self.create_service(
            UpdateIdentity,
            '/person_state/update_identity',
            self.update_identity_callback
        )

        # Timers
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_states
        )

        self.cleanup_timer = self.create_timer(
            1.0,  # Check every second
            self.cleanup_stale_persons
        )

        self.get_logger().info('Person State Manager started')

    def track_callback(self, msg):
        """Update person states from tracking information."""
        # Use track timestamp instead of current time to maintain temporal alignment
        current_time = Time.from_msg(msg.header.stamp)
        current_track_ids = set()

        for track in msg.tracks:
            track_id = track.track_id
            current_track_ids.add(track_id)

            if track_id not in self.persons:
                # New person detected
                self.persons[track_id] = {
                    'track_id': track_id,
                    'identity': '',
                    'identity_confidence': 0.0,
                    'bbox_x': track.bbox_x,
                    'bbox_y': track.bbox_y,
                    'bbox_w': track.bbox_w,
                    'bbox_h': track.bbox_h,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'requires_identification': True,
                    'tracking_confidence': track.tracking_confidence,
                    'frames_since_last_seen': track.frames_since_last_seen
                }
                self.get_logger().info(f'New person detected: track_id={track_id}')
            else:
                # Update existing person
                person = self.persons[track_id]
                person['bbox_x'] = track.bbox_x
                person['bbox_y'] = track.bbox_y
                person['bbox_w'] = track.bbox_w
                person['bbox_h'] = track.bbox_h
                person['last_seen'] = current_time
                person['tracking_confidence'] = track.tracking_confidence
                person['frames_since_last_seen'] = track.frames_since_last_seen

    def get_info_callback(self, request, response):
        """Service callback to get information about a person."""
        track_id = request.track_id

        if track_id not in self.persons:
            response.success = False
            response.message = f'Track ID {track_id} not found'
            return response

        person_data = self.persons[track_id]
        response.success = True
        response.message = f'Found person with track_id={track_id}'
        response.person_state = self._create_person_state_msg(person_data)

        return response

    def request_identification_callback(self, request, response):
        """Service callback to request face recognition for a person."""
        track_id = request.track_id

        if track_id not in self.persons:
            response.success = False
            response.message = f'Track ID {track_id} not found'
            response.already_identified = False
            return response

        person = self.persons[track_id]

        # Check if already identified
        if person['identity'] != '':
            response.success = True
            response.message = f'Person already identified as {person["identity"]}'
            response.already_identified = True
            return response

        # Add to identification queue
        if track_id not in self.identification_queue:
            self.identification_queue.append(track_id)
            person['requires_identification'] = True
            self.get_logger().info(f'Identification requested for track_id={track_id}')

        response.success = True
        response.message = f'Identification queued for track_id={track_id}'
        response.already_identified = False

        return response

    def update_identity_callback(self, request, response):
        """Service callback to update person identity (called by face recognition)."""
        track_id = request.track_id
        identity = request.identity
        confidence = request.confidence

        if track_id not in self.persons:
            response.success = False
            response.message = f'Track ID {track_id} not found'
            return response

        person = self.persons[track_id]

        # Update identity
        old_identity = person['identity']
        person['identity'] = identity
        person['identity_confidence'] = confidence
        person['requires_identification'] = False

        # Remove from identification queue
        if track_id in self.identification_queue:
            self.identification_queue.remove(track_id)

        # Log state transition
        if old_identity == '':
            self.get_logger().info(
                f'Person identified: track_id={track_id}, identity={identity}, '
                f'confidence={confidence:.2f}'
            )
        else:
            self.get_logger().info(
                f'Person identity updated: track_id={track_id}, '
                f'old={old_identity}, new={identity}, confidence={confidence:.2f}'
            )

        response.success = True
        response.message = f'Identity updated for track_id={track_id}'

        return response

    def publish_states(self):
        """Publish current person states."""
        state_array = PersonStateArray()
        state_array.header.stamp = self.get_clock().now().to_msg()
        state_array.header.frame_id = 'person_state_manager'

        identified_count = 0
        unidentified_count = 0
        pending_id_count = 0

        for person_data in self.persons.values():
            person_state = self._create_person_state_msg(person_data)
            state_array.persons.append(person_state)

            # Update statistics
            if person_data['identity'] != '':
                identified_count += 1
            else:
                unidentified_count += 1
                if person_data['requires_identification']:
                    pending_id_count += 1

        state_array.total_tracked = len(self.persons)
        state_array.identified_count = identified_count
        state_array.unidentified_count = unidentified_count
        state_array.pending_identification_count = pending_id_count

        self.state_pub.publish(state_array)

    def cleanup_stale_persons(self):
        """Remove persons that haven't been seen for a while."""
        current_time = self.get_clock().now()
        stale_ids = []

        for track_id, person in self.persons.items():
            time_since_seen = (current_time - person['last_seen']).nanoseconds / 1e9

            if time_since_seen > self.cleanup_timeout:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            person = self.persons[track_id]
            identity_str = person['identity'] if person['identity'] != '' else 'unidentified'
            self.get_logger().info(
                f'Lost track: track_id={track_id}, identity={identity_str}, '
                f'inactive for {self.cleanup_timeout}s'
            )

            # Remove from identification queue if present
            if track_id in self.identification_queue:
                self.identification_queue.remove(track_id)

            del self.persons[track_id]

    def _create_person_state_msg(self, person_data):
        """Create a PersonState message from internal data."""
        msg = PersonState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'person_state_manager'

        msg.track_id = person_data['track_id']
        msg.identity = person_data['identity']
        msg.identity_confidence = person_data['identity_confidence']
        msg.bbox_x = person_data['bbox_x']
        msg.bbox_y = person_data['bbox_y']
        msg.bbox_w = person_data['bbox_w']
        msg.bbox_h = person_data['bbox_h']
        msg.first_seen = person_data['first_seen'].to_msg()
        msg.last_seen = person_data['last_seen'].to_msg()
        msg.requires_identification = person_data['requires_identification']
        msg.tracking_confidence = person_data['tracking_confidence']
        msg.frames_since_last_seen = person_data['frames_since_last_seen']

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PersonStateManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Person State Manager')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
