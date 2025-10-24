"""
Identification Coordinator Node for Ball-e Robot

This node coordinates when to trigger face recognition based on person states.
It implements smart triggering logic to balance identification coverage with
computational efficiency.

Triggering Logic:
1. New unidentified tracks
2. Confidence decay (identity confidence drops below threshold)
3. Periodic re-check to avoid persistent misidentification
4. Rate limiting to prevent overwhelming the system
"""

import rclpy
from rclpy.node import Node
from msgs_interfaces.msg import PersonStateArray, IdentityUpdate
from msgs_interfaces.srv import RequestIdentification
import time


class IdentificationCoordinator(Node):
    """Coordinates face recognition triggering based on person states."""

    def __init__(self):
        super().__init__('identification_coordinator')

        # Declare parameters
        self.declare_parameter('max_requests_per_second', 2.0)  # Rate limit
        self.declare_parameter('confidence_threshold', 0.5)  # Re-identify below this
        self.declare_parameter('recheck_interval', 60.0)  # Re-check every N seconds
        self.declare_parameter('new_track_delay', 1.0)  # Wait N seconds before ID'ing new track
        self.declare_parameter('enable_auto_identification', True)

        self.max_requests_per_second = self.get_parameter('max_requests_per_second').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.recheck_interval = self.get_parameter('recheck_interval').value
        self.new_track_delay = self.get_parameter('new_track_delay').value
        self.enable_auto_identification = self.get_parameter('enable_auto_identification').value

        # Logging
        self.get_logger().info('Identification Coordinator initialized with:')
        self.get_logger().info(f'  max_requests_per_second: {self.max_requests_per_second}')
        self.get_logger().info(f'  confidence_threshold: {self.confidence_threshold}')
        self.get_logger().info(f'  recheck_interval: {self.recheck_interval}s')
        self.get_logger().info(f'  new_track_delay: {self.new_track_delay}s')
        self.get_logger().info(f'  enable_auto_identification: {self.enable_auto_identification}')

        # Track state
        self.track_first_seen = {}  # {track_id: timestamp}
        self.track_last_identified = {}  # {track_id: timestamp}
        self.pending_requests = set()  # track_ids currently being processed

        # Rate limiting
        self.request_times = []  # List of request timestamps
        self.rate_limit_window = 1.0  # 1 second window

        # Service client
        self.request_id_client = self.create_client(
            RequestIdentification,
            '/person_state/request_identification'
        )

        # Wait for service
        self.get_logger().info('Waiting for request_identification service...')
        while not self.request_id_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('  Still waiting for /person_state/request_identification...')

        # Subscribers
        self.state_sub = self.create_subscription(
            PersonStateArray,
            '/person_state/all',
            self.state_callback,
            10
        )

        self.identity_update_sub = self.create_subscription(
            IdentityUpdate,
            '/face_recognition/identity_update',
            self.identity_update_callback,
            10
        )

        # Timer for periodic checks
        self.check_timer = self.create_timer(
            5.0,  # Check every 5 seconds
            self.periodic_check
        )

        self.get_logger().info('Identification Coordinator started')

    def state_callback(self, msg):
        """Process person states and trigger identification when needed."""
        if not self.enable_auto_identification:
            return

        current_time = time.time()

        for person in msg.persons:
            track_id = person.track_id

            # Track when we first saw this person
            if track_id not in self.track_first_seen:
                self.track_first_seen[track_id] = current_time
                self.get_logger().debug(f'First time seeing track_id={track_id}')

            # Skip if already processing this track
            if track_id in self.pending_requests:
                continue

            # Rule 1: New unidentified tracks (after delay)
            if person.identity == '' and not person.requires_identification:
                time_since_first_seen = current_time - self.track_first_seen[track_id]
                if time_since_first_seen >= self.new_track_delay:
                    self.get_logger().info(
                        f'Triggering identification for new track: track_id={track_id} '
                        f'(seen for {time_since_first_seen:.1f}s)'
                    )
                    self._request_identification(track_id, 'new_unidentified_track')
                    continue

            # Rule 2: Confidence decay (identity exists but confidence is low)
            if (person.identity != '' and
                person.identity_confidence < self.confidence_threshold):
                self.get_logger().info(
                    f'Triggering re-identification due to low confidence: '
                    f'track_id={track_id}, identity={person.identity}, '
                    f'confidence={person.identity_confidence:.2f}'
                )
                self._request_identification(track_id, 'confidence_decay')
                continue

            # Rule 3: Periodic re-check (avoid persistent misidentification)
            if person.identity != '' and track_id in self.track_last_identified:
                time_since_last_id = current_time - self.track_last_identified[track_id]
                if time_since_last_id >= self.recheck_interval:
                    self.get_logger().info(
                        f'Triggering periodic re-check: track_id={track_id}, '
                        f'identity={person.identity}, '
                        f'last_identified={time_since_last_id:.1f}s ago'
                    )
                    self._request_identification(track_id, 'periodic_recheck')
                    continue

    def identity_update_callback(self, msg):
        """Handle identity update notifications."""
        track_id = msg.track_id

        # Remove from pending requests
        if track_id in self.pending_requests:
            self.pending_requests.remove(track_id)

        # Update last identification time
        self.track_last_identified[track_id] = time.time()

        self.get_logger().debug(
            f'Identity updated for track_id={track_id}: {msg.identity} '
            f'(confidence={msg.confidence:.2f}, processing_time={msg.processing_time_ms:.1f}ms)'
        )

    def periodic_check(self):
        """Periodic check for any missed identification opportunities."""
        # Clean up old request times for rate limiting
        current_time = time.time()
        self.request_times = [
            t for t in self.request_times
            if current_time - t < self.rate_limit_window
        ]

        # Clean up stale pending requests (timeout after 30 seconds)
        stale_requests = []
        for track_id in self.pending_requests:
            if track_id not in self.track_last_identified:
                # Never identified, check if it's been too long
                if track_id in self.track_first_seen:
                    time_pending = current_time - self.track_first_seen[track_id]
                    if time_pending > 30.0:
                        stale_requests.append(track_id)
            else:
                # Check if identification is taking too long
                time_since_request = current_time - self.track_last_identified[track_id]
                if time_since_request > 30.0:
                    stale_requests.append(track_id)

        for track_id in stale_requests:
            self.get_logger().warning(
                f'Removing stale identification request for track_id={track_id}'
            )
            self.pending_requests.remove(track_id)

    def _request_identification(self, track_id, reason):
        """Request identification with rate limiting."""
        current_time = time.time()

        # Check rate limit
        if not self._check_rate_limit():
            self.get_logger().warning(
                f'Rate limit exceeded, deferring identification for track_id={track_id}'
            )
            return

        # Add to pending
        self.pending_requests.add(track_id)

        # Create service request
        request = RequestIdentification.Request()
        request.track_id = track_id

        # Call service asynchronously
        try:
            future = self.request_id_client.call_async(request)

            def handle_response(future):
                try:
                    response = future.result()
                    if response.success:
                        self.get_logger().info(
                            f'Identification requested for track_id={track_id} '
                            f'(reason: {reason}): {response.message}'
                        )
                    else:
                        self.get_logger().warning(
                            f'Identification request failed for track_id={track_id}: '
                            f'{response.message}'
                        )
                        # Remove from pending on failure
                        if track_id in self.pending_requests:
                            self.pending_requests.remove(track_id)
                except Exception as e:
                    self.get_logger().error(
                        f'Failed to process identification response for track_id={track_id}: {e}'
                    )
                    if track_id in self.pending_requests:
                        self.pending_requests.remove(track_id)

            future.add_done_callback(handle_response)

            # Record request time for rate limiting
            self.request_times.append(current_time)

        except Exception as e:
            self.get_logger().error(
                f'Failed to request identification for track_id={track_id}: {e}'
            )
            if track_id in self.pending_requests:
                self.pending_requests.remove(track_id)

    def _check_rate_limit(self):
        """Check if we can make another identification request."""
        current_time = time.time()

        # Clean up old timestamps
        self.request_times = [
            t for t in self.request_times
            if current_time - t < self.rate_limit_window
        ]

        # Check if we're under the limit
        return len(self.request_times) < self.max_requests_per_second


def main(args=None):
    rclpy.init(args=args)
    node = IdentificationCoordinator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Identification Coordinator')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
