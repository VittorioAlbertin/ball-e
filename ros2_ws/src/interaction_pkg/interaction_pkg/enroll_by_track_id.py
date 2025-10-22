#!/usr/bin/env python3
"""
CLI tool to enroll a person by their track ID.

Usage: ros2 run interaction_pkg enroll_by_track_id <track_id> <name> [notes]

This tool:
1. Monitors the face recognition pipeline
2. Requests identification for the specified track_id
3. Captures the face embedding from the recognition process
4. Saves the person to the database with the provided name (overwriting if already exists)
"""

import rclpy
from rclpy.node import Node
from msgs_interfaces.msg import IdentityUpdate, PersonStateArray
from msgs_interfaces.srv import AddPerson, RequestIdentification
import sys
import time


class EnrollByTrackIDNode(Node):
    def __init__(self, track_id, name, notes):
        super().__init__('enroll_by_track_id')

        self.target_track_id = track_id
        self.target_name = name
        self.target_notes = notes
        self.embedding = None
        self.enrolled = False
        self.identification_requested = False

        self.get_logger().info(f'Enrolling track_id={track_id} as: {name}')

        # Subscribe to identity updates to capture embeddings
        self.identity_sub = self.create_subscription(
            IdentityUpdate,
            '/face_recognition/identity_update',
            self.identity_callback,
            10
        )

        # Subscribe to person states to verify track exists and trigger identification
        self.state_sub = self.create_subscription(
            PersonStateArray,
            '/person_state/all',
            self.state_callback,
            10
        )

        # Service client to add person
        self.add_person_client = self.create_client(AddPerson, 'people_db/add_person')

        # Service client to request identification
        self.request_id_client = self.create_client(RequestIdentification, '/person_state/request_identification')

        # Wait for services
        self.get_logger().info('Waiting for services...')
        if not self.add_person_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('people_db/add_person service not available!')
            raise RuntimeError('people_db/add_person service not available')

        if not self.request_id_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('request_identification service not available!')
            raise RuntimeError('request_identification service not available')

        self.get_logger().info('Services ready')

        self.track_exists = False
        self.last_seen_time = time.time()

        # Timer to check if track is still active and request identification
        self.timer = self.create_timer(0.5, self.check_and_request_identification)

    def state_callback(self, msg):
        """Monitor person states to verify track exists and get info."""
        for person in msg.persons:
            if person.track_id == self.target_track_id:
                self.track_exists = True
                self.last_seen_time = time.time()

                # Log track info
                if not self.identification_requested:
                    self.get_logger().info(
                        f'Found track_id={self.target_track_id}: '
                        f'identity="{person.identity}", '
                        f'requires_id={person.requires_identification}'
                    )
                return

    def identity_callback(self, msg):
        """Capture embedding when target track is identified."""
        if msg.track_id == self.target_track_id and not self.enrolled:
            self.get_logger().info(f'✓ Captured embedding for track_id={self.target_track_id}')
            self.get_logger().info(f'  Embedding size: {len(msg.face_embedding)}')
            self.get_logger().info(f'  Processing time: {msg.processing_time_ms:.1f}ms')
            self.get_logger().info(f'  Face size: {msg.face_size_pixels:.0f}px')
            self.get_logger().info(f'  Quality OK: {msg.face_quality_ok}')

            if msg.found_in_database:
                self.get_logger().info(f'  Currently recognized as: {msg.identity}')

            # Store embedding (always use latest, even if already in DB)
            self.embedding = msg.face_embedding

            # Enroll immediately
            self.enroll_person()

    def enroll_person(self):
        """Call AddPerson service to save to database."""
        if self.embedding is None:
            self.get_logger().error('No embedding captured yet!')
            return

        request = AddPerson.Request()
        request.name = self.target_name
        request.face_embedding = self.embedding
        request.preferences_json = ''
        request.notes = self.target_notes

        self.get_logger().info(f'Enrolling {self.target_name} to database...')

        future = self.add_person_client.call_async(request)
        future.add_done_callback(self.enrollment_response_callback)

    def enrollment_response_callback(self, future):
        """Handle enrollment response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'✓ SUCCESS: {response.message}')
                self.get_logger().info(f'  Person ID: {response.person_id}')
                self.enrolled = True
                # Shutdown after successful enrollment
                rclpy.shutdown()
            else:
                self.get_logger().error(f'✗ FAILED: {response.message}')
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            rclpy.shutdown()

    def check_and_request_identification(self):
        """Check if track is active and request identification if needed."""
        if not self.track_exists:
            elapsed = time.time() - self.last_seen_time
            if elapsed > 10.0:
                self.get_logger().error(f'Track {self.target_track_id} not found after {elapsed:.1f}s')
                self.get_logger().error('Make sure the person is visible to the camera!')
                self.get_logger().error('Check visualization for correct track ID')
                rclpy.shutdown()
        else:
            # Track exists - request identification if not already done
            if not self.identification_requested and not self.enrolled:
                self.get_logger().info(f'Requesting identification for track_id={self.target_track_id}...')
                self.request_identification()
                self.identification_requested = True

    def request_identification(self):
        """Request identification for the target track."""
        request = RequestIdentification.Request()
        request.track_id = self.target_track_id

        try:
            future = self.request_id_client.call_async(request)
            future.add_done_callback(self.identification_request_callback)
        except Exception as e:
            self.get_logger().error(f'Failed to request identification: {e}')

    def identification_request_callback(self, future):
        """Handle identification request response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'✓ Identification requested successfully')
                self.get_logger().info(f'  Waiting for face recognition to complete...')
            else:
                self.get_logger().error(f'✗ Identification request failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Identification request error: {e}')


def main(args=None):
    if len(sys.argv) < 3:
        print("Usage: ros2 run interaction_pkg enroll_by_track_id <track_id> <name> [notes]")
        print("")
        print("Example:")
        print("  ros2 run interaction_pkg enroll_by_track_id 1 'John Doe' 'Friend from work'")
        print("")
        print("Steps:")
        print("  1. Launch the system: ros2 launch robot_bringup ball_e_full_system_launch.py")
        print("  2. Check visualization to see track IDs")
        print("  3. Run this command with the desired track_id")
        print("")
        sys.exit(1)

    try:
        track_id = int(sys.argv[1])
    except ValueError:
        print(f"ERROR: track_id must be an integer, got: {sys.argv[1]}")
        sys.exit(1)

    name = sys.argv[2]
    notes = sys.argv[3] if len(sys.argv) > 3 else ''

    print(f"")
    print(f"Enrolling track_id={track_id} as '{name}'")
    print(f"Notes: {notes if notes else '(none)'}")
    print(f"")
    print(f"Waiting for identification to complete...")
    print(f"(The system will automatically trigger face recognition for this track)")
    print(f"")

    rclpy.init(args=args)

    try:
        node = EnrollByTrackIDNode(track_id, name, notes)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nCancelled by user")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
