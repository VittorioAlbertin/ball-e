"""
Face Enrollment Node
Handles enrolling unknown faces into the people database.

Listens to face recognition messages and prompts the user to add unknown faces.
Prevents spam by only asking once per face (using a cooldown period).
"""

import rclpy
from rclpy.node import Node
from msgs_interfaces.msg import FaceRecognition
from msgs_interfaces.srv import AddPerson, EnrollPendingFace
import time


class FaceEnrollmentNode(Node):
    """
    Node that handles enrolling unknown faces into the database.
    Asks the user if they want to save unknown faces and their name.
    """

    def __init__(self):
        super().__init__('face_enrollment_node')

        # Parameters
        self.declare_parameter('cooldown_seconds', 10.0)  # Don't ask about same face for N seconds
        self.declare_parameter('min_confidence', 0.5)  # Minimum confidence to offer enrollment

        self.cooldown_seconds = self.get_parameter('cooldown_seconds').value
        self.min_confidence = self.get_parameter('min_confidence').value

        # Track when we last saw unknown faces (to prevent spam)
        self.last_unknown_time = 0.0
        self.pending_enrollment = None  # Store pending face data

        # Subscriber to face recognition results
        self.face_sub = self.create_subscription(
            FaceRecognition,
            '/face/recognition',
            self.face_callback,
            10
        )

        # Service client to add people to database
        self.add_person_client = self.create_client(AddPerson, 'people_db/add_person')

        # Service for enrolling pending faces
        self.enroll_service = self.create_service(
            EnrollPendingFace,
            'enroll_pending_face',
            self.enroll_service_callback
        )

        # Wait for add_person service to be available
        self.get_logger().info('Waiting for people_db/add_person service...')
        while not self.add_person_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for people_db/add_person service...')

        self.get_logger().info('Face Enrollment Node initialized')
        self.get_logger().info('Listening for unknown faces...')
        self.get_logger().info('Service available: /enroll_pending_face')

    def face_callback(self, msg):
        """Handle incoming face recognition messages"""
        # Only interested in unknown faces with good confidence
        if msg.found or msg.confidence < self.min_confidence:
            return

        # Check cooldown to prevent spam
        current_time = time.time()
        if current_time - self.last_unknown_time < self.cooldown_seconds:
            return

        # Update last seen time
        self.last_unknown_time = current_time

        # Store the face data
        self.pending_enrollment = {
            'embedding': msg.face_embedding,
            'confidence': msg.confidence,
            'timestamp': current_time
        }

        # Prompt user
        self.get_logger().info('='*60)
        self.get_logger().info('UNKNOWN FACE DETECTED!')
        self.get_logger().info(f'Confidence: {msg.confidence:.2f}')
        self.get_logger().info('='*60)
        self.get_logger().info('Would you like to add this person to the database?')
        self.get_logger().info('Use the add_face service or call /enroll_face action')
        self.get_logger().info('')
        self.get_logger().info('Example:')
        self.get_logger().info('  ros2 run interaction_pkg enroll_face "John Doe"')
        self.get_logger().info('='*60)

    def enroll_service_callback(self, request, response):
        """Service callback to enroll the pending face"""
        try:
            # Validate pending enrollment exists
            if self.pending_enrollment is None:
                self.get_logger().error('No pending face to enroll!')
                response.success = False
                response.message = 'No pending face to enroll'
                response.person_id = -1
                return response

            # Check if enrollment data is still recent (within 60 seconds)
            if time.time() - self.pending_enrollment['timestamp'] > 60.0:
                self.get_logger().error('Pending face enrollment expired (>60 seconds old)')
                self.pending_enrollment = None
                response.success = False
                response.message = 'Enrollment data expired (>60 seconds old)'
                response.person_id = -1
                return response

            # Create service request for add_person
            add_request = AddPerson.Request()
            add_request.name = request.name
            add_request.face_embedding = self.pending_enrollment['embedding']
            add_request.notes = request.notes if request.notes else f"Added via face enrollment (confidence: {self.pending_enrollment['confidence']:.2f})"
            add_request.preferences_json = ''

            # Call service synchronously (we're already in a service callback context)
            self.get_logger().info(f'Enrolling face as: {request.name}')

            # Use synchronous call since we're in a service callback
            add_response = self.add_person_client.call(add_request)

            if add_response is not None:
                if add_response.success:
                    self.get_logger().info(f'Successfully enrolled {request.name} with ID: {add_response.person_id}')
                    self.pending_enrollment = None  # Clear pending enrollment
                    response.success = True
                    response.message = f'Successfully enrolled {request.name}'
                    response.person_id = add_response.person_id
                else:
                    self.get_logger().error(f'Failed to enroll: {add_response.message}')
                    response.success = False
                    response.message = add_response.message
                    response.person_id = -1
            else:
                self.get_logger().error('add_person service call returned None')
                response.success = False
                response.message = 'Failed to call add_person service'
                response.person_id = -1

        except Exception as e:
            self.get_logger().error(f'Exception during enrollment: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f'Exception: {str(e)}'
            response.person_id = -1

        return response


def main(args=None):
    rclpy.init(args=args)
    node = FaceEnrollmentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
