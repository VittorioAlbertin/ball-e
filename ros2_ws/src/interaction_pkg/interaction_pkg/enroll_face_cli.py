"""
CLI tool to enroll the pending face.
Usage: ros2 run interaction_pkg enroll_face "Person Name" ["Optional notes"]
"""

import rclpy
from rclpy.node import Node
from msgs_interfaces.srv import EnrollPendingFace
import sys


def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: ros2 run interaction_pkg enroll_face <name> [notes]")
        print("Example: ros2 run interaction_pkg enroll_face 'John Doe' 'Friend from work'")
        sys.exit(1)

    name = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else ''

    rclpy.init(args=args)

    # Create temporary node to call service
    node = Node('enroll_face_cli')

    # Create service client
    client = node.create_client(EnrollPendingFace, 'enroll_pending_face')

    print(f"Waiting for enroll_pending_face service...")
    if not client.wait_for_service(timeout_sec=5.0):
        print("ERROR: Service not available!")
        print("Make sure face_enrollment_node is running")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Create request
    request = EnrollPendingFace.Request()
    request.name = name
    request.notes = notes

    print(f"Enrolling pending face as: {name}")

    # Call service
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is not None:
        response = future.result()
        if response.success:
            print(f"✓ SUCCESS: {response.message}")
            print(f"  Person ID: {response.person_id}")
        else:
            print(f"✗ FAILED: {response.message}")
    else:
        print("✗ Service call failed")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
