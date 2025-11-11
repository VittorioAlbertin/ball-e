#!/usr/bin/env python3
"""
Identification Visualization Node

DESCRIPTION:
    The IdentificationVisualization node provides real-time visual feedback of the
    person identification system. It subscribes to low-resolution camera images and
    person state data, then draws bounding boxes around tracked persons with labels
    showing their track ID, identity (name or "unknown"), and confidence scores.

RESPONSIBILITIES:
    1. Subscribe to low-res camera images and person state array
    2. Synchronize images with person state data using timestamps
    3. Draw bounding boxes around each tracked person
    4. Annotate boxes with track ID, identity, and confidence
    5. Color-code boxes based on identification status:
       - GREEN: Identified person (has name)
       - YELLOW: Pending identification
       - RED: Unknown person (failed to identify)
    6. Publish annotated images for visualization

SUBSCRIPTIONS:
    - /camera/image_low_res (sensor_msgs/Image): Low-res camera stream (640x360)
    - /person_state/array (PersonStateArray): Person states with identities

PUBLICATIONS:
    - /identification/visualization (sensor_msgs/Image): Annotated images

PARAMETERS:
    - box_thickness (int): Thickness of bounding box lines [default: 2]
    - font_scale (float): Scale of text labels [default: 0.6]
    - show_confidence (bool): Show confidence scores [default: true]
    - show_track_id (bool): Show track IDs [default: true]

VISUALIZATION:
    Box Colors:
    - GREEN (0, 255, 0): Identified person with name
    - YELLOW (0, 255, 255): Pending identification
    - RED (0, 0, 255): Unknown person

    Label Format:
    - With confidence: "ID:1 John Doe (0.85)"
    - Without confidence: "ID:1 John Doe"
    - Pending: "ID:1 Identifying..."
    - Unknown: "ID:1 Unknown"

AUTHOR: Vittorio Albertin
DATE: 2025-10-29
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs_interfaces.msg import PersonStateArray
from cv_bridge import CvBridge
import cv2
import numpy as np


class IdentificationVisualizationNode(Node):
    def __init__(self):
        super().__init__('identification_visualization_node')

        # Declare parameters
        self.declare_parameter('box_thickness', 2)
        self.declare_parameter('font_scale', 0.6)
        self.declare_parameter('show_confidence', True)
        self.declare_parameter('show_track_id', True)

        # Get parameters
        self.box_thickness = self.get_parameter('box_thickness').value
        self.font_scale = self.get_parameter('font_scale').value
        self.show_confidence = self.get_parameter('show_confidence').value
        self.show_track_id = self.get_parameter('show_track_id').value

        # Bridge for ROS-CV conversion
        self.bridge = CvBridge()

        # Latest person state data
        self.latest_person_states = None
        self.person_states_lock = False

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_low_res',
            self.image_callback,
            10
        )

        self.state_sub = self.create_subscription(
            PersonStateArray,
            '/person_state/array',
            self.state_callback,
            10
        )

        # Publications
        self.viz_pub = self.create_publisher(Image, '/identification/visualization', 10)

        self.get_logger().info("IdentificationVisualizationNode initialized")
        self.get_logger().info(f"  Box thickness: {self.box_thickness}")
        self.get_logger().info(f"  Font scale: {self.font_scale}")
        self.get_logger().info(f"  Show confidence: {self.show_confidence}")
        self.get_logger().info(f"  Show track ID: {self.show_track_id}")

    def state_callback(self, msg: PersonStateArray):
        """
        Store latest person state data.

        Args:
            msg: PersonStateArray with current person states
        """
        if not self.person_states_lock:
            self.person_states_lock = True
            self.latest_person_states = msg
            self.person_states_lock = False

    def image_callback(self, msg: Image):
        """
        Process image and draw annotations.

        Args:
            msg: Image message from low-res camera stream
        """
        if self.latest_person_states is None or self.person_states_lock:
            return

        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Draw annotations
            annotated_frame = self.draw_annotations(frame, self.latest_person_states)

            # Publish annotated image
            viz_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            viz_msg.header = msg.header
            self.viz_pub.publish(viz_msg)

        except Exception as e:
            self.get_logger().error(f"Visualization failed: {e}")

    def draw_annotations(self, frame: np.ndarray, person_states: PersonStateArray) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: OpenCV image
            person_states: PersonStateArray with person data

        Returns:
            np.ndarray: Annotated frame
        """
        annotated = frame.copy()

        for person in person_states.persons:
            # Get bounding box (ensure int type for OpenCV)
            x = int(person.bbox_x)
            y = int(person.bbox_y)
            w = int(person.bbox_w)
            h = int(person.bbox_h)

            # Skip invalid bounding boxes
            if w <= 0 or h <= 0:
                continue

            # Determine box color based on identification status
            if person.requires_identification:
                # YELLOW: Pending identification
                color = (0, 255, 255)
                status = "Identifying..."
            elif person.identity and person.identity != 'unknown':
                # GREEN: Identified
                color = (0, 255, 0)
                status = person.identity
            else:
                # RED: Unknown
                color = (0, 0, 255)
                status = "Unknown"

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                color,
                self.box_thickness
            )

            # Build label text
            label_parts = []

            if self.show_track_id:
                label_parts.append(f"ID:{person.track_id}")

            label_parts.append(status)

            if self.show_confidence and person.identity_confidence > 0:
                label_parts.append(f"({person.identity_confidence:.2f})")

            label = " ".join(label_parts)

            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.box_thickness
            )

            # Position label above bounding box
            label_y = int(y - 10 if y - 10 > label_h else y + h + label_h + 10)

            # Draw filled rectangle for text background
            cv2.rectangle(
                annotated,
                (int(x), int(label_y - label_h - baseline)),
                (int(x + label_w), int(label_y + baseline)),
                color,
                -1  # Filled
            )

            # Draw text
            cv2.putText(
                annotated,
                label,
                (int(x), int(label_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),  # Black text
                self.box_thickness
            )

        # Draw summary info
        summary = (
            f"Total: {person_states.total_tracked} | "
            f"Identified: {person_states.identified_count} | "
            f"Unknown: {person_states.unidentified_count} | "
            f"Pending: {person_states.pending_identification_count}"
        )

        # Draw summary background
        (sum_w, sum_h), sum_baseline = cv2.getTextSize(
            summary,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        cv2.rectangle(
            annotated,
            (5, 5),
            (15 + sum_w, 15 + sum_h),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            summary,
            (10, 10 + sum_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = IdentificationVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
