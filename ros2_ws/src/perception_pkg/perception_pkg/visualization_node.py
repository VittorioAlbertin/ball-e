"""
Visualization Node for Ball-e Robot

This node creates annotated visualizations of tracked persons using PersonState
instead of raw detections. Shows:
- Persistent track_id (color-coded)
- Identity name with confidence
- Tracking status (new/tracked/identifying)
- Bounding boxes with color coding
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import ros2_numpy as rnp
from sensor_msgs.msg import Image
from msgs_interfaces.msg import PersonStateArray
import colorsys


class VisualizationNode(Node):
    """Visualization node for person tracking and identification."""

    def __init__(self):
        super().__init__('visualization_node')

        # Declare parameters
        self.declare_parameter('show_track_id', True)
        self.declare_parameter('show_identity', True)
        self.declare_parameter('show_confidence', True)
        self.declare_parameter('show_status', True)
        self.declare_parameter('box_thickness', 2)
        self.declare_parameter('font_scale', 0.6)

        self.show_track_id = self.get_parameter('show_track_id').value
        self.show_identity = self.get_parameter('show_identity').value
        self.show_confidence = self.get_parameter('show_confidence').value
        self.show_status = self.get_parameter('show_status').value
        self.box_thickness = self.get_parameter('box_thickness').value
        self.font_scale = self.get_parameter('font_scale').value

        # Current camera image
        self.current_image = None
        self.current_header = None

        # Color mapping for track IDs (persistent colors)
        self.track_colors = {}

        # Logging
        self.get_logger().info('Visualization Node initialized')
        self.get_logger().info(f'  show_track_id: {self.show_track_id}')
        self.get_logger().info(f'  show_identity: {self.show_identity}')
        self.get_logger().info(f'  show_confidence: {self.show_confidence}')
        self.get_logger().info(f'  show_status: {self.show_status}')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.state_sub = self.create_subscription(
            PersonStateArray,
            '/person_state/all',
            self.state_callback,
            10
        )

        # Publisher
        self.vis_pub = self.create_publisher(
            Image,
            '/visualization/annotated_image',
            10
        )

        self.get_logger().info('Visualization Node started')

    def image_callback(self, msg):
        """Store current camera image."""
        try:
            self.current_image = rnp.numpify(msg)
            self.current_header = msg.header
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def state_callback(self, msg):
        """Process person states and create visualization."""
        if self.current_image is None:
            return

        # Create a copy for annotation
        annotated = self.current_image.copy()

        # Draw statistics
        self._draw_statistics(annotated, msg)

        # Draw each person
        for person in msg.persons:
            self._draw_person(annotated, person)

        # Publish annotated image
        try:
            vis_msg = rnp.msgify(Image, annotated, encoding='bgr8')
            vis_msg.header = self.current_header
            self.vis_pub.publish(vis_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish visualization: {e}")

    def _draw_person(self, image, person):
        """Draw a single person's annotation on the image."""
        track_id = person.track_id

        # Get bounding box
        x = int(person.bbox_x)
        y = int(person.bbox_y)
        w = int(person.bbox_w)
        h = int(person.bbox_h)

        # Validate bbox
        if w <= 0 or h <= 0:
            return

        x_max = x + w
        y_max = y + h

        # Get color for this track (persistent)
        color = self._get_track_color(track_id, person)

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x_max, y_max), color, self.box_thickness)

        # Build label text
        labels = []

        if self.show_track_id:
            labels.append(f"ID:{track_id}")

        if self.show_identity and person.identity != '':
            identity_text = person.identity
            if self.show_confidence:
                identity_text += f" ({person.identity_confidence:.0%})"
            labels.append(identity_text)
        elif person.requires_identification:
            labels.append("Identifying...")
        elif person.identity == '':
            labels.append("Unknown")

        if self.show_status:
            status = self._get_status_text(person)
            if status:
                labels.append(f"[{status}]")

        # Draw labels
        y_offset = y - 10
        for label in labels:
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )

            # Ensure label is within image bounds
            if y_offset - label_h - baseline < 0:
                y_offset = y + h + label_h + baseline + 10

            label_pt1 = (x, y_offset - label_h - baseline - 5)
            label_pt2 = (x + label_w + 10, y_offset + baseline)

            cv2.rectangle(image, label_pt1, label_pt2, color, -1)

            # Draw label text
            cv2.putText(
                image, label,
                (x + 5, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                2
            )

            y_offset -= (label_h + baseline + 10)

        # Draw tracking confidence indicator
        conf_bar_width = w
        conf_bar_height = 5
        conf_bar_x = x
        conf_bar_y = y_max + 5

        # Background bar
        cv2.rectangle(
            image,
            (conf_bar_x, conf_bar_y),
            (conf_bar_x + conf_bar_width, conf_bar_y + conf_bar_height),
            (100, 100, 100),
            -1
        )

        # Confidence bar
        conf_width = int(conf_bar_width * person.tracking_confidence)
        cv2.rectangle(
            image,
            (conf_bar_x, conf_bar_y),
            (conf_bar_x + conf_width, conf_bar_y + conf_bar_height),
            color,
            -1
        )

    def _draw_statistics(self, image, state_msg):
        """Draw summary statistics on the image."""
        stats_text = [
            f"Tracked: {state_msg.total_tracked}",
            f"Identified: {state_msg.identified_count}",
            f"Unknown: {state_msg.unidentified_count}",
            f"Pending ID: {state_msg.pending_identification_count}"
        ]

        y_offset = 30
        for text in stats_text:
            # Draw text background
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            cv2.rectangle(
                image,
                (10, y_offset - text_h - baseline - 5),
                (20 + text_w, y_offset + baseline),
                (0, 0, 0),
                -1
            )

            # Draw text
            cv2.putText(
                image, text,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            y_offset += text_h + baseline + 15

    def _get_track_color(self, track_id, person):
        """Get a consistent color for a track ID."""
        if track_id not in self.track_colors:
            # Generate color based on track_id (hue rotation)
            hue = (track_id * 137.5) % 360  # Golden angle for good distribution
            # Higher saturation and value for identified persons
            saturation = 0.9 if person.identity != '' else 0.7
            value = 0.9 if person.identity != '' else 0.7

            rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            self.track_colors[track_id] = bgr

        # Adjust color based on status
        base_color = self.track_colors[track_id]

        if person.requires_identification:
            # Yellow tint for identifying
            return tuple(min(255, int(c * 1.2)) for c in base_color)
        elif person.identity == '':
            # Dimmer for unknown
            return tuple(int(c * 0.6) for c in base_color)
        else:
            # Brighter for identified
            return base_color

    def _get_status_text(self, person):
        """Get status text for a person."""
        status_parts = []

        # Check if new (seen for less than 3 seconds)
        current_time_ns = self.get_clock().now().nanoseconds
        first_seen_ns = person.first_seen.sec * 1e9 + person.first_seen.nanosec
        time_since_first_seen = (current_time_ns - first_seen_ns) / 1e9

        if time_since_first_seen < 3.0:
            status_parts.append("NEW")

        # Check tracking quality
        if person.tracking_confidence < 0.5:
            status_parts.append("LOW-CONF")

        # Check if not seen recently
        if person.frames_since_last_seen > 5:
            status_parts.append(f"MISS:{person.frames_since_last_seen}")

        return " ".join(status_parts)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Visualization Node')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
