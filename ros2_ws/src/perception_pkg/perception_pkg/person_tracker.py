"""
Person Tracker Node for Ball-e Robot

This node implements ByteTrack algorithm for multi-person tracking.
It subscribes to YOLO detections, filters for person detections,
and assigns persistent track IDs across frames.

ByteTrack Reference: https://arxiv.org/abs/2110.06864
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from msgs_interfaces.msg import PersonTrack, PersonTrackArray
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    """Represents a single tracked person."""

    _next_id = 1  # Class variable for generating unique IDs

    def __init__(self, bbox, detection_conf, frame_id):
        """
        Initialize a new track.

        Args:
            bbox: [x, y, w, h] bounding box
            detection_conf: Detection confidence from YOLO
            frame_id: Current frame number
        """
        self.track_id = Track._next_id
        Track._next_id += 1

        self.bbox = np.array(bbox, dtype=np.float32)
        self.detection_conf = detection_conf
        self.tracking_conf = detection_conf

        self.hits = 1  # Number of frames with successful detection
        self.age = 1  # Total frames since track creation
        self.frames_since_update = 0  # Frames without detection
        self.last_frame_id = frame_id
        self.is_confirmed = False

    def update(self, bbox, detection_conf, frame_id):
        """Update track with new detection."""
        self.bbox = np.array(bbox, dtype=np.float32)
        self.detection_conf = detection_conf
        self.tracking_conf = min(1.0, self.tracking_conf * 0.9 + detection_conf * 0.1)

        self.hits += 1
        self.frames_since_update = 0
        self.last_frame_id = frame_id

    def predict(self):
        """Predict next position (for ByteTrack, just keep current position)."""
        self.age += 1
        self.frames_since_update += 1
        # Decay tracking confidence when not updated
        self.tracking_conf *= 0.95

    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.frames_since_update += 1
        self.tracking_conf *= 0.90


def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        bbox1, bbox2: [x, y, w, h] format

    Returns:
        IoU score (0 to 1)
    """
    # Convert to [x1, y1, x2, y2] format
    box1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    box2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]

    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


class PersonTrackerNode(Node):
    """ROS2 node for person tracking using ByteTrack algorithm."""

    def __init__(self):
        super().__init__('person_tracker')

        # Declare and get parameters
        self.declare_parameter('max_age', 30)
        self.declare_parameter('min_hits', 10)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('high_conf_threshold', 0.6)
        self.declare_parameter('low_conf_threshold', 0.1)

        self.max_age = self.get_parameter('max_age').value
        self.min_hits = self.get_parameter('min_hits').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.high_conf_thresh = self.get_parameter('high_conf_threshold').value
        self.low_conf_thresh = self.get_parameter('low_conf_threshold').value

        # Log parameters
        self.get_logger().info(f'Person Tracker initialized with:')
        self.get_logger().info(f'  max_age: {self.max_age}')
        self.get_logger().info(f'  min_hits: {self.min_hits}')
        self.get_logger().info(f'  iou_threshold: {self.iou_threshold}')
        self.get_logger().info(f'  high_conf_threshold: {self.high_conf_thresh}')
        self.get_logger().info(f'  low_conf_threshold: {self.low_conf_thresh}')
        self.get_logger().info('Outputting tracks in low-res coordinates (640x360)')

        # Track management
        self.tracks = []
        self.frame_count = 0

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/yolo/detections',
            self.detection_callback,
            10
        )

        # Publisher
        self.track_pub = self.create_publisher(
            PersonTrackArray,
            '/person_tracker/tracks',
            10
        )

        self.get_logger().info('Person Tracker Node started')

    def detection_callback(self, msg):
        """Process YOLO detections and update tracks."""
        self.frame_count += 1

        # Filter for person detections (class 0 in COCO dataset)
        person_detections = []
        for detection in msg.detections:
            for result in detection.results:
                if result.hypothesis.class_id == 'person' or result.hypothesis.class_id == '0':
                    # Extract bbox (Detection2D uses center + size format)
                    center_x = detection.bbox.center.position.x
                    center_y = detection.bbox.center.position.y
                    size_x = detection.bbox.size_x
                    size_y = detection.bbox.size_y

                    # Convert to [x, y, w, h] format (top-left corner)
                    x = center_x - size_x / 2.0
                    y = center_y - size_y / 2.0
                    w = size_x
                    h = size_y

                    bbox = [x, y, w, h]
                    conf = result.hypothesis.score

                    person_detections.append({
                        'bbox': bbox,
                        'confidence': conf
                    })

        # Update tracks with detections
        self.update_tracks(person_detections)

        # Publish tracks
        self.publish_tracks(msg.header)

    def update_tracks(self, detections):
        """
        Update tracks using ByteTrack two-stage association.

        Stage 1: Associate high-confidence detections with existing tracks
        Stage 2: Associate low-confidence detections with unmatched tracks
        """
        # Separate high and low confidence detections
        high_conf_dets = [d for d in detections if d['confidence'] >= self.high_conf_thresh]
        low_conf_dets = [d for d in detections if self.low_conf_thresh <= d['confidence'] < self.high_conf_thresh]

        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # Stage 1: Match high confidence detections with tracks
        unmatched_tracks, unmatched_dets = self.associate_detections_to_tracks(
            high_conf_dets, self.tracks, self.iou_threshold
        )

        # Stage 2: Match low confidence detections with remaining tracks
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        if len(low_conf_dets) > 0 and len(remaining_tracks) > 0:
            unmatched_tracks_low, _ = self.associate_detections_to_tracks(
                low_conf_dets, remaining_tracks, self.iou_threshold
            )
            # Update unmatched_tracks to only include those still unmatched after stage 2
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_low]

        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets:
            det = high_conf_dets[det_idx]
            new_track = Track(det['bbox'], det['confidence'], self.frame_count)
            self.tracks.append(new_track)
            self.get_logger().info(f'Track {new_track.track_id} created')

        # Mark unmatched tracks and remove old ones
        tracks_to_remove = []
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_missed()

            if track.frames_since_update > self.max_age:
                tracks_to_remove.append(track)

        # Remove dead tracks
        for track in tracks_to_remove:
            self.get_logger().info(f'Track {track.track_id} deleted (age: {track.age}, missed: {track.frames_since_update})')
            self.tracks.remove(track)

        # Update confirmation status
        for track in self.tracks:
            if track.hits >= self.min_hits:
                track.is_confirmed = True

    def associate_detections_to_tracks(self, detections, tracks, iou_threshold):
        """
        Associate detections to tracks using IoU and Hungarian algorithm.

        Returns:
            unmatched_tracks: Indices of tracks without matches
            unmatched_detections: Indices of detections without matches
        """
        if len(tracks) == 0 or len(detections) == 0:
            return list(range(len(tracks))), list(range(len(detections)))

        # Compute IoU cost matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = compute_iou(det['bbox'], track.bbox)

        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1 - iou_matrix

        # Hungarian algorithm for optimal assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        # Filter matches based on IoU threshold
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))

        for det_idx, track_idx in zip(det_indices, track_indices):
            if iou_matrix[det_idx, track_idx] >= iou_threshold:
                # Valid match
                matched_indices.append((det_idx, track_idx))

                # Update track
                track = tracks[track_idx]
                det = detections[det_idx]
                track.update(det['bbox'], det['confidence'], self.frame_count)

                # Remove from unmatched
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)

        return unmatched_tracks, unmatched_detections

    def publish_tracks(self, header):
        """Publish current tracks."""
        track_array = PersonTrackArray()
        track_array.header = header
        track_array.header.frame_id = 'person_tracker'

        for track in self.tracks:
            # Only publish confirmed tracks
            if not track.is_confirmed:
                continue

            track_msg = PersonTrack()
            track_msg.header = header

            track_msg.track_id = track.track_id
            # Output bbox in low-res coordinates (640x360)
            track_msg.bbox_x = float(track.bbox[0])
            track_msg.bbox_y = float(track.bbox[1])
            track_msg.bbox_w = float(track.bbox[2])
            track_msg.bbox_h = float(track.bbox[3])

            track_msg.tracking_confidence = float(track.tracking_conf)
            track_msg.detection_confidence = float(track.detection_conf)
            track_msg.frames_since_last_seen = track.frames_since_update

            # Mark as new if just confirmed
            track_msg.is_new_track = (track.hits == self.min_hits)

            track_array.tracks.append(track_msg)

        self.track_pub.publish(track_array)

        if len(track_array.tracks) > 0:
            track_ids = [t.track_id for t in track_array.tracks]
            self.get_logger().debug(f'Published {len(track_array.tracks)} tracks: {track_ids}')


def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Person Tracker Node')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
