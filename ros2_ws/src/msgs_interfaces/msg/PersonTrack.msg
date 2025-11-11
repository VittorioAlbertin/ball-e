# PersonTrack.msg
# Message for person tracking with persistent IDs

std_msgs/Header header

# Persistent tracking ID assigned by the tracker
int32 track_id

# Bounding box in image coordinates (x, y, width, height)
float32 bbox_x
float32 bbox_y
float32 bbox_w
float32 bbox_h

# Tracking confidence score (0.0 to 1.0)
float32 tracking_confidence

# Number of frames since this track was last detected
int32 frames_since_last_seen

# Flag indicating if this is a newly created track
bool is_new_track

# Detection confidence from YOLO (0.0 to 1.0)
float32 detection_confidence