# FaceRecognition.msg
# Message published when a face is recognized

std_msgs/Header header  # Timestamp from original frame
int32 track_id
int32 face_x
int32 face_y
int32 face_w
int32 face_h
float32 detection_confidence
