# SpeechSegment.msg
# Detected speech segment with audio data

std_msgs/Header header

# Audio data for the speech segment
float32[] audio_data
uint32 sample_rate

# Timing information
float32 duration_seconds    # Duration of speech in seconds
builtin_interfaces/Time start_time   # When speech started
builtin_interfaces/Time end_time     # When speech ended

# Detection metadata
float32 confidence          # Confidence that this is speech (0-1)
float32 average_rms         # Average volume level
bool is_complete            # True if speech segment is complete (silence detected after)
