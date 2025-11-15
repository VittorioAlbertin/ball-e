# AudioChunk.msg
# Raw audio data chunk from microphone

std_msgs/Header header

# Audio data as 32-bit float samples (normalized -1.0 to 1.0)
float32[] audio_data

# Audio parameters
uint32 sample_rate          # Sample rate in Hz (e.g., 16000)
uint32 num_channels         # Number of channels (1=mono, 2=stereo)
uint32 num_samples          # Number of samples in this chunk

# Quality metrics
float32 rms_level           # Root mean square level (volume indicator)
bool is_clipping            # True if audio is clipping
