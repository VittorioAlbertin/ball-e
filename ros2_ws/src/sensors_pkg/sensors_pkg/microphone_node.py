"""
Microphone Node for Ball-e Robot

DESCRIPTION:
    Captures audio from the system microphone and publishes raw audio chunks.
    Includes basic voice activity detection (VAD) to publish speech segments
    when speech is detected.

PUBLICATIONS:
    - /microphone/audio_raw (AudioChunk): Continuous raw audio stream
    - /microphone/speech_segment (SpeechSegment): Detected speech segments

PARAMETERS:
    - device_index (int): Audio device index (-1 for default) [default: -1]
    - sample_rate (int): Sample rate in Hz [default: 16000]
    - chunk_duration_ms (int): Duration of each audio chunk in ms [default: 100]
    - vad_threshold (float): RMS threshold for voice activity [default: 0.01]
    - speech_min_duration (float): Minimum speech duration in seconds [default: 0.5]
    - speech_max_silence (float): Max silence before ending segment in seconds [default: 1.0]

AUTHOR: Ball-e Team
DATE: 2025-11-15
"""

import rclpy
from rclpy.node import Node
import numpy as np
import sounddevice as sd
from collections import deque
import threading

from msgs_interfaces.msg import AudioChunk, SpeechSegment


class MicrophoneNode(Node):
    def __init__(self):
        super().__init__('microphone_node')

        # Declare parameters
        self.declare_parameter('device_index', -1)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_duration_ms', 200)  # Larger chunks reduce overflow
        self.declare_parameter('vad_threshold', 0.01)
        self.declare_parameter('speech_min_duration', 0.5)
        self.declare_parameter('speech_max_silence', 1.0)
        self.declare_parameter('publish_raw_audio', False)  # Disable by default to reduce load

        # Get parameters
        self.device_index = self.get_parameter('device_index').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_duration_ms = self.get_parameter('chunk_duration_ms').value
        self.vad_threshold = self.get_parameter('vad_threshold').value
        self.speech_min_duration = self.get_parameter('speech_min_duration').value
        self.speech_max_silence = self.get_parameter('speech_max_silence').value
        self.publish_raw_audio = self.get_parameter('publish_raw_audio').value

        # Calculate chunk size in samples
        self.chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)

        # Publishers
        self.audio_pub = self.create_publisher(AudioChunk, '/microphone/audio_raw', 10)
        self.speech_pub = self.create_publisher(SpeechSegment, '/microphone/speech_segment', 10)

        # VAD state
        self.is_speaking = False
        self.speech_buffer = []
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_samples = 0
        self.silence_threshold_samples = int(self.speech_max_silence * self.sample_rate)

        # Audio stream
        self.stream = None
        self.audio_queue = deque(maxlen=100)
        self.lock = threading.Lock()

        # Start audio capture
        self.start_audio_stream()

        # Timer to process audio queue
        self.create_timer(0.01, self.process_audio_queue)  # 100 Hz processing

        self.get_logger().info(f"MicrophoneNode initialized")
        self.get_logger().info(f"  Device: {self.device_index if self.device_index >= 0 else 'default'}")
        self.get_logger().info(f"  Sample rate: {self.sample_rate} Hz")
        self.get_logger().info(f"  Chunk size: {self.chunk_samples} samples ({self.chunk_duration_ms} ms)")
        self.get_logger().info(f"  VAD threshold: {self.vad_threshold}")

    def start_audio_stream(self):
        """Start the audio input stream."""
        device = self.device_index if self.device_index >= 0 else None

        # Get device info
        try:
            if device is None:
                device_info = sd.query_devices(kind='input')
                device = sd.default.device[0]  # Get default input device index
            else:
                device_info = sd.query_devices(device)
            self.get_logger().info(f"Using device: {device_info['name']}")
            self.get_logger().info(f"Device default sample rate: {device_info['default_samplerate']} Hz")
        except Exception as e:
            self.get_logger().error(f"Failed to query device: {e}")
            self._log_available_devices()
            return

        # Try to use requested sample rate, fall back to device default or common rates
        sample_rates_to_try = [
            self.sample_rate,  # Requested rate
            int(device_info['default_samplerate']),  # Device default
            48000,  # Common rates
            44100,
            32000,
            16000,
            8000
        ]
        # Remove duplicates while preserving order
        sample_rates_to_try = list(dict.fromkeys(sample_rates_to_try))

        for rate in sample_rates_to_try:
            try:
                self.get_logger().info(f"Trying sample rate: {rate} Hz")

                # Update sample rate and chunk size
                self.sample_rate = rate
                self.chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)
                self.silence_threshold_samples = int(self.speech_max_silence * self.sample_rate)

                self.stream = sd.InputStream(
                    device=device,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    blocksize=self.chunk_samples,
                    callback=self.audio_callback
                )
                self.stream.start()
                self.get_logger().info(f"Audio stream started successfully at {rate} Hz")
                return

            except Exception as e:
                self.get_logger().warning(f"Failed to start at {rate} Hz: {e}")
                if self.stream:
                    try:
                        self.stream.close()
                    except:
                        pass
                    self.stream = None

        # All rates failed
        self.get_logger().error("Failed to start audio stream with any sample rate")
        self._log_available_devices()

    def _log_available_devices(self):
        """Log available audio input devices."""
        self.get_logger().error("Available input devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                self.get_logger().error(
                    f"  [{i}] {dev['name']} (default: {dev['default_samplerate']} Hz)"
                )

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - runs in separate thread."""
        # Input overflow is common and not critical - only log other issues
        if status and 'input overflow' not in str(status):
            self.get_logger().warning(f"Audio stream status: {status}")

        # Convert to mono if needed and copy data
        audio_data = indata[:, 0].copy()

        # Queue the audio data for processing in main thread
        with self.lock:
            self.audio_queue.append((audio_data, self.get_clock().now().to_msg()))

    def process_audio_queue(self):
        """Process queued audio data in main ROS thread."""
        # Process all queued chunks to avoid overflow
        chunks_to_process = []
        with self.lock:
            while self.audio_queue:
                chunks_to_process.append(self.audio_queue.popleft())

        for audio_data, timestamp in chunks_to_process:
            # Calculate RMS for VAD
            rms = np.sqrt(np.mean(audio_data ** 2))
            is_clipping = bool(np.max(np.abs(audio_data)) > 0.99)

            # Publish raw audio chunk (optional - can be disabled to reduce load)
            if self.publish_raw_audio:
                audio_msg = AudioChunk()
                audio_msg.header.stamp = timestamp
                audio_msg.header.frame_id = 'microphone'
                audio_msg.audio_data = audio_data.tolist()
                audio_msg.sample_rate = int(self.sample_rate)
                audio_msg.num_channels = 1
                audio_msg.num_samples = len(audio_data)
                audio_msg.rms_level = float(rms)
                audio_msg.is_clipping = is_clipping
                self.audio_pub.publish(audio_msg)

            # Voice Activity Detection
            self.process_vad(audio_data, rms, timestamp)

    def process_vad(self, audio_data, rms, timestamp):
        """
        Simple energy-based Voice Activity Detection.

        Args:
            audio_data: Audio samples
            rms: Root mean square energy
            timestamp: ROS timestamp
        """
        is_speech = rms > self.vad_threshold

        # Debug: log RMS levels periodically
        if not hasattr(self, '_rms_log_counter'):
            self._rms_log_counter = 0
        self._rms_log_counter += 1
        if self._rms_log_counter % 50 == 0:  # Every 10 seconds at 200ms chunks
            self.get_logger().info(f"VAD: RMS={rms:.6f}, threshold={self.vad_threshold}, is_speech={is_speech}")

        if is_speech:
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = timestamp
                self.speech_buffer = []
                self.silence_samples = 0
                self.get_logger().info(f"Speech STARTED (RMS={rms:.6f})")

            # Add to buffer
            self.speech_buffer.extend(audio_data.tolist())
            self.last_speech_time = timestamp
            self.silence_samples = 0

        else:  # Not speech
            if self.is_speaking:
                # Count silence duration
                self.silence_samples += len(audio_data)

                # Still add to buffer during short silences
                self.speech_buffer.extend(audio_data.tolist())

                # Check if silence is long enough to end speech
                if self.silence_samples >= self.silence_threshold_samples:
                    # Speech ended
                    self.get_logger().info(f"Speech ENDED (silence={self.silence_samples/self.sample_rate:.2f}s)")
                    self.end_speech_segment(timestamp)

    def end_speech_segment(self, end_time):
        """
        Finalize and publish a speech segment.

        Args:
            end_time: Timestamp when speech ended
        """
        if not self.speech_buffer:
            self.is_speaking = False
            return

        # Calculate duration
        duration = len(self.speech_buffer) / self.sample_rate

        # Only publish if minimum duration met
        if duration >= self.speech_min_duration:
            speech_msg = SpeechSegment()
            speech_msg.header.stamp = end_time
            speech_msg.header.frame_id = 'microphone'
            speech_msg.audio_data = self.speech_buffer
            speech_msg.sample_rate = self.sample_rate
            speech_msg.duration_seconds = float(duration)
            speech_msg.start_time = self.speech_start_time
            speech_msg.end_time = end_time
            speech_msg.confidence = 1.0  # Simple VAD doesn't have confidence
            speech_msg.average_rms = float(np.sqrt(np.mean(np.array(self.speech_buffer) ** 2)))
            speech_msg.is_complete = True

            self.speech_pub.publish(speech_msg)
            self.get_logger().info(f"Speech segment published: {duration:.2f}s, RMS={speech_msg.average_rms:.4f}")

        else:
            self.get_logger().debug(f"Speech segment too short ({duration:.2f}s < {self.speech_min_duration}s), discarding")

        # Reset state
        self.is_speaking = False
        self.speech_buffer = []
        self.speech_start_time = None
        self.silence_samples = 0

    def destroy_node(self):
        """Clean up audio stream on shutdown."""
        self.get_logger().info("Shutting down MicrophoneNode")
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MicrophoneNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
