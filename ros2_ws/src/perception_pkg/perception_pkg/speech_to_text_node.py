"""
Speech-to-Text Node for Ball-e Robot

DESCRIPTION:
    Transcribes speech to text using OpenAI Whisper model.
    Subscribes to speech segments and publishes transcriptions.

MODEL DETAILS:
    - Model: OpenAI Whisper (tiny, base, small, medium, large)
    - Input: Audio waveform
    - Output: Text transcription
    - Multilingual support

SUBSCRIPTIONS:
    - /microphone/speech_segment (SpeechSegment): Detected speech segments

PUBLICATIONS:
    - /speech/transcription (String): Transcribed text

PARAMETERS:
    - model_size (str): Whisper model size [default: 'base']
    - language (str): Language code or 'auto' [default: 'auto']
    - device (str): Device to use ('cpu' or 'cuda') [default: 'cpu']
    - auto_transcribe (bool): Automatically transcribe segments [default: true]

AUTHOR: Ball-e Team
DATE: 2025-11-15
"""

import rclpy
from rclpy.node import Node
import numpy as np
import os

from msgs_interfaces.msg import SpeechSegment
from std_msgs.msg import String

import whisper


class SpeechToTextNode(Node):
    def __init__(self):
        super().__init__('speech_to_text_node')

        # Declare parameters
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('language', 'auto')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('auto_transcribe', True)

        self.model_size = self.get_parameter('model_size').value
        self.language = self.get_parameter('language').value
        self.device = self.get_parameter('device').value
        self.auto_transcribe = self.get_parameter('auto_transcribe').value

        # Load Whisper model from local models directory
        self.get_logger().info(f"Loading Whisper model: {self.model_size}...")
        try:
            # Use explicit path to models/whisper directory
            models_dir = '/ball-e/ros2_ws/models/whisper'
            os.makedirs(models_dir, exist_ok=True)

            self.get_logger().info(f"Whisper model directory: {models_dir}")
            self.model = whisper.load_model(self.model_size, device=self.device, download_root=models_dir)
            self.get_logger().info(f"Whisper model loaded successfully")
            self.get_logger().info(f"  Model size: {self.model_size}")
            self.get_logger().info(f"  Device: {self.device}")
            self.get_logger().info(f"  Language: {self.language}")
        except Exception as e:
            self.get_logger().error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")

        # Subscription for automatic transcription
        if self.auto_transcribe:
            self.speech_sub = self.create_subscription(
                SpeechSegment,
                '/microphone/speech_segment',
                self.speech_segment_callback,
                10
            )
            self.get_logger().info("Auto-transcription enabled: subscribing to speech segments")

        # Publisher for transcriptions
        self.transcription_pub = self.create_publisher(
            String,
            '/speech/transcription',
            10
        )

        self.get_logger().info("SpeechToTextNode initialized")

    def speech_segment_callback(self, msg):
        """
        Callback for automatic speech segment transcription.

        Args:
            msg: SpeechSegment message with detected speech
        """
        audio_data = np.array(msg.audio_data, dtype=np.float32)
        sample_rate = msg.sample_rate

        self.get_logger().info(
            f"Transcribing speech segment: {msg.duration_seconds:.2f}s"
        )

        # Transcribe
        transcription = self.transcribe_audio(audio_data, sample_rate)

        if transcription:
            # Publish transcription
            transcription_msg = String()
            transcription_msg.data = transcription
            self.transcription_pub.publish(transcription_msg)

            self.get_logger().info(f"Transcription: '{transcription}'")
        else:
            self.get_logger().warning("No transcription produced")

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio using Whisper.

        Args:
            audio_data: Audio samples (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz

        Returns:
            str: Transcribed text or empty string on failure
        """
        try:
            # Whisper expects audio at 16kHz
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                self.get_logger().debug(f"Resampled audio from {sample_rate}Hz to 16000Hz")

            # Pad/trim to 30 seconds (Whisper's expected input length)
            audio_data = whisper.pad_or_trim(audio_data)

            # Make log-mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_data).to(self.model.device)

            # Decode
            options = whisper.DecodingOptions(
                language=None if self.language == 'auto' else self.language,
                fp16=self.device == 'cuda'
            )
            result = whisper.decode(self.model, mel, options)

            return result.text.strip()

        except Exception as e:
            self.get_logger().error(f"Error transcribing audio: {e}")
            return ""

    def transcribe_audio_full(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """
        Transcribe audio with full Whisper output (timestamps, language detection, etc.).

        Args:
            audio_data: Audio samples (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz

        Returns:
            dict: Full Whisper transcription result
        """
        try:
            # Resample if needed
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)

            # Use Whisper's transcribe function for full output
            result = self.model.transcribe(
                audio_data,
                language=None if self.language == 'auto' else self.language,
                fp16=self.device == 'cuda'
            )

            return result

        except Exception as e:
            self.get_logger().error(f"Error transcribing audio: {e}")
            return {}


def main(args=None):
    rclpy.init(args=args)
    node = SpeechToTextNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
