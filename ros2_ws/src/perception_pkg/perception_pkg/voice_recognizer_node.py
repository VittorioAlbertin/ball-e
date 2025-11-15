"""
Voice Recognizer Node for Ball-e Robot

DESCRIPTION:
    Generates speaker embeddings using Resemblyzer for voice identification.
    Provides a service for on-demand embedding generation from audio segments.

MODEL DETAILS:
    - Model: Resemblyzer (based on GE2E loss speaker encoder)
    - Input: Audio waveform (any length, resampled to 16kHz internally)
    - Output: 256-dimensional embedding vector (L2 normalized)
    - Optimized for speaker verification/identification

SUBSCRIPTIONS:
    - /microphone/speech_segment (SpeechSegment): Detected speech segments

SERVICES:
    - /voice_recognition/generate_embedding (GenerateVoiceEmbedding): Generate voice embedding

PUBLICATIONS:
    - /voice_recognition/speaker_embedding (Float32MultiArray): Latest speaker embedding

PARAMETERS:
    - auto_recognize (bool): Automatically process speech segments [default: true]

AUTHOR: Ball-e Team
DATE: 2025-11-15
"""

import rclpy
from rclpy.node import Node
import numpy as np

from msgs_interfaces.msg import SpeechSegment
from msgs_interfaces.srv import GenerateVoiceEmbedding
from std_msgs.msg import Float32MultiArray

# Resemblyzer imports
from resemblyzer import VoiceEncoder, preprocess_wav


class VoiceRecognizerNode(Node):
    def __init__(self):
        super().__init__('voice_recognizer_node')

        # Declare parameters
        self.declare_parameter('auto_recognize', True)

        self.auto_recognize = self.get_parameter('auto_recognize').value

        # Load Resemblyzer model
        self.get_logger().info("Loading Resemblyzer voice encoder model...")
        try:
            self.encoder = VoiceEncoder()
            self.get_logger().info("Resemblyzer model loaded successfully")
            self.get_logger().info(f"  Embedding dimension: 256")
        except Exception as e:
            self.get_logger().error(f"Failed to load Resemblyzer model: {e}")
            raise RuntimeError(f"Failed to load Resemblyzer model: {e}")

        # Subscription for automatic processing
        if self.auto_recognize:
            self.speech_sub = self.create_subscription(
                SpeechSegment,
                '/microphone/speech_segment',
                self.speech_segment_callback,
                10
            )
            self.get_logger().info("Auto-recognition enabled: subscribing to speech segments")

        # Service for on-demand embedding generation
        self.embedding_service = self.create_service(
            GenerateVoiceEmbedding,
            '/voice_recognition/generate_embedding',
            self.generate_embedding_callback
        )

        # Publisher for latest embedding (useful for visualization/debugging)
        self.embedding_pub = self.create_publisher(
            Float32MultiArray,
            '/voice_recognition/speaker_embedding',
            10
        )

        self.get_logger().info("VoiceRecognizerNode initialized")

    def generate_embedding_callback(self, request, response):
        """
        Service callback for generating voice embedding from audio.

        Args:
            request: GenerateVoiceEmbedding.Request with audio_data and sample_rate
            response: GenerateVoiceEmbedding.Response

        Returns:
            Response with embedding or error
        """
        try:
            audio_data = np.array(request.audio_data, dtype=np.float32)
            sample_rate = request.sample_rate

            self.get_logger().info(
                f"Generating voice embedding: {len(audio_data)} samples at {sample_rate} Hz "
                f"({len(audio_data)/sample_rate:.2f}s)"
            )

            # Generate embedding
            embedding = self.compute_embedding(audio_data, sample_rate)

            if embedding is None:
                response.success = False
                response.message = "Failed to compute embedding"
                response.confidence = 0.0
                return response

            response.success = True
            response.embedding = embedding.tolist()
            response.message = "Embedding generated successfully"
            response.confidence = 1.0  # Could compute based on audio quality

            self.get_logger().info(
                f"Voice embedding generated: dim={len(embedding)}, "
                f"norm={np.linalg.norm(embedding):.4f}"
            )

            return response

        except Exception as e:
            response.success = False
            response.message = f"Error generating embedding: {str(e)}"
            response.confidence = 0.0
            self.get_logger().error(response.message)
            return response

    def speech_segment_callback(self, msg):
        """
        Callback for automatic speech segment processing.

        Args:
            msg: SpeechSegment message with detected speech
        """
        audio_data = np.array(msg.audio_data, dtype=np.float32)
        sample_rate = msg.sample_rate

        self.get_logger().info(
            f"Processing speech segment: {msg.duration_seconds:.2f}s, "
            f"RMS={msg.average_rms:.4f}"
        )

        # Generate embedding
        embedding = self.compute_embedding(audio_data, sample_rate)

        if embedding is not None:
            # Publish embedding
            embedding_msg = Float32MultiArray()
            embedding_msg.data = embedding.tolist()
            self.embedding_pub.publish(embedding_msg)

            self.get_logger().info(
                f"Speaker embedding published: dim={len(embedding)}, "
                f"norm={np.linalg.norm(embedding):.4f}"
            )

    def compute_embedding(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute speaker embedding using Resemblyzer.

        Args:
            audio_data: Audio samples (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz

        Returns:
            np.ndarray: 256-dimensional embedding vector or None on failure
        """
        try:
            # Resemblyzer expects audio at 16kHz
            # If sample rate is different, we need to resample
            if sample_rate != 16000:
                # Simple resampling using scipy
                from scipy import signal
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                self.get_logger().debug(f"Resampled audio from {sample_rate}Hz to 16000Hz")

            # Preprocess audio (normalize, trim silence, etc.)
            # Note: Resemblyzer expects audio values in range [-1, 1] which we already have
            wav = preprocess_wav(audio_data)

            if len(wav) < 1600:  # Less than 0.1 seconds at 16kHz
                self.get_logger().warning("Audio too short for embedding generation")
                return None

            # Generate embedding
            embedding = self.encoder.embed_utterance(wav)

            # Normalize to unit length (Resemblyzer should already do this, but be safe)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            self.get_logger().error(f"Error computing embedding: {e}")
            return None

    def embed_multiple_utterances(self, audio_segments: list, sample_rate: int = 16000) -> np.ndarray:
        """
        Compute average embedding from multiple audio segments.
        Useful for enrollment with multiple samples.

        Args:
            audio_segments: List of audio data arrays
            sample_rate: Sample rate (same for all segments)

        Returns:
            np.ndarray: Averaged 256-dimensional embedding
        """
        embeddings = []
        for audio_data in audio_segments:
            emb = self.compute_embedding(audio_data, sample_rate)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            self.get_logger().warning("No valid embeddings computed from segments")
            return None

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        # Re-normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        self.get_logger().info(f"Averaged {len(embeddings)} embeddings")
        return avg_embedding.astype(np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceRecognizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
