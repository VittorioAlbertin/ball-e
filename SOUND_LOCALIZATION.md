# Sound Localization for Ball-e Robot

This document outlines various approaches for implementing azimuth localization of sound sources using a 2-microphone array, with specific focus on speech localization for the Ball-e robot project.

## Project Requirements

- **Hardware**: Compact robot (15-20cm diameter) with integrated 2-microphone array
- **Environment**: Indoor/noisy conditions with reverberation
- **Latency**: Medium (100-200ms acceptable)
- **Accuracy**: Medium (±10-15° azimuth accuracy)
- **Primary use case**: Speech localization (with potential for other sound sources)

## Approaches Overview

### Approach 1: GCC-PHAT (Generalized Cross-Correlation - Phase Transform) ⭐ RECOMMENDED

#### How it works
- Cross-correlates signals from both microphones in frequency domain
- Phase Transform whitens the spectrum, making it robust to reverberation
- Finds time delay that maximizes correlation
- Converts time delay to azimuth angle

#### Pros
- Excellent performance in reverberant/noisy environments (perfect for indoor use case)
- Computationally efficient (can run in real-time easily)
- Well-established, reliable method
- Simple to implement with libraries like `scipy`
- Proven track record in robotics applications

#### Cons
- Requires known microphone spacing
- Performance degrades at very low SNR (<0dB)
- May need spatial aliasing consideration for wide spacing

#### Best for
Indoor robotics with moderate noise and reverberation (Ball-e's use case)

#### Implementation complexity
**Low-Medium** - Standard signal processing libraries available

---

### Approach 2: Classic TDOA with Cross-Correlation

#### How it works
- Direct time-domain cross-correlation between microphone signals
- Finds peak to determine time delay of arrival (TDOA)
- Simpler than GCC-PHAT

#### Pros
- Very simple to implement
- Low computational cost
- Good baseline for testing
- Works well in anechoic/clean environments

#### Cons
- Poor performance in reverberant environments
- Sensitive to noise
- Not suitable for noisy indoor scenarios
- Direct reflections can create false peaks

#### Best for
Controlled environments, outdoor with line-of-sight, testing/prototyping

#### Implementation complexity
**Low** - Basic correlation is straightforward

---

### Approach 3: Deep Learning Based (CRNN / SRP-PHAT + DNN)

#### How it works
- Use Convolutional Recurrent Neural Networks trained on spatial features
- **Option A**: Raw audio → CNN → LSTM → angle estimation
- **Option B**: GCC-PHAT features → DNN → angle estimation
- **Option C**: Spectrogram → 2D CNN → angle regression

#### Pros
- Can learn to handle complex acoustic environments
- Potentially superior performance in challenging conditions
- Can handle multiple simultaneous sources
- Can learn speaker-specific characteristics
- Adaptable to specific robot configuration

#### Cons
- Requires large training dataset with ground truth angles
- Higher computational cost (but manageable for 100-500ms latency)
- More complex to implement and maintain
- Needs retraining for different microphone configurations
- Longer development time

#### Best for
When you have training data and need maximum robustness to noise/reverberation, or when dealing with multiple simultaneous sources

#### Implementation complexity
**High** - Requires ML expertise, dataset creation, training pipeline

---

### Approach 4: Beamforming + DOA Estimation

#### How it works
- Delay-and-sum beamforming with steering vector
- Scan azimuth space and find direction with maximum energy
- Methods: SRP-PHAT (Steered Response Power with PHAT), MUSIC (limited with 2 mics)

#### Pros
- Can provide spatial filtering (noise reduction)
- Dual purpose: localization + signal enhancement
- Can output enhanced audio for downstream processing (ASR, etc.)

#### Cons
- Limited resolution with only 2 microphones
- Computationally more expensive for real-time scanning
- Requires more careful calibration
- Better performance with more microphones

#### Best for
When you also need enhanced audio output, or when upgrading to >2 microphones

#### Implementation complexity
**Medium-High** - More complex algorithms, calibration needed

---

### Approach 5: Phase Difference of Arrival (PDOA)

#### How it works
- Uses phase difference between microphones at specific frequencies
- Works well for narrowband signals
- Particularly effective at voice pitch frequencies (100-300 Hz for speech)

#### Pros
- Very efficient for speech (can focus on voice pitch frequency)
- Simple computation
- Low computational cost

#### Cons
- Limited to narrowband signals
- Phase wrapping issues with wide microphone spacing
- Less robust than GCC-PHAT in noise
- Requires frequency selection/tracking

#### Best for
Speech-only applications with close microphone spacing, supplementary method

#### Implementation complexity
**Low-Medium** - Straightforward but needs phase unwrapping

---

## Recommendation for Ball-e

### Primary Choice: GCC-PHAT (Approach 1)

**Rationale:**

1. **Perfect fit for environment**: Handles indoor reverberation and noise well
2. **Achievable accuracy**: Can easily reach ±10-15° with proper implementation
3. **Real-time capable**: Runs comfortably within 100-200ms latency
4. **Proven in robotics**: Widely used in mobile robots, smart speakers, and service robots
5. **Easy integration**: Works with standard Python audio libraries (scipy, numpy)
6. **Foundation for future work**: Can be enhanced later with:
   - ML refinement on top of GCC-PHAT features
   - Upgrade to more microphones using same principles
   - Hybrid approaches

### Optional Enhancement Path

1. **Phase 1**: Implement GCC-PHAT for immediate results
2. **Phase 2**: Collect real-world data from robot in target environments
3. **Phase 3**: Train a small neural network to refine GCC-PHAT estimates
4. **Phase 4**: Implement sensor fusion with camera-based person tracking

---

## Implementation Plan

### Component 1: Microphone Node (sensors_pkg)

**Purpose**: Capture and publish synchronized stereo audio

**Features**:
- ROS2 node for audio capture
- Support for various USB audio interfaces
- Configurable sample rate (16kHz or 48kHz)
- Publish synchronized audio streams
- Timestamp synchronization for sensor fusion
- Diagnostics (signal levels, clipping detection)

**Topics**:
- `/microphone/audio_left` - Left channel audio
- `/microphone/audio_right` - Right channel audio
- `/microphone/audio_stereo` - Synchronized stereo pair
- `/microphone/diagnostics` - Signal quality metrics

**Technology**:
- `sounddevice` or `pyaudio` for audio capture
- ROS2 Audio message types (or custom)

---

### Component 2: Sound Localization Node (perception_pkg)

**Purpose**: Estimate azimuth angle of sound sources using GCC-PHAT

**Features**:
- GCC-PHAT implementation for azimuth estimation
- Configurable parameters:
  - Frame size (e.g., 1024, 2048 samples)
  - Hop length (e.g., 512 samples)
  - Frequency range for speech (e.g., 200-4000 Hz)
  - Microphone spacing (configurable for different hardware)
- Temporal smoothing/filtering of angle estimates
- Voice activity detection (to process only speech)
- Confidence scoring

**Topics**:
- Subscribe: `/microphone/audio_stereo`
- Publish: `/sound_localization/azimuth` - Estimated angle
- Publish: `/sound_localization/confidence` - Confidence score
- Publish: `/sound_localization/visualization` - Debug visualization

**Parameters**:
- `microphone_spacing` (meters)
- `sample_rate` (Hz)
- `frame_size` (samples)
- `hop_length` (samples)
- `freq_min`, `freq_max` (Hz) - frequency range for processing
- `smoothing_window` (frames) - temporal smoothing
- `vad_threshold` (float) - voice activity detection threshold

**Technology**:
- `scipy.signal` for cross-correlation and signal processing
- `numpy` for numerical operations
- Custom GCC-PHAT implementation

---

### Component 3: Message Interfaces (msgs_interfaces)

**Purpose**: Define custom message types for audio and localization data

**Messages to add**:

```
# AudioFrame.msg - Stereo audio frame
std_msgs/Header header
float32[] left_channel
float32[] right_channel
uint32 sample_rate
uint32 frame_size

# SoundDirection.msg - Sound source direction
std_msgs/Header header
float32 azimuth_deg        # Azimuth angle in degrees (-180 to 180)
float32 confidence         # Confidence score (0.0 to 1.0)
string source_type         # "speech", "noise", "unknown"
```

---

### Component 4: Integration & Launch Files

**Purpose**: Integrate sound localization with existing Ball-e system

**Launch files** (robot_bringup):
- `microphone_launch.py` - Launch microphone node only (for testing)
- `sound_localization_launch.py` - Launch full audio localization pipeline
- `multimodal_tracking_launch.py` - Integrate audio + visual tracking

**Sensor Fusion Potential**:
- Combine visual person tracking with audio direction
- Match tracked person IDs with sound source directions
- Enhance person state with "currently speaking" flag
- Improve tracking robustness when person is out of camera view

**Integration points**:
- Person state manager can subscribe to `/sound_localization/azimuth`
- Match audio azimuth with visual person positions
- Trigger face recognition when speech detected from specific direction

---

### Component 5: Testing & Calibration Utilities

**Purpose**: Tools for testing, calibration, and debugging

**Utilities**:

1. **Microphone Calibration Script**
   - Measure actual microphone spacing
   - Test impulse response
   - Verify channel synchronization
   - Speed of sound calibration for environment

2. **Visualization Tools**
   - Real-time azimuth plot
   - Cross-correlation visualization
   - Spectrogram display
   - Polar plot of sound source direction

3. **Test Scripts**
   - Known angle test (place speaker at known positions)
   - Accuracy measurement script
   - Multi-source separation test
   - Reverberation robustness test

4. **Diagnostic Node**
   - Monitor signal-to-noise ratio
   - Detect clipping or saturation
   - Measure latency end-to-end
   - Log performance metrics

---

## Technical Details

### GCC-PHAT Algorithm Implementation

```
1. Capture stereo audio frames (left and right channels)
2. Apply window function (Hamming/Hanning)
3. Compute FFT of both signals
4. Calculate cross-power spectrum: R(f) = X_left(f) * conj(X_right(f))
5. Apply PHAT weighting: R_phat(f) = R(f) / |R(f)|
6. Compute IFFT to get cross-correlation in time domain
7. Find peak of cross-correlation → time delay τ
8. Convert time delay to azimuth: θ = arcsin(c*τ / d)
   where c = speed of sound (343 m/s), d = microphone spacing
```

### Key Parameters

- **Sample Rate**: 16 kHz (sufficient for speech) or 48 kHz (higher quality)
- **Frame Size**: 1024-2048 samples (trade-off between time/frequency resolution)
- **Hop Length**: 512 samples (50% overlap is typical)
- **Frequency Range**: 200-4000 Hz (speech bandwidth)
- **Microphone Spacing**: 10-15 cm recommended for compact robot

### Spatial Resolution

With 2 microphones, azimuth resolution depends on:
- Microphone spacing (wider = better resolution)
- Sample rate (higher = finer time delay resolution)
- Frequency content (higher frequencies = better resolution)

**Expected accuracy with recommended setup**:
- ±10° in good conditions (SNR > 10 dB)
- ±15° in noisy conditions (SNR 0-10 dB)
- Degrades gracefully in reverberation

### Computational Requirements

**GCC-PHAT per frame**:
- 2x FFT (N log N complexity)
- 1x IFFT (N log N complexity)
- Element-wise operations (linear)

**Estimated CPU**: <5% on modern ARM processor (Raspberry Pi 4) at 16 kHz, 1024 frame size

**Latency breakdown**:
- Audio buffering: ~64-128 ms (depends on buffer size)
- GCC-PHAT computation: <10 ms
- Publishing/communication: ~5-10 ms
- **Total**: <150 ms (well within requirement)

---

## Hardware Recommendations

### When Ready for Hardware Integration

**Microphone Spacing**: 10-15 cm
- Good balance for speech frequencies
- Avoids spatial aliasing up to ~8 kHz
- Fits within 15-20 cm robot diameter

**Sample Rate**: 16 kHz or 48 kHz
- 16 kHz: Sufficient for speech, lower computational cost
- 48 kHz: Better resolution, useful for broader sound sources

**Microphone Type**: MEMS or Electret
- MEMS: Compact, consistent, good SNR, easy integration
- Electret: Potentially better sound quality, larger

**Recommended Hardware**:

1. **USB Audio Interfaces**:
   - Focusrite Scarlett 2i2 (professional quality)
   - Behringer U-Phoria UMC202HD (budget option)
   - Any USB stereo interface with simultaneous capture

2. **All-in-One Arrays** (if using Raspberry Pi):
   - ReSpeaker 2-Mic HAT (affordable, proven)
   - Matrix Voice (more mics, expandable)
   - Seed Studio ReSpeaker USB (plug-and-play)

3. **Custom Build**:
   - 2x INMP441 MEMS microphones (I2S interface)
   - 2x Electret microphones + stereo ADC
   - Requires custom PCB or breadboard

**Calibration Needs**:
- Measure exact microphone spacing (ruler or calipers)
- Verify left/right channel assignment
- Test impulse response for channel matching
- Optional: measure temperature for speed of sound correction

---

## Future Enhancements

### Phase 1 (Current): Basic GCC-PHAT
- Implement core GCC-PHAT algorithm
- Single source localization
- ROS2 integration

### Phase 2: Robustness Improvements
- Voice activity detection (only process speech)
- Multi-hypothesis tracking (handle multiple peaks)
- Temporal filtering (Kalman filter for smooth tracking)
- Adaptive frequency range selection

### Phase 3: Multi-Modal Integration
- Sensor fusion with camera
- Match audio direction with visual person tracks
- Speaking person identification
- Audio-visual attention mechanism

### Phase 4: Advanced Methods
- Add more microphones (upgrade to 4-mic array)
- Deep learning refinement on GCC-PHAT features
- Sound source separation before localization
- 3D localization (azimuth + elevation)

### Phase 5: Higher-Level Features
- Speaker diarization (who is speaking when)
- Sound event classification (speech vs. other sounds)
- Spatial audio map (track multiple sources over time)
- Integration with natural language processing

---

## References and Resources

### Academic Papers
- "The Generalized Cross Correlation Method for Estimation of Time Delay" - Knapp & Carter, 1976
- "Robust DOA Estimation Using GCC-PHAT" - Various authors
- "Sound Source Localization in Robotics: A Survey" - Recent reviews

### Software Libraries
- **pyroomacoustics**: Python library for room acoustics simulation and array processing
- **scipy.signal**: Cross-correlation, FFT, signal processing
- **sounddevice**: Python audio I/O
- **librosa**: Audio analysis (for features, onset detection, etc.)

### Existing Implementations
- [ODAS (Open embeddeD Audition System)](https://github.com/introlab/odas) - Open source sound localization
- [BeamformIt](https://github.com/xanguera/BeamformIt) - Acoustic beamforming
- [SSL (Sound Source Localization) examples](https://github.com/topics/sound-source-localization)

### Datasets for Testing
- **LOCATA Challenge**: Recorded audio with ground truth positions
- **DIRHA**: Distant speech recognition in home environments
- **TAU Spatial Sound Events**: Spatial audio recordings

---

## Integration with Ball-e Architecture

### Current System
Ball-e currently has:
- `sensors_pkg`: Camera node
- `perception_pkg`: YOLO detection, face recognition, person tracking
- `navigation_pkg`: Movement control
- `interaction_pkg`: Person state management, database
- `emotion_pkg`: Emotional states

### Sound Localization Fit

**Add to `sensors_pkg`**:
- `microphone_node.py` - Audio capture

**Add to `perception_pkg`**:
- `sound_localizer_node.py` - GCC-PHAT localization
- `utils/gcc_phat.py` - Core algorithm
- `utils/audio_processing.py` - Preprocessing utilities

**Extend `msgs_interfaces`**:
- `AudioFrame.msg`
- `SoundDirection.msg`

**Add launch files to `robot_bringup`**:
- `microphone_launch.py`
- `sound_localization_launch.py`
- `audio_visual_tracking_launch.py`

### Sensor Fusion Strategy

**Match audio with visual tracking**:
1. Sound localization provides azimuth angle
2. Visual tracking provides person positions and track IDs
3. Fusion node compares azimuth with person angles from camera
4. Best match → assign "speaking" attribute to person
5. Update person state in `person_state_manager`

**Benefits**:
- Know who is speaking in multi-person scenarios
- Track persons even when they leave camera FOV (using audio)
- Trigger face recognition when someone speaks
- Improve human-robot interaction (look at speaking person)

---

## Conclusion

GCC-PHAT is the recommended starting point for Ball-e's sound localization:
- Proven, reliable method
- Matches requirements perfectly
- Easy to implement and integrate
- Provides foundation for future enhancements

The implementation will follow ROS2 best practices and integrate seamlessly with Ball-e's existing perception pipeline, enabling multi-modal person tracking and enhanced human-robot interaction.

**Next Steps**:
1. Acquire 2-microphone USB audio interface for testing
2. Implement microphone_node for audio capture
3. Implement sound_localizer_node with GCC-PHAT
4. Test with known speaker positions
5. Integrate with person tracking system
6. Deploy and evaluate in target environment
