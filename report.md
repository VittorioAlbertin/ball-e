Development Workflow Reconstruction

  Phase 1: Foundation (April-May 2025)

  Commits: 2b7d162 → 75e6bb8

  Initial Goal: Basic face recognition system
  - First MVP for Face Recognition (May 6)
  - Python-based, no ROS integration yet
  - Simple face database management
  - Focus: Prove face recognition works

  Problems Faced:
  - No framework structure
  - Manual camera handling
  - No persistent storage

  ---
  Phase 2: ROS 2 Integration (September-October 2025)

  Commits: f8782a2 → b0b3233

  Key Developments:
  - Refactoring into ROS 2 workspace (Sep 30)
  - Docker containerization for reproducibility (Oct 6)
    - GPU mounting issues (PR #10)
    - Colcon build automation (PR #9)
  - Person database node added (Oct 19-20, PR #13)
  - Face recognition pipeline (Oct 20-22, PR #14)
    - ByteTrack person tracker
    - YOLO detection
    - Documentation added

  Problems Faced:
  - "piango non va un cazzo" / "neither god knows whats going on" -
  Integration struggles
  - "tryiunbg to debug, everything sucks" - Debugging ROS nodes
  - "idk, works but doesnt work" - Intermittent failures
  - Camera resolution issues (needed 4K for quality faces)
  - Model path management (offline saving)

  Solutions:
  - Systematic refactoring (PR #18, Oct 29-Nov 11)
  - High/low resolution camera streams
  - Proper model caching

  ---
  Phase 3: Audio Pipeline & Voice Recognition (November 14-16, 2025)

  Commits: 7e7d51b → dc3f71f

  Major Pivot: From Localization Fusion to Identification Fusion

  Original Vision:
  - Direction of Arrival (DoA) from microphone array
  - Fuse audio spatial data with visual tracking
  - "Where is the sound coming from?" + "Where is the person?"

  Reality Check:
  - No access to DoA microphone array hardware
  - Pivot decision: Fuse identity instead of location

  New Implementation:
  - Microphone node with VAD (Voice Activity Detection)
  - Resemblyzer for speaker embeddings (256-dim)
  - OpenAI Whisper for speech-to-text
  - SpeechSegment messages for complete utterances

  Problems Faced:
  - "neither god knows whats going on" (Nov 15) - Complex async pipeline
  - Bayesian tracker theory vs practice gap
  - Voice evidence assignment problem (who spoke?)

  Temporary Solution:
  - Apply voice scores to ALL tracked persons
  - Works correctly with single person in frame
  - Placeholder until DoA hardware available

  ---
  Phase 4: Bayesian Tracker as Single Source of Truth (November 17, 2025)

  Commits: dc3f71f → 7eb3764

  Critical Realization: Multiple sources of truth causing inconsistencies

  Problems Identified (from identification_logic_report.md):
  1. Face recognition returning raw similarity, not Bayesian confidence
  2. Identity stored as strings (names) vs ints (person_id) - type mismatch
  3. Voice updates modifying state directly, bypassing Bayesian tracker
  4. Confidence values not comparable between modalities
  5. No database name lookup (get_person_name was broken)
  6. State updated in multiple places, not synchronized

  Solutions Implemented:
  1. Single Source of Truth: Bayesian tracker owns identity
    - Face → updates tracker → returns Bayesian result
    - Voice → updates tracker only (no state modification)
    - State synchronized from tracker via 1 Hz timer
  2. Type Consistency: person_id (int) everywhere internally
    - -1 for unknown
    - Names fetched from DB only at publish time
    - GetPerson service client with caching
  3. Frame Pinning: Explicit cache control
    - Pin frames before identification starts
    - Release after completion
    - Prevents cache misses during slow processing
  4. Confidence Decay Triggers Re-ID:
    - No periodic timers
    - Uncertainty-driven: when confidence drops below threshold
    - min_hits increased to 20 for stable faces
  5. History Limiting: 100 entries max to prevent memory bloat

  ---
  Current State & Future Direction

  What Works Now:
  - Face + Voice Bayesian fusion
  - Single person identification robust
  - Confidence decay triggers re-identification
  - Names resolved from database correctly
  - Frame caching prevents pipeline failures

  Known Limitation (Problem 1 in report):
  # Voice evidence applied to ALL tracks
  for track_id in self.identity_trackers.keys():
      tracker.update_voice(voice_scores)  # WRONG for multi-person

  Future Integration (When DoA Available):
  1. Microphone array provides sound source angle
  2. Match DoA angle with visual track positions
  3. Assign voice evidence ONLY to matching track
  4. Enable true multi-person voice identification

  Remaining Work:
  - Load known_person_ids from database at startup
  - DoA integration for spatial audio
  - Speaker diarization for overlapping speech

  ---
  Key Insights from Development

  1. Iterative Problem Solving: Each commit reveals a discovery → problem →
  solution cycle
  2. Hardware Constraints Drive Design: DoA unavailability forced creative
  pivot
  3. Type Systems Matter: String vs int identity caused cascading bugs
  4. Single Source of Truth: Crucial for multi-modal fusion consistency
  5. Uncertainty Quantification: Bayesian approach naturally handles doubt
  6. Frustration is Part of Process: "piango non va un cazzo" → "enhanced
  identification logic"