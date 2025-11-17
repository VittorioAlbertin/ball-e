"""
Temporal Bayesian Identity Tracker

DESCRIPTION:
    Tracks identity belief over time using recursive Bayesian updates.
    This is a Bayes Filter for discrete identity states.

    Supports asynchronous updates from face and voice recognition,
    combining evidence over time to produce stable identity estimates.

SCIENTIFIC BASIS:
    - Bayes' Theorem for posterior probability estimation
    - Naive Bayes assumption (conditional independence of modalities)
    - Recursive Bayesian filtering for temporal tracking

USAGE:
    tracker = TemporalBayesianIdentity(known_person_ids)
    tracker.update_face({person_id: score, ...})
    tracker.update_voice({person_id: score, ...})
    identity, confidence = tracker.get_identity()

AUTHOR: Ball-e Team
DATE: 2025-11-15
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional


class TemporalBayesianIdentity:
    """
    Tracks identity belief over time using recursive Bayesian updates.

    This class maintains a probability distribution over all known persons,
    updating it with evidence from face and voice recognition.
    """

    def __init__(
        self,
        known_person_ids: List[int],
        prior_unknown: float = 0.1,
        decay_rate_per_second: float = 0.05
    ):
        """
        Initialize the Bayesian identity tracker.

        Args:
            known_person_ids: List of known person IDs from database
            prior_unknown: Prior probability that person is unknown (0-1)
            decay_rate_per_second: How fast confidence decays without evidence
        """
        self.known_person_ids = list(known_person_ids)
        self.prior_unknown = prior_unknown
        self.decay_rate = decay_rate_per_second

        # Initialize belief distribution (uniform over known persons + unknown)
        self._init_beliefs()

        # Track last update times
        self.last_face_update = None
        self.last_voice_update = None
        self.last_any_update = None
        self.creation_time = time.time()

        # Evidence history (for debugging/analysis)
        self.update_history = []
        self.max_history_size = 100  # Limit history to prevent memory bloat

    def _init_beliefs(self):
        """Initialize belief distribution with uniform prior."""
        n_known = len(self.known_person_ids)
        n_total = n_known + 1  # +1 for unknown
        uniform_prob = 1.0 / n_total

        # All identities (including unknown) start with equal probability
        self.belief = {pid: uniform_prob for pid in self.known_person_ids}
        self.belief['unknown'] = uniform_prob

    def update_face(self, face_scores: Dict[int, float], timestamp: Optional[float] = None):
        """
        Update belief with face recognition scores.

        Args:
            face_scores: {person_id: similarity_score} for all known persons
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        self._update_with_evidence(face_scores, 'face', timestamp)
        self.last_face_update = timestamp
        self.last_any_update = timestamp

    def update_voice(self, voice_scores: Dict[int, float], timestamp: Optional[float] = None):
        """
        Update belief with voice recognition scores.

        Args:
            voice_scores: {person_id: similarity_score} for all known persons
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        self._update_with_evidence(voice_scores, 'voice', timestamp)
        self.last_voice_update = timestamp
        self.last_any_update = timestamp

    def _update_with_evidence(self, scores: Dict[int, float], modality: str, timestamp: float):
        """
        Perform Bayesian update with evidence from a single modality.

        Uses Bayes' theorem:
        P(identity | evidence) ∝ P(evidence | identity) × P(identity)

        Args:
            scores: {person_id: score} for known persons
            modality: 'face' or 'voice'
            timestamp: Time of observation
        """
        if not scores:
            return

        # Convert similarity scores to likelihoods
        # Interpretation: score is P(observation | identity)
        likelihoods = self._scores_to_likelihoods(scores)

        # Bayesian update for each person
        new_belief = {}
        for person_id in self.belief:
            # Prior: current belief
            prior = self.belief[person_id]

            # Likelihood
            if person_id in likelihoods:
                likelihood = likelihoods[person_id]
            elif person_id == 'unknown':
                # Unknown person gets low likelihood from biometric systems
                # (assumes biometric systems are unlikely to match unknown person)
                likelihood = self._compute_unknown_likelihood(scores)
            else:
                # Person not in current scores - use uninformative likelihood
                likelihood = 0.1  # Low but not zero

            # Posterior (unnormalized)
            new_belief[person_id] = prior * likelihood

        # Normalize to ensure valid probability distribution
        self._normalize_belief(new_belief)

        self.belief = new_belief

        # Record update for debugging
        self.update_history.append({
            'timestamp': timestamp,
            'modality': modality,
            'scores': scores.copy(),
            'belief_after': self.belief.copy()
        })

        # Trim history to prevent memory bloat
        if len(self.update_history) > self.max_history_size:
            self.update_history = self.update_history[-self.max_history_size:]

    def _scores_to_likelihoods(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Convert similarity scores to likelihood values.

        For cosine similarity scores in [0, 1]:
        - High score (0.8+) → high likelihood
        - Low score (0.3-) → low likelihood
        - Mid score (0.5) → uninformative

        Args:
            scores: {person_id: similarity_score}

        Returns:
            {person_id: likelihood}
        """
        likelihoods = {}
        for person_id, score in scores.items():
            # Transform score to likelihood
            # This transformation affects fusion behavior significantly
            # Using sigmoid-like transformation centered at 0.5
            # Higher scores → higher likelihood
            likelihood = max(0.001, min(0.999, score))
            likelihoods[person_id] = likelihood

        return likelihoods

    def _compute_unknown_likelihood(self, scores: Dict[int, float]) -> float:
        """
        Compute likelihood for unknown person.

        If all scores are low, it's more likely the person is unknown.
        If some score is high, it's less likely the person is unknown.

        Args:
            scores: {person_id: similarity_score}

        Returns:
            float: Likelihood that person is unknown
        """
        if not scores:
            return 0.5  # Uninformative

        max_score = max(scores.values())

        # If max score is high, unlikely to be unknown
        # If max score is low, more likely to be unknown
        unknown_likelihood = max(0.01, 1.0 - max_score)

        return unknown_likelihood

    def _normalize_belief(self, belief: Dict):
        """
        Normalize belief distribution to sum to 1.

        Args:
            belief: Dictionary to normalize in-place
        """
        total = sum(belief.values())
        if total > 0:
            for key in belief:
                belief[key] /= total
        else:
            # Reset to uniform if degenerate
            n = len(belief)
            for key in belief:
                belief[key] = 1.0 / n

    def predict(self, time_elapsed: float = None):
        """
        Prediction step: model uncertainty increase over time.

        Call this periodically to model the decay of confidence
        when no evidence is available.

        Args:
            time_elapsed: Seconds since last update (computed automatically if None)
        """
        if time_elapsed is None:
            if self.last_any_update is None:
                return
            time_elapsed = time.time() - self.last_any_update

        # Decay towards uniform distribution
        decay_amount = self.decay_rate * time_elapsed
        decay_amount = min(decay_amount, 0.5)  # Cap decay to prevent collapse

        n = len(self.belief)
        uniform_prob = 1.0 / n

        for person_id in self.belief:
            # Interpolate towards uniform
            self.belief[person_id] = (1 - decay_amount) * self.belief[person_id] + \
                                      decay_amount * uniform_prob

    def get_identity(self) -> Tuple[str, float]:
        """
        Get most likely identity and confidence.

        Returns:
            tuple: (identity, confidence)
                - identity: person_id (int) or 'unknown' (str)
                - confidence: probability in [0, 1]
        """
        if not self.belief:
            return 'unknown', 0.0

        best_id = max(self.belief, key=self.belief.get)
        confidence = self.belief[best_id]

        return best_id, confidence

    def get_identity_with_decay(self) -> Tuple[str, float]:
        """
        Get identity with time-based confidence decay applied.

        This doesn't modify internal state, just returns decayed confidence.

        Returns:
            tuple: (identity, decayed_confidence)
        """
        identity, raw_confidence = self.get_identity()

        if self.last_any_update is None:
            return identity, raw_confidence

        # Apply time decay
        time_elapsed = time.time() - self.last_any_update
        decay_factor = (1 - self.decay_rate) ** time_elapsed
        decayed_confidence = raw_confidence * decay_factor

        return identity, decayed_confidence

    def get_all_beliefs(self) -> Dict:
        """
        Get full probability distribution over identities.

        Returns:
            dict: {person_id: probability}
        """
        return self.belief.copy()

    def add_person(self, person_id: int):
        """
        Add a new person to the tracker (e.g., after enrollment).

        Args:
            person_id: New person's database ID
        """
        if person_id in self.known_person_ids:
            return

        self.known_person_ids.append(person_id)

        # Give all identities (including unknown) equal probability
        n_total = len(self.known_person_ids) + 1  # +1 for unknown
        uniform_prob = 1.0 / n_total

        # Recalculate all probabilities
        for pid in self.known_person_ids:
            self.belief[pid] = uniform_prob
        self.belief['unknown'] = uniform_prob

    def remove_person(self, person_id: int):
        """
        Remove a person from the tracker.

        Args:
            person_id: Person's database ID to remove
        """
        if person_id not in self.known_person_ids:
            return

        self.known_person_ids.remove(person_id)
        del self.belief[person_id]

        # Re-normalize remaining beliefs
        self._normalize_belief(self.belief)

    def reset(self):
        """Reset beliefs to initial uniform distribution."""
        self._init_beliefs()
        self.last_face_update = None
        self.last_voice_update = None
        self.last_any_update = None
        self.update_history = []

    def get_time_since_last_update(self) -> Dict[str, Optional[float]]:
        """
        Get time elapsed since last updates.

        Returns:
            dict: {
                'face': seconds or None,
                'voice': seconds or None,
                'any': seconds or None
            }
        """
        now = time.time()
        return {
            'face': now - self.last_face_update if self.last_face_update else None,
            'voice': now - self.last_voice_update if self.last_voice_update else None,
            'any': now - self.last_any_update if self.last_any_update else None
        }

    def __repr__(self):
        identity, confidence = self.get_identity()
        return f"TemporalBayesianIdentity(identity={identity}, confidence={confidence:.3f})"


class MultiTrackIdentityManager:
    """
    Manages multiple TemporalBayesianIdentity trackers for multiple tracks.

    This class provides a convenient interface for managing identity trackers
    for all tracked persons in the scene.
    """

    def __init__(self, known_person_ids: List[int]):
        """
        Initialize the multi-track manager.

        Args:
            known_person_ids: List of known person IDs from database
        """
        self.known_person_ids = list(known_person_ids)
        self.trackers = {}  # {track_id: TemporalBayesianIdentity}

    def create_tracker(self, track_id: int) -> TemporalBayesianIdentity:
        """
        Create a new tracker for a track.

        Args:
            track_id: Track ID from visual tracker

        Returns:
            TemporalBayesianIdentity: New tracker instance
        """
        tracker = TemporalBayesianIdentity(self.known_person_ids)
        self.trackers[track_id] = tracker
        return tracker

    def get_tracker(self, track_id: int) -> Optional[TemporalBayesianIdentity]:
        """
        Get tracker for a track, creating if necessary.

        Args:
            track_id: Track ID

        Returns:
            TemporalBayesianIdentity or None
        """
        if track_id not in self.trackers:
            return self.create_tracker(track_id)
        return self.trackers.get(track_id)

    def remove_tracker(self, track_id: int):
        """
        Remove tracker for a track (when track is lost).

        Args:
            track_id: Track ID to remove
        """
        if track_id in self.trackers:
            del self.trackers[track_id]

    def add_known_person(self, person_id: int):
        """
        Add a new known person to all trackers.

        Args:
            person_id: New person's database ID
        """
        self.known_person_ids.append(person_id)
        for tracker in self.trackers.values():
            tracker.add_person(person_id)

    def get_all_identities(self) -> Dict[int, Tuple[str, float]]:
        """
        Get identities for all tracked persons.

        Returns:
            dict: {track_id: (identity, confidence)}
        """
        return {
            track_id: tracker.get_identity()
            for track_id, tracker in self.trackers.items()
        }

    def cleanup_old_trackers(self, max_age_seconds: float = 60.0):
        """
        Remove trackers that haven't been updated recently.

        Args:
            max_age_seconds: Maximum time without updates before removal
        """
        current_time = time.time()
        to_remove = []

        for track_id, tracker in self.trackers.items():
            if tracker.last_any_update is None:
                age = current_time - tracker.creation_time
            else:
                age = current_time - tracker.last_any_update

            if age > max_age_seconds:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.trackers[track_id]
