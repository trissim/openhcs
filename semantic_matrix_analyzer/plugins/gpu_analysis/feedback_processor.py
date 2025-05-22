"""
Feedback Processing System for GPU Analysis Plugin.

This module provides functionality for processing human feedback,
learning from it, and using it to adapt the agent's behavior.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path

# Use relative imports to avoid circular imports
from .dynamic_config import ConfigObserver
from .pattern_extraction import Pattern, PatternMatch, PatternExtractor
from .intent_extraction import Intent, IntentExtractor


class FeedbackType(Enum):
    """Types of feedback that can be received."""

    CORRECTION = "correction"  # Correcting a mistake
    PREFERENCE = "preference"  # Expressing a preference
    CLARIFICATION = "clarification"  # Clarifying a misunderstanding
    CONFIRMATION = "confirmation"  # Confirming a correct action
    REJECTION = "rejection"  # Rejecting a suggestion or action


@dataclass
class Feedback:
    """Represents a piece of feedback from a human."""

    id: str  # Unique identifier
    timestamp: float  # When the feedback was received
    feedback_type: FeedbackType  # Type of feedback
    content: str  # The actual feedback content
    context: Dict[str, Any]  # Context in which the feedback was given
    metadata: Dict[str, Any]  # Additional metadata
    processed: bool = False  # Whether the feedback has been processed


@dataclass
class FeedbackResult:
    """Result of processing feedback."""

    feedback: Feedback  # The original feedback
    success: bool  # Whether processing was successful
    confidence: float  # Confidence in the processing (0.0 to 1.0)
    changes: Dict[str, Any]  # Changes made as a result of the feedback
    intents: List[Intent]  # Intents extracted from the feedback
    patterns: List[Pattern]  # Patterns extracted from the feedback


class FeedbackProcessor(ConfigObserver):
    """
    Processes human feedback and adapts agent behavior.

    This class provides:
    1. Feedback collection and storage
    2. Feedback categorization
    3. Feedback-to-configuration mapping
    4. Learning from feedback over time
    """

    def __init__(self, config_manager, pattern_extractor: PatternExtractor,
                 intent_extractor: IntentExtractor, learning_rate_manager):
        """
        Initialize the feedback processor.

        Args:
            config_manager: Dynamic configuration manager
            pattern_extractor: Pattern extractor
            intent_extractor: Intent extractor
            learning_rate_manager: Learning rate manager
        """
        self.config_manager = config_manager
        self.pattern_extractor = pattern_extractor
        self.intent_extractor = intent_extractor
        self.learning_rate_manager = learning_rate_manager

        # Register as observer for configuration changes
        self.config_manager.register_observer(self)

        self.feedback_history: List[Feedback] = []
        self.feedback_results: Dict[str, FeedbackResult] = {}

        # Initialize default feedback configuration if not present
        config = config_manager.get_config()
        if "feedback" not in config:
            feedback_config = {
                "feedback": {
                    "max_history_size": 100,
                    "min_confidence_threshold": 0.5,
                    "feedback_types": {
                        "correction": {
                            "enabled": True,
                            "weight": 1.0
                        },
                        "preference": {
                            "enabled": True,
                            "weight": 0.8
                        },
                        "clarification": {
                            "enabled": True,
                            "weight": 0.7
                        },
                        "confirmation": {
                            "enabled": True,
                            "weight": 0.9
                        },
                        "rejection": {
                            "enabled": True,
                            "weight": 0.9
                        }
                    },
                    "storage": {
                        "enabled": True,
                        "path": "~/.sma/feedback"
                    }
                }
            }
            config_manager.update_config(feedback_config, "system")

        # Load feedback history if storage is enabled
        self._load_feedback_history()

    def on_config_changed(self, config: Dict[str, Any], changed_keys: Set[str], source: str) -> None:
        """
        Called when configuration changes.

        Args:
            config: The new configuration
            changed_keys: Set of keys that were changed
            source: Source of the change (e.g., "user", "system", "feedback")
        """
        # No specific action needed for now
        pass

    def _load_feedback_history(self) -> None:
        """Load feedback history from storage if enabled."""
        config = self.config_manager.get_config()

        if not config["feedback"]["storage"]["enabled"]:
            return

        storage_path = Path(config["feedback"]["storage"]["path"]).expanduser()
        history_file = storage_path / "feedback_history.json"

        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)

        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    history_data = json.load(f)

                for item in history_data:
                    self.feedback_history.append(Feedback(
                        id=item["id"],
                        timestamp=item["timestamp"],
                        feedback_type=FeedbackType(item["feedback_type"]),
                        content=item["content"],
                        context=item["context"],
                        metadata=item["metadata"],
                        processed=item["processed"]
                    ))

                # Load results if available
                results_file = storage_path / "feedback_results.json"
                if results_file.exists():
                    with open(results_file, "r") as f:
                        results_data = json.load(f)

                    for feedback_id, result in results_data.items():
                        self.feedback_results[feedback_id] = FeedbackResult(
                            feedback=next((f for f in self.feedback_history if f.id == feedback_id), None),
                            success=result["success"],
                            confidence=result["confidence"],
                            changes=result["changes"],
                            intents=result["intents"],
                            patterns=result["patterns"]
                        )
            except Exception as e:
                print(f"Error loading feedback history: {e}")

    def _save_feedback_history(self) -> None:
        """Save feedback history to storage if enabled."""
        config = self.config_manager.get_config()

        if not config["feedback"]["storage"]["enabled"]:
            return

        storage_path = Path(config["feedback"]["storage"]["path"]).expanduser()
        history_file = storage_path / "feedback_history.json"

        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)

        try:
            # Convert feedback objects to dictionaries
            history_data = []
            for feedback in self.feedback_history:
                history_data.append({
                    "id": feedback.id,
                    "timestamp": feedback.timestamp,
                    "feedback_type": feedback.feedback_type.value,
                    "content": feedback.content,
                    "context": feedback.context,
                    "metadata": feedback.metadata,
                    "processed": feedback.processed
                })

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2)

            # Save results if available
            if self.feedback_results:
                results_file = storage_path / "feedback_results.json"
                results_data = {}

                for feedback_id, result in self.feedback_results.items():
                    results_data[feedback_id] = {
                        "success": result.success,
                        "confidence": result.confidence,
                        "changes": result.changes,
                        "intents": result.intents,
                        "patterns": result.patterns
                    }

                with open(results_file, "w") as f:
                    json.dump(results_data, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback history: {e}")

    def add_feedback(self, content: str, feedback_type: Union[FeedbackType, str],
                    context: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Feedback:
        """
        Add a new piece of feedback.

        Args:
            content: The feedback content
            feedback_type: Type of feedback
            context: Context in which the feedback was given
            metadata: Additional metadata

        Returns:
            The created Feedback object
        """
        # Convert string to FeedbackType if necessary
        if isinstance(feedback_type, str):
            try:
                feedback_type = FeedbackType(feedback_type.lower())
            except ValueError:
                feedback_type = FeedbackType.CLARIFICATION  # Default

        # Create feedback object
        feedback = Feedback(
            id=f"feedback_{int(time.time())}_{hash(content) % 10000}",
            timestamp=time.time(),
            feedback_type=feedback_type,
            content=content,
            context=context or {},
            metadata=metadata or {}
        )

        # Add to history
        self.feedback_history.append(feedback)

        # Trim history if needed
        config = self.config_manager.get_config()
        max_history = config["feedback"]["max_history_size"]
        if len(self.feedback_history) > max_history:
            self.feedback_history = self.feedback_history[-max_history:]

        # Save history
        self._save_feedback_history()

        return feedback

    def process_feedback(self, feedback: Union[Feedback, str],
                        context: Optional[Dict[str, Any]] = None) -> FeedbackResult:
        """
        Process a piece of feedback and adapt behavior accordingly.

        Args:
            feedback: The feedback to process (Feedback object or content string)
            context: Additional context for processing

        Returns:
            Result of processing the feedback
        """
        # Convert string to Feedback if necessary
        if isinstance(feedback, str):
            feedback = self.add_feedback(feedback, FeedbackType.CLARIFICATION, context)

        # Skip if already processed
        if feedback.processed and feedback.id in self.feedback_results:
            return self.feedback_results[feedback.id]

        # Get configuration
        config = self.config_manager.get_config()

        # Check if feedback type is enabled
        feedback_type_config = config["feedback"]["feedback_types"].get(
            feedback.feedback_type.value, {"enabled": True, "weight": 1.0}
        )

        if not feedback_type_config.get("enabled", True):
            # Create a result indicating feedback type is disabled
            result = FeedbackResult(
                feedback=feedback,
                success=False,
                confidence=0.0,
                changes={},
                intents=[],
                patterns=[]
            )
            self.feedback_results[feedback.id] = result
            feedback.processed = True
            self._save_feedback_history()
            return result

        # Extract patterns and intents from feedback
        patterns = self.pattern_extractor.extract_patterns(
            feedback.content, "text", feedback.context
        )

        intents = self.intent_extractor.extract_intent(
            feedback.content, "text", feedback.context
        )

        # Determine configuration changes based on feedback
        changes = self._determine_changes(feedback, patterns, intents)

        # Calculate confidence in the changes
        confidence = self._calculate_confidence(feedback, patterns, intents)

        # Apply changes if confidence is high enough
        success = False
        if confidence >= config["feedback"]["min_confidence_threshold"]:
            # Get adaptation rate
            adaptation_rate = self.learning_rate_manager.get_adaptation_rate({
                "confidence": confidence,
                "importance": feedback_type_config.get("weight", 1.0)
            })

            # Apply changes with adaptation rate
            self._apply_changes(changes, adaptation_rate)
            success = True

        # Create result
        result = FeedbackResult(
            feedback=feedback,
            success=success,
            confidence=confidence,
            changes=changes,
            intents=intents,
            patterns=patterns
        )

        # Store result
        self.feedback_results[feedback.id] = result
        feedback.processed = True

        # Update learning rate based on success
        self.learning_rate_manager.update_from_feedback(success, confidence)

        # Save history
        self._save_feedback_history()

        return result

    def _determine_changes(self, feedback: Feedback, patterns: List[Pattern],
                          intents: List[Intent]) -> Dict[str, Any]:
        """
        Determine configuration changes based on feedback.

        Args:
            feedback: The feedback
            patterns: Patterns extracted from feedback
            intents: Intents extracted from feedback

        Returns:
            Dictionary of configuration changes
        """
        changes = {}

        # Process based on feedback type
        if feedback.feedback_type == FeedbackType.CORRECTION:
            # For corrections, look for specific patterns indicating what was wrong
            for pattern in patterns:
                if "weight" in pattern.content.lower():
                    # Extract weight changes
                    changes["weights"] = self._extract_weight_changes(feedback.content)
                elif "threshold" in pattern.content.lower():
                    # Extract threshold changes
                    changes["thresholds"] = self._extract_threshold_changes(feedback.content)

        elif feedback.feedback_type == FeedbackType.PREFERENCE:
            # For preferences, adjust weights based on what the user prefers
            changes["weights"] = self._extract_preference_changes(feedback.content)

        elif feedback.feedback_type == FeedbackType.CLARIFICATION:
            # For clarifications, adjust thresholds to be more precise
            changes["thresholds"] = self._extract_clarification_changes(feedback.content)

        elif feedback.feedback_type == FeedbackType.CONFIRMATION:
            # For confirmations, slightly increase weights of confirmed patterns
            pattern_weights = {}
            for pattern in patterns:
                pattern_weights[pattern.id] = min(pattern.weight * 1.1, 1.0)
            changes["pattern_weights"] = pattern_weights

        elif feedback.feedback_type == FeedbackType.REJECTION:
            # For rejections, decrease weights of rejected patterns
            pattern_weights = {}
            for pattern in patterns:
                pattern_weights[pattern.id] = max(pattern.weight * 0.9, 0.1)
            changes["pattern_weights"] = pattern_weights

        return changes

    def _extract_weight_changes(self, content: str) -> Dict[str, float]:
        """
        Extract weight changes from feedback content.

        Args:
            content: Feedback content

        Returns:
            Dictionary of weight changes
        """
        # This would use NLP to extract weight changes
        # For now, this is a placeholder
        return {}

    def _extract_threshold_changes(self, content: str) -> Dict[str, float]:
        """
        Extract threshold changes from feedback content.

        Args:
            content: Feedback content

        Returns:
            Dictionary of threshold changes
        """
        # This would use NLP to extract threshold changes
        # For now, this is a placeholder
        return {}

    def _extract_preference_changes(self, content: str) -> Dict[str, float]:
        """
        Extract preference changes from feedback content.

        Args:
            content: Feedback content

        Returns:
            Dictionary of preference changes
        """
        # This would use NLP to extract preference changes
        # For now, this is a placeholder
        return {}

    def _extract_clarification_changes(self, content: str) -> Dict[str, float]:
        """
        Extract clarification changes from feedback content.

        Args:
            content: Feedback content

        Returns:
            Dictionary of clarification changes
        """
        # This would use NLP to extract clarification changes
        # For now, this is a placeholder
        return {}

    def _calculate_confidence(self, feedback: Feedback, patterns: List[Pattern],
                             intents: List[Intent]) -> float:
        """
        Calculate confidence in the changes.

        Args:
            feedback: The feedback
            patterns: Patterns extracted from feedback
            intents: Intents extracted from feedback

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence on:
        # 1. Number and confidence of patterns
        # 2. Number and confidence of intents
        # 3. Feedback type weight

        config = self.config_manager.get_config()

        # Get feedback type weight
        feedback_type_weight = config["feedback"]["feedback_types"].get(
            feedback.feedback_type.value, {"weight": 1.0}
        ).get("weight", 1.0)

        # Calculate pattern confidence
        pattern_confidence = 0.0
        if patterns:
            pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)

        # Calculate intent confidence
        intent_confidence = 0.0
        if intents:
            intent_confidence = sum(i.confidence for i in intents) / len(intents)

        # Combine confidences
        combined_confidence = (
            pattern_confidence * 0.4 +
            intent_confidence * 0.4 +
            feedback_type_weight * 0.2
        )

        return combined_confidence

    def _apply_changes(self, changes: Dict[str, Any], adaptation_rate: float) -> None:
        """
        Apply configuration changes with adaptation rate.

        Args:
            changes: Dictionary of changes to apply
            adaptation_rate: Rate at which to apply changes (0.0 to 1.0)
        """
        config = self.config_manager.get_config()
        updates = {}

        # Apply weight changes
        if "weights" in changes:
            if "weights" not in updates:
                updates["weights"] = {}

            for key, value in changes["weights"].items():
                current = config.get("weights", {}).get(key, 0.5)
                updates["weights"][key] = current + (value - current) * adaptation_rate

        # Apply threshold changes
        if "thresholds" in changes:
            if "thresholds" not in updates:
                updates["thresholds"] = {}

            for key, value in changes["thresholds"].items():
                current = config.get("thresholds", {}).get(key, 0.5)
                updates["thresholds"][key] = current + (value - current) * adaptation_rate

        # Apply pattern weight changes
        if "pattern_weights" in changes:
            for pattern_id, weight in changes["pattern_weights"].items():
                if pattern_id in self.pattern_extractor.patterns:
                    pattern = self.pattern_extractor.patterns[pattern_id]
                    current = pattern.weight
                    pattern.weight = current + (weight - current) * adaptation_rate

        # Apply updates if any
        if updates:
            self.config_manager.update_config(updates, "feedback")
