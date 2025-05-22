"""
Dynamic Configuration Manager for GPU Analysis Plugin.

This module provides a dynamic configuration manager for the GPU Analysis Plugin
that can self-modify based on human feedback regarding intention, enabling
adaptive analysis and continuous improvement.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import datetime
import json
import torch
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class DynamicConfigManager:
    """
    Dynamic configuration manager for the GPU Analysis Plugin.

    This class manages the dynamic configuration of the GPU Analysis Plugin,
    allowing it to adapt based on human feedback regarding intention.
    """

    def __init__(self, initial_config: Dict[str, Any], context: Optional[Any] = None):
        """
        Initialize the dynamic configuration manager.

        Args:
            initial_config: Initial configuration dictionary
            context: Optional plugin context for logging
        """
        self.config = initial_config
        self.context = context
        self.feedback_history = []
        self.learning_rate = 0.1  # Rate at which to adjust weights based on feedback

        # Log initialization
        self.log("info", "Dynamic configuration manager initialized")

    def log(self, level: str, message: str) -> None:
        """
        Log a message using the context logger if available.

        Args:
            level: Log level
            message: Message to log
        """
        if self.context and hasattr(self.context, 'log'):
            self.context.log(level, message)
        else:
            if level == "debug":
                logger.debug(message)
            elif level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
            else:
                logger.info(message)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Current configuration dictionary
        """
        return self.config

    def update_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Update configuration based on human feedback.

        This method updates the configuration based on human feedback regarding
        intention, adjusting weights and thresholds to better align with the
        intended behavior.

        Args:
            feedback: Feedback dictionary containing:
                - intent_name: Name of the intent
                - is_correct: Whether the analysis was correct
                - corrections: Dictionary of corrections
                - confidence: Confidence in the feedback (0.0-1.0)
        """
        try:
            # Validate feedback
            if not isinstance(feedback, dict):
                self.log("error", f"Invalid feedback: {feedback}. Must be a dictionary.")
                return

            if "intent_name" not in feedback:
                self.log("error", f"Invalid feedback: {feedback}. Missing required property 'intent_name'.")
                return

            if "is_correct" not in feedback:
                self.log("error", f"Invalid feedback: {feedback}. Missing required property 'is_correct'.")
                return

            # Add timestamp to feedback
            feedback["timestamp"] = datetime.datetime.now().isoformat()

            # Add to feedback history
            self.feedback_history.append(feedback)

            # Update configuration based on feedback
            intent_name = feedback["intent_name"]
            is_correct = feedback["is_correct"]
            corrections = feedback.get("corrections", {})
            confidence = feedback.get("confidence", 1.0)

            # Adjust pattern weights
            if "patterns" in corrections:
                self._adjust_pattern_weights(intent_name, corrections["patterns"], is_correct, confidence)

            # Adjust analyzer thresholds
            if "thresholds" in corrections:
                self._adjust_analyzer_thresholds(corrections["thresholds"], is_correct, confidence)

            # Adjust intent alignments
            if "intent_alignments" in corrections:
                self._adjust_intent_alignments(corrections["intent_alignments"], is_correct, confidence)

            self.log("info", f"Configuration updated based on feedback for intent: {intent_name}")
        except Exception as e:
            self.log("error", f"Error updating configuration from feedback: {e}")

    def _adjust_pattern_weights(self, intent_name: str, pattern_corrections: Dict[str, Any],
                               is_correct: bool, confidence: float) -> None:
        """
        Adjust pattern weights based on feedback.

        Args:
            intent_name: Name of the intent
            pattern_corrections: Dictionary of pattern corrections
            is_correct: Whether the analysis was correct
            confidence: Confidence in the feedback (0.0-1.0)
        """
        # Get patterns for the intent
        intent_patterns = []
        for intent in self.config.get("intents", []):
            if intent.get("name") == intent_name:
                intent_patterns = intent.get("patterns", [])
                break

        # Adjust weights for patterns
        for pattern_name, correction in pattern_corrections.items():
            # Find the pattern
            for pattern in self.config.get("patterns", []):
                if pattern.get("name") == pattern_name:
                    # Get current weight
                    current_weight = pattern.get("weight", 1.0)

                    # Calculate adjustment
                    if is_correct:
                        # If analysis was correct, increase weight
                        adjustment = self.learning_rate * confidence
                    else:
                        # If analysis was incorrect, decrease weight
                        adjustment = -self.learning_rate * confidence

                    # Apply adjustment
                    new_weight = max(0.1, min(5.0, current_weight + adjustment))
                    pattern["weight"] = new_weight

                    self.log("debug", f"Adjusted weight for pattern {pattern_name}: {current_weight} -> {new_weight}")
                    break

    def _adjust_analyzer_thresholds(self, threshold_corrections: Dict[str, Any],
                                   is_correct: bool, confidence: float) -> None:
        """
        Adjust analyzer thresholds based on feedback.

        Args:
            threshold_corrections: Dictionary of threshold corrections
            is_correct: Whether the analysis was correct
            confidence: Confidence in the feedback (0.0-1.0)
        """
        # Adjust thresholds for analyzers
        for analyzer_name, correction in threshold_corrections.items():
            # Find the analyzer
            if analyzer_name in self.config.get("analyzers", {}):
                analyzer_config = self.config["analyzers"][analyzer_name]

                # Adjust threshold
                if "threshold" in correction:
                    current_threshold = analyzer_config.get("confidence_threshold", 0.6)

                    # Calculate adjustment
                    if is_correct:
                        # If analysis was correct, decrease threshold (more permissive)
                        adjustment = -self.learning_rate * confidence
                    else:
                        # If analysis was incorrect, increase threshold (more strict)
                        adjustment = self.learning_rate * confidence

                    # Apply adjustment
                    new_threshold = max(0.1, min(0.9, current_threshold + adjustment))
                    analyzer_config["confidence_threshold"] = new_threshold

                    self.log("debug", f"Adjusted threshold for analyzer {analyzer_name}: {current_threshold} -> {new_threshold}")

    def _adjust_intent_alignments(self, alignment_corrections: Dict[str, Any],
                                 is_correct: bool, confidence: float) -> None:
        """
        Adjust intent alignments based on feedback.

        Args:
            alignment_corrections: Dictionary of alignment corrections
            is_correct: Whether the analysis was correct
            confidence: Confidence in the feedback (0.0-1.0)
        """
        # Adjust weights for intent alignments
        for intent_name, correction in alignment_corrections.items():
            # Find the intent
            for intent in self.config.get("intents", []):
                if intent.get("name") == intent_name:
                    # Adjust weight
                    current_weight = intent.get("weight", 1.0)

                    # Calculate adjustment
                    if is_correct:
                        # If analysis was correct, increase weight
                        adjustment = self.learning_rate * confidence
                    else:
                        # If analysis was incorrect, decrease weight
                        adjustment = -self.learning_rate * confidence

                    # Apply adjustment
                    new_weight = max(0.1, min(5.0, current_weight + adjustment))
                    intent["weight"] = new_weight

                    self.log("debug", f"Adjusted weight for intent {intent_name}: {current_weight} -> {new_weight}")
                    break

    def save_config(self, file_path: Union[str, Path]) -> bool:
        """
        Save the current configuration to a file.

        Args:
            file_path: Path to save the configuration to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert file path to Path
            file_path = Path(file_path)

            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save configuration
            with open(file_path, "w") as f:
                json.dump(self.config, f, indent=2)

            self.log("info", f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            self.log("error", f"Error saving configuration to {file_path}: {e}")
            return False

    def load_config(self, file_path: Union[str, Path]) -> bool:
        """
        Load configuration from a file.

        Args:
            file_path: Path to load the configuration from

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert file path to Path
            file_path = Path(file_path)

            # Check if file exists
            if not file_path.exists():
                self.log("error", f"Configuration file not found: {file_path}")
                return False

            # Load configuration
            with open(file_path, "r") as f:
                self.config = json.load(f)

            self.log("info", f"Configuration loaded from {file_path}")
            return True
        except Exception as e:
            self.log("error", f"Error loading configuration from {file_path}: {e}")
            return False

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update the configuration with new values.

        Args:
            new_config: New configuration values

        Returns:
            True if successful, False otherwise
        """
        try:
            # Merge new configuration with existing configuration
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        deep_update(d[k], v)
                    else:
                        d[k] = v

            deep_update(self.config, new_config)

            self.log("info", "Configuration updated")
            return True
        except Exception as e:
            self.log("error", f"Error updating configuration: {e}")
            return False
