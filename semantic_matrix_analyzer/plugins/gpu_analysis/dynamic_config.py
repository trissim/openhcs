"""
Dynamic Configuration System for GPU Analysis Plugin.

This module provides a dynamic configuration system that enables adaptive learning
from human feedback, pattern recognition, and intent extraction. It extends the
basic configuration integration with observer pattern, version tracking, and
learning rate management.
"""

from typing import Any, Dict, List, Optional, Callable, Set, Tuple, Union
from pathlib import Path
import copy
import time
import math
import logging

from gpu_analysis.config_integration import get_gpu_config_from_sma, validate_config

# Configure logging
logger = logging.getLogger(__name__)


class ConfigObserver:
    """Interface for objects that need to be notified of configuration changes."""

    def on_config_changed(self, config: Dict[str, Any], changed_keys: Set[str], source: str) -> None:
        """
        Called when configuration changes.

        Args:
            config: The new configuration
            changed_keys: Set of keys that were changed
            source: Source of the change (e.g., "user", "system", "feedback")
        """
        pass


class DynamicConfigManager:
    """
    Dynamic configuration manager with observer pattern and version tracking.

    This class extends the basic configuration integration with:
    1. Observer pattern for notifying components of changes
    2. Version tracking for configuration changes
    3. Change history for auditing and rollback
    4. Source attribution for changes
    """

    def __init__(self, sma_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dynamic configuration manager.

        Args:
            sma_config: SMA configuration dictionary
        """
        self.sma_config = sma_config or {}
        self.config = get_gpu_config_from_sma(self.sma_config)
        self.observers: List[ConfigObserver] = []
        self.version = 0
        self.history: List[Dict[str, Any]] = []
        self.max_history = 100  # Maximum number of history entries to keep

    def register_observer(self, observer: ConfigObserver) -> None:
        """
        Register an observer to be notified of configuration changes.

        Args:
            observer: Object implementing the ConfigObserver interface
        """
        if observer not in self.observers:
            self.observers.append(observer)

    def unregister_observer(self, observer: ConfigObserver) -> None:
        """
        Unregister an observer.

        Args:
            observer: Observer to unregister
        """
        if observer in self.observers:
            self.observers.remove(observer)

    def notify_observers(self, changed_keys: Set[str], source: str) -> None:
        """
        Notify all observers of configuration changes.

        Args:
            changed_keys: Set of keys that were changed
            source: Source of the change
        """
        config = self.get_config()
        for observer in self.observers:
            observer.on_config_changed(config, changed_keys, source)

    def update_config(self, updates: Dict[str, Any], source: str = "system") -> None:
        """
        Update configuration with new values and notify observers.

        Args:
            updates: Dictionary of updates to apply
            source: Source of the update (e.g., "user", "system", "feedback")
        """
        # Track which keys are being updated
        changed_keys = set()
        self._track_updates(updates, changed_keys)

        # Save current config to history before updating
        self._add_to_history(source, changed_keys)

        # Apply updates recursively
        self._recursive_update(self.config, updates)

        # Increment version
        self.version += 1

        # Notify observers
        self.notify_observers(changed_keys, source)

    def _recursive_update(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionaries.

        Args:
            target: Target dictionary to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._recursive_update(target[key], value)
            else:
                target[key] = value

    def _track_updates(self, updates: Dict[str, Any], changed_keys: Set[str], prefix: str = "") -> None:
        """
        Track which keys are being updated.

        Args:
            updates: Dictionary of updates
            changed_keys: Set to collect changed keys
            prefix: Prefix for nested keys
        """
        for key, value in updates.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._track_updates(value, changed_keys, full_key)
            else:
                changed_keys.add(full_key)

    def _add_to_history(self, source: str, changed_keys: Set[str]) -> None:
        """
        Add current configuration to history.

        Args:
            source: Source of the change
            changed_keys: Keys that were changed
        """
        # Create history entry
        entry = {
            "version": self.version,
            "timestamp": time.time(),
            "source": source,
            "changed_keys": list(changed_keys),
            "config": copy.deepcopy(self.config)
        }

        # Add to history
        self.history.append(entry)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Current configuration dictionary
        """
        return self.config

    def get_version(self) -> int:
        """
        Get the current configuration version.

        Returns:
            Current version number
        """
        return self.version

    def rollback(self, version: Optional[int] = None) -> bool:
        """
        Rollback to a previous configuration version.

        Args:
            version: Version to rollback to, or None for previous version

        Returns:
            True if rollback was successful, False otherwise
        """
        if not self.history:
            return False

        # Determine target version
        target_version = version if version is not None else self.version - 1

        # Find entry with target version
        for entry in reversed(self.history):
            if entry["version"] == target_version:
                # Apply configuration
                self.config = copy.deepcopy(entry["config"])

                # Update version
                self.version = target_version

                # Notify observers
                self.notify_observers(set(entry["changed_keys"]), "rollback")

                return True

        return False

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the current configuration.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return validate_config(self.config)


class LearningRateManager:
    """
    Manages learning rate parameters for adaptive configuration.

    This class provides:
    1. Parameters to control adaptation rate
    2. Algorithms to adjust parameters based on feedback
    3. Context-aware adaptation rate calculation
    """

    def __init__(self, config_manager: DynamicConfigManager):
        """
        Initialize the learning rate manager.

        Args:
            config_manager: Dynamic configuration manager
        """
        self.config_manager = config_manager

        # Initialize default learning parameters if not present
        config = config_manager.get_config()
        if "learning" not in config:
            learning_config = {
                "learning": {
                    "base_adaptation_rate": 0.1,
                    "confidence_threshold": 0.7,
                    "feedback_weight": 0.8,
                    "stability_factor": 0.5,
                    "min_adaptation_rate": 0.01,
                    "max_adaptation_rate": 0.5,
                    "decay_factor": 0.95,
                    "success_count": 0,
                    "failure_count": 0
                }
            }
            config_manager.update_config(learning_config, "system")

    def get_adaptation_rate(self, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get the current adaptation rate, possibly adjusted for context.

        Args:
            context: Optional context information

        Returns:
            Adaptation rate (0.0 to 1.0)
        """
        config = self.config_manager.get_config()
        base_rate = config["learning"]["base_adaptation_rate"]

        # If no context, return base rate
        if not context:
            return base_rate

        # Adjust based on context
        confidence = context.get("confidence", 1.0)
        importance = context.get("importance", 1.0)

        # Higher confidence and importance increase adaptation rate
        adjusted_rate = base_rate * confidence * importance

        # Ensure within bounds
        min_rate = config["learning"]["min_adaptation_rate"]
        max_rate = config["learning"]["max_adaptation_rate"]

        return max(min_rate, min(adjusted_rate, max_rate))

    def update_from_feedback(self, success: bool, confidence: float = 1.0) -> None:
        """
        Update learning parameters based on feedback success.

        Args:
            success: Whether the adaptation was successful
            confidence: Confidence in the feedback (0.0 to 1.0)
        """
        config = self.config_manager.get_config()
        learning = config["learning"]

        # Update success/failure counts
        if success:
            learning["success_count"] += 1
        else:
            learning["failure_count"] += 1

        # Calculate success ratio
        total = learning["success_count"] + learning["failure_count"]
        success_ratio = learning["success_count"] / total if total > 0 else 0.5

        # Adjust adaptation rate based on success ratio
        if success:
            # Increase adaptation rate if successful
            new_rate = learning["base_adaptation_rate"] * (1.0 + (success_ratio - 0.5) * confidence)
        else:
            # Decrease adaptation rate if unsuccessful
            new_rate = learning["base_adaptation_rate"] * learning["decay_factor"]

        # Ensure within bounds
        min_rate = learning["min_adaptation_rate"]
        max_rate = learning["max_adaptation_rate"]
        new_rate = max(min_rate, min(new_rate, max_rate))

        # Update configuration
        updates = {
            "learning": {
                "base_adaptation_rate": new_rate,
                "success_count": learning["success_count"],
                "failure_count": learning["failure_count"]
            }
        }

        self.config_manager.update_config(updates, "feedback")
