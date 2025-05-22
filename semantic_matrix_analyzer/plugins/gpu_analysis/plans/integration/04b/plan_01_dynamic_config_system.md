# plan_01_dynamic_config_system.md
## Component: Brain Agent Dynamic Configuration System

### Objective
Create a dynamic configuration system for the brain agent that enables adaptive learning from human feedback, pattern recognition, and intent extraction. This system will allow the brain agent to continuously improve its performance based on user interactions and feedback.

### Plan
1. Implement a DynamicConfigManager class that extends the existing ConfigManager
2. Create a FeedbackProcessor to handle and learn from human feedback
3. Develop PatternExtractor and IntentExtractor components
4. Implement LearningRateManager to control adaptation parameters
5. Integrate all components into a cohesive system
6. Connect the system to the pipeline compiler for real-time adaptation

### Findings
The current codebase has a ConfigManager class in `semantic_matrix_analyzer/config_manager.py` that handles basic configuration management, but lacks dynamic update capabilities, observer pattern implementation, and learning rate parameters. The brain agent requires these components to implement a dynamic configuration feedback system.

Key components needed:
- DynamicConfigManager: For managing configuration updates
- FeedbackProcessor: For processing human feedback
- PatternExtractor and IntentExtractor: For pattern recognition and intent extraction
- LearningRateManager: For controlling adaptation rates

### Implementation Draft

#### 1. DynamicConfigManager Class

```python
"""
Dynamic Configuration Manager for adaptive brain agent.

This module extends the basic ConfigManager with dynamic update capabilities,
observer pattern implementation, and version tracking.
"""

from typing import Any, Dict, List, Optional, Callable, Set, Union
from pathlib import Path
import copy
import time

from semantic_matrix_analyzer.config_manager import ConfigManager


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
    
    This class extends the basic ConfigManager with:
    1. Observer pattern for notifying components of changes
    2. Version tracking for configuration changes
    3. Change history for auditing and rollback
    4. Source attribution for changes
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the dynamic configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_manager = ConfigManager(config_path)
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
        
        # Apply updates
        self.config_manager.update_config(updates)
        
        # Increment version
        self.version += 1
        
        # Notify observers
        self.notify_observers(changed_keys, source)
        
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
            "config": copy.deepcopy(self.config_manager.get_config())
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
        return self.config_manager.get_config()
        
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
                self.config_manager.save_config(entry["config"])
                
                # Update version
                self.version = target_version
                
                # Notify observers
                self.notify_observers(set(entry["changed_keys"]), "rollback")
                
                return True
                
        return False
```

#### 2. Learning Rate Parameters

```python
"""
Learning Rate Manager for adaptive brain agent.

This module provides parameters and algorithms for controlling
the rate of adaptation based on feedback and context.
"""

from typing import Any, Dict, Optional
import math


class LearningRateManager:
    """
    Manages learning rate parameters for adaptive configuration.
    
    This class provides:
    1. Parameters to control adaptation rate
    2. Algorithms to adjust parameters based on feedback
    3. Context-aware adaptation rate calculation
    """
    
    def __init__(self, config_manager):
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
```

The remaining components (FeedbackProcessor, PatternExtractor, and IntentExtractor) will be implemented in subsequent plan files to keep this plan focused and manageable.
