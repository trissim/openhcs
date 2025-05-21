"""
Memory tracker registry for OpenHCS.

This module provides a registry for memory trackers, using a schema-based approach
to load and validate memory trackers.

Doctrinal Clauses:
- Clause 88 (No Inferred Capabilities): Capabilities are explicitly declared in schemas
- Clause 92 (Interface Fraud): Memory trackers are validated against their interfaces
- Clause 245 (Declarative Enforcement): Behavior follows declared schema
- Clause 510 (Schema Doctrine): All memory tracker declarations are defined in TOML schema files
- Clause 521 (Fixes Must Eliminate Their Source): Eliminates dynamic registration mechanisms
"""

import logging
from typing import Any, List, Type

from openhcs.core.memory.loaders.memory_tracker_loader import (
    MemoryTrackerLoader, MemoryTrackerNotFoundError)

logger = logging.getLogger(__name__)


class MemoryTrackerRegistry:
    """
    Registry for memory trackers.
    
    This class provides a registry for memory trackers, using a schema-based approach
    to load and validate memory trackers.
    
    Attributes:
        schema_path: Path to the schema file
    """
    
    def __init__(self, schema_path: str = "schemas/memory_tracker_registry.toml"):
        """
        Initialize the memory tracker registry.
        
        Args:
            schema_path: Path to the schema file
        """
        self.loader = MemoryTrackerLoader(schema_path)
        self._trackers = {}
        
    def get_tracker(self, name: str) -> Any:
        """
        Get a memory tracker by name.
        
        Args:
            name: The memory tracker name
            
        Returns:
            Memory tracker instance
            
        Raises:
            MemoryTrackerNotFoundError: If the memory tracker is not found
        """
        if not self._trackers:
            self._trackers = self.loader.load_trackers()
        
        if name not in self._trackers:
            available = list(self._trackers.keys())
            raise MemoryTrackerNotFoundError(f"Memory tracker '{name}' not found. Available: {available}")
        
        factory = self._trackers[name]
        return factory()
    
    def list_trackers(self) -> List[str]:
        """
        List all available memory tracker names.
        
        Returns:
            List of memory tracker names
        """
        if not self._trackers:
            self._trackers = self.loader.load_trackers()
        
        return list(self._trackers.keys())


# Create a global instance of the registry
memory_tracker_registry = MemoryTrackerRegistry()


def get_tracker(name: str) -> Any:
    """
    Get a memory tracker by name.
    
    Args:
        name: The memory tracker name
        
    Returns:
        Memory tracker instance
        
    Raises:
        MemoryTrackerNotFoundError: If the memory tracker is not found
    """
    return memory_tracker_registry.get_tracker(name)


def list_trackers() -> List[str]:
    """
    List all available memory tracker names.
    
    Returns:
        List of memory tracker names
    """
    return memory_tracker_registry.list_trackers()


class MemoryTrackerSpec:
    """
    Specification for a memory tracker.
    
    This class provides a specification for a memory tracker, including
    its name, class, and capabilities.
    
    Attributes:
        name: The memory tracker name
        tracker_cls: The memory tracker class
        accurate: Whether the tracker provides accurate memory information
        synchronous: Whether the tracker's memory calls are synchronous
    """
    
    def __init__(self, name: str, tracker_cls: Type, accurate: bool, synchronous: bool):
        """
        Initialize the memory tracker specification.
        
        Args:
            name: The memory tracker name
            tracker_cls: The memory tracker class
            accurate: Whether the tracker provides accurate memory information
            synchronous: Whether the tracker's memory calls are synchronous
        """
        self.name = name
        self.tracker_cls = tracker_cls
        self.accurate = accurate
        self.synchronous = synchronous


def list_available_tracker_specs(include_sync_true_only: bool = False) -> List[MemoryTrackerSpec]:
    """
    List specifications for all registered and available memory trackers.
    
    Args:
        include_sync_true_only: If True, only include trackers marked as synchronous.
                             Default is False (include all).
    
    Returns:
        A list of MemoryTrackerSpec objects.
    """
    specs = []
    for name in list_trackers():
        try:
            tracker = get_tracker(name)
            spec = MemoryTrackerSpec(
                name=name,
                tracker_cls=tracker.__class__,
                accurate=getattr(tracker, "accurate", False),
                synchronous=getattr(tracker, "synchronous", False)
            )
            if include_sync_true_only and not spec.synchronous:
                continue
            specs.append(spec)
        except Exception as e:
            logger.error(f"Error creating spec for tracker {name}: {e}")
    return specs
