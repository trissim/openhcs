"""
Tracker module for interactive refactoring.

This module provides functionality for tracking progress in refactoring workflows.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.refactoring.workflow import (
    RefactoringWorkflow, RefactoringStep, RefactoringCheckpoint,
    StepStatus, CheckpointStatus
)

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    """An event in a refactoring workflow."""
    
    id: str
    workflow_id: str
    event_type: str  # "step_status_changed", "checkpoint_status_changed", etc.
    timestamp: datetime
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressEvent':
        """Create from dictionary after deserialization."""
        return cls(
            id=data["id"],
            workflow_id=data["workflow_id"],
            event_type=data["event_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            details=data.get("details", {})
        )


class ProgressTracker:
    """Tracks progress in refactoring workflows."""
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.events: Dict[str, List[ProgressEvent]] = {}
    
    def add_event(self, event: ProgressEvent) -> None:
        """Add an event to the tracker.
        
        Args:
            event: The event to add.
        """
        if event.workflow_id not in self.events:
            self.events[event.workflow_id] = []
        
        self.events[event.workflow_id].append(event)
    
    def get_events(self, workflow_id: str) -> List[ProgressEvent]:
        """Get all events for a workflow.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            A list of events for the workflow.
        """
        return self.events.get(workflow_id, [])
    
    def get_events_by_type(self, workflow_id: str, event_type: str) -> List[ProgressEvent]:
        """Get events of a specific type for a workflow.
        
        Args:
            workflow_id: The workflow ID.
            event_type: The event type.
            
        Returns:
            A list of events of the specified type for the workflow.
        """
        return [event for event in self.get_events(workflow_id) if event.event_type == event_type]
    
    def get_step_events(self, workflow_id: str, step_id: str) -> List[ProgressEvent]:
        """Get events for a specific step in a workflow.
        
        Args:
            workflow_id: The workflow ID.
            step_id: The step ID.
            
        Returns:
            A list of events for the step.
        """
        return [
            event for event in self.get_events(workflow_id)
            if event.event_type == "step_status_changed" and event.details.get("step_id") == step_id
        ]
    
    def get_checkpoint_events(self, workflow_id: str, checkpoint_id: str) -> List[ProgressEvent]:
        """Get events for a specific checkpoint in a workflow.
        
        Args:
            workflow_id: The workflow ID.
            checkpoint_id: The checkpoint ID.
            
        Returns:
            A list of events for the checkpoint.
        """
        return [
            event for event in self.get_events(workflow_id)
            if event.event_type == "checkpoint_status_changed" and event.details.get("checkpoint_id") == checkpoint_id
        ]
    
    def get_step_status_history(self, workflow_id: str, step_id: str) -> List[Tuple[datetime, StepStatus]]:
        """Get the status history for a step.
        
        Args:
            workflow_id: The workflow ID.
            step_id: The step ID.
            
        Returns:
            A list of (timestamp, status) tuples for the step.
        """
        events = self.get_step_events(workflow_id, step_id)
        return [(event.timestamp, StepStatus(event.details["status"])) for event in events]
    
    def get_checkpoint_status_history(self, workflow_id: str, checkpoint_id: str) -> List[Tuple[datetime, CheckpointStatus]]:
        """Get the status history for a checkpoint.
        
        Args:
            workflow_id: The workflow ID.
            checkpoint_id: The checkpoint ID.
            
        Returns:
            A list of (timestamp, status) tuples for the checkpoint.
        """
        events = self.get_checkpoint_events(workflow_id, checkpoint_id)
        return [(event.timestamp, CheckpointStatus(event.details["status"])) for event in events]
    
    def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get the progress of a workflow.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            A dictionary with progress information.
        """
        events = self.get_events(workflow_id)
        
        # Get the latest status for each step
        step_statuses = {}
        for event in events:
            if event.event_type == "step_status_changed":
                step_id = event.details.get("step_id")
                status = event.details.get("status")
                if step_id and status:
                    step_statuses[step_id] = StepStatus(status)
        
        # Get the latest status for each checkpoint
        checkpoint_statuses = {}
        for event in events:
            if event.event_type == "checkpoint_status_changed":
                checkpoint_id = event.details.get("checkpoint_id")
                status = event.details.get("status")
                if checkpoint_id and status:
                    checkpoint_statuses[checkpoint_id] = CheckpointStatus(status)
        
        # Calculate progress statistics
        total_steps = len(step_statuses)
        completed_steps = sum(1 for status in step_statuses.values() if status == StepStatus.COMPLETED)
        failed_steps = sum(1 for status in step_statuses.values() if status == StepStatus.FAILED)
        skipped_steps = sum(1 for status in step_statuses.values() if status == StepStatus.SKIPPED)
        
        total_checkpoints = len(checkpoint_statuses)
        verified_checkpoints = sum(1 for status in checkpoint_statuses.values() if status == CheckpointStatus.VERIFIED)
        rejected_checkpoints = sum(1 for status in checkpoint_statuses.values() if status == CheckpointStatus.REJECTED)
        
        # Calculate completion percentage
        completion_percentage = 0
        if total_steps > 0:
            completion_percentage = (completed_steps / total_steps) * 100
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "skipped_steps": skipped_steps,
            "total_checkpoints": total_checkpoints,
            "verified_checkpoints": verified_checkpoints,
            "rejected_checkpoints": rejected_checkpoints,
            "completion_percentage": completion_percentage
        }
    
    def get_workflow_timeline(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get a timeline of events for a workflow.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            A list of event dictionaries, sorted by timestamp.
        """
        events = self.get_events(workflow_id)
        
        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        # Convert events to dictionaries
        timeline = []
        for event in events:
            timeline_event = {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type
            }
            
            # Add event-specific details
            if event.event_type == "step_status_changed":
                step_id = event.details.get("step_id")
                status = event.details.get("status")
                timeline_event["step_id"] = step_id
                timeline_event["status"] = status
            elif event.event_type == "checkpoint_status_changed":
                checkpoint_id = event.details.get("checkpoint_id")
                status = event.details.get("status")
                timeline_event["checkpoint_id"] = checkpoint_id
                timeline_event["status"] = status
            
            timeline.append(timeline_event)
        
        return timeline
