"""
Workflow module for interactive refactoring.

This module provides functionality for defining and managing refactoring workflows.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a refactoring step."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointStatus(Enum):
    """Status of a refactoring checkpoint."""
    
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass
class RefactoringStep:
    """A step in a refactoring workflow."""
    
    id: str
    title: str
    description: str
    file_paths: List[Path]
    status: StepStatus = StepStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    changes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "file_paths": [str(path) for path in self.file_paths],
            "status": self.status.value,
            "dependencies": self.dependencies,
            "changes": self.changes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefactoringStep':
        """Create from dictionary after deserialization."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            file_paths=[Path(path) for path in data["file_paths"]],
            status=StepStatus(data["status"]),
            dependencies=data.get("dependencies", []),
            changes=data.get("changes", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
    
    def update_status(self, status: StepStatus) -> None:
        """Update the status of the step.
        
        Args:
            status: The new status.
        """
        self.status = status
        self.updated_at = datetime.now()
    
    def can_execute(self, completed_steps: List[str]) -> bool:
        """Check if the step can be executed.
        
        Args:
            completed_steps: A list of completed step IDs.
            
        Returns:
            True if the step can be executed, False otherwise.
        """
        return all(dep in completed_steps for dep in self.dependencies)


@dataclass
class RefactoringCheckpoint:
    """A checkpoint in a refactoring workflow."""
    
    id: str
    title: str
    description: str
    step_ids: List[str]
    status: CheckpointStatus = CheckpointStatus.PENDING
    verification_criteria: List[str] = field(default_factory=list)
    verification_results: Dict[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "step_ids": self.step_ids,
            "status": self.status.value,
            "verification_criteria": self.verification_criteria,
            "verification_results": self.verification_results,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefactoringCheckpoint':
        """Create from dictionary after deserialization."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            step_ids=data["step_ids"],
            status=CheckpointStatus(data["status"]),
            verification_criteria=data.get("verification_criteria", []),
            verification_results=data.get("verification_results", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
    
    def update_status(self, status: CheckpointStatus) -> None:
        """Update the status of the checkpoint.
        
        Args:
            status: The new status.
        """
        self.status = status
        self.updated_at = datetime.now()
    
    def add_verification_result(self, criterion: str, result: bool) -> None:
        """Add a verification result.
        
        Args:
            criterion: The verification criterion.
            result: The verification result.
        """
        self.verification_results[criterion] = result
        self.updated_at = datetime.now()
    
    def is_verified(self) -> bool:
        """Check if the checkpoint is verified.
        
        Returns:
            True if all verification criteria are met, False otherwise.
        """
        return all(self.verification_results.get(criterion, False) for criterion in self.verification_criteria)


@dataclass
class RefactoringWorkflow:
    """A workflow for interactive refactoring."""
    
    id: str
    title: str
    description: str
    steps: Dict[str, RefactoringStep] = field(default_factory=dict)
    checkpoints: Dict[str, RefactoringCheckpoint] = field(default_factory=dict)
    current_step_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "steps": {step_id: step.to_dict() for step_id, step in self.steps.items()},
            "checkpoints": {checkpoint_id: checkpoint.to_dict() for checkpoint_id, checkpoint in self.checkpoints.items()},
            "current_step_id": self.current_step_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefactoringWorkflow':
        """Create from dictionary after deserialization."""
        workflow = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            current_step_id=data.get("current_step_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        # Load steps
        for step_id, step_data in data.get("steps", {}).items():
            workflow.steps[step_id] = RefactoringStep.from_dict(step_data)
        
        # Load checkpoints
        for checkpoint_id, checkpoint_data in data.get("checkpoints", {}).items():
            workflow.checkpoints[checkpoint_id] = RefactoringCheckpoint.from_dict(checkpoint_data)
        
        return workflow
    
    def add_step(self, step: RefactoringStep) -> None:
        """Add a step to the workflow.
        
        Args:
            step: The step to add.
        """
        self.steps[step.id] = step
        self.updated_at = datetime.now()
    
    def add_checkpoint(self, checkpoint: RefactoringCheckpoint) -> None:
        """Add a checkpoint to the workflow.
        
        Args:
            checkpoint: The checkpoint to add.
        """
        self.checkpoints[checkpoint.id] = checkpoint
        self.updated_at = datetime.now()
    
    def get_step(self, step_id: str) -> Optional[RefactoringStep]:
        """Get a step by ID.
        
        Args:
            step_id: The step ID.
            
        Returns:
            The step, or None if not found.
        """
        return self.steps.get(step_id)
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[RefactoringCheckpoint]:
        """Get a checkpoint by ID.
        
        Args:
            checkpoint_id: The checkpoint ID.
            
        Returns:
            The checkpoint, or None if not found.
        """
        return self.checkpoints.get(checkpoint_id)
    
    def get_next_step(self) -> Optional[RefactoringStep]:
        """Get the next step to execute.
        
        Returns:
            The next step, or None if there are no more steps.
        """
        completed_step_ids = [step_id for step_id, step in self.steps.items() if step.status == StepStatus.COMPLETED]
        
        for step in self.steps.values():
            if step.status == StepStatus.PENDING and step.can_execute(completed_step_ids):
                return step
        
        return None
    
    def get_checkpoint_for_step(self, step_id: str) -> Optional[RefactoringCheckpoint]:
        """Get the checkpoint for a step.
        
        Args:
            step_id: The step ID.
            
        Returns:
            The checkpoint, or None if not found.
        """
        for checkpoint in self.checkpoints.values():
            if step_id in checkpoint.step_ids:
                return checkpoint
        
        return None
    
    def update_step_status(self, step_id: str, status: StepStatus) -> None:
        """Update the status of a step.
        
        Args:
            step_id: The step ID.
            status: The new status.
        """
        step = self.get_step(step_id)
        if step:
            step.update_status(status)
            self.updated_at = datetime.now()
    
    def update_checkpoint_status(self, checkpoint_id: str, status: CheckpointStatus) -> None:
        """Update the status of a checkpoint.
        
        Args:
            checkpoint_id: The checkpoint ID.
            status: The new status.
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint:
            checkpoint.update_status(status)
            self.updated_at = datetime.now()
    
    def set_current_step(self, step_id: Optional[str]) -> None:
        """Set the current step.
        
        Args:
            step_id: The step ID, or None to clear the current step.
        """
        self.current_step_id = step_id
        self.updated_at = datetime.now()
    
    def get_progress(self) -> Tuple[int, int]:
        """Get the progress of the workflow.
        
        Returns:
            A tuple of (completed_steps, total_steps).
        """
        completed_steps = sum(1 for step in self.steps.values() if step.status == StepStatus.COMPLETED)
        total_steps = len(self.steps)
        return completed_steps, total_steps
    
    def save(self, file_path: Path) -> None:
        """Save the workflow to a file.
        
        Args:
            file_path: The path to save to.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: Path) -> 'RefactoringWorkflow':
        """Load a workflow from a file.
        
        Args:
            file_path: The path to load from.
            
        Returns:
            The loaded workflow.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
