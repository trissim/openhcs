"""
Manager module for interactive refactoring.

This module provides functionality for managing refactoring workflows.
"""

import json
import logging
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.refactoring.workflow import (
    RefactoringWorkflow, RefactoringStep, RefactoringCheckpoint,
    StepStatus, CheckpointStatus
)

logger = logging.getLogger(__name__)


class RefactoringManager:
    """Manages refactoring workflows."""
    
    def __init__(self, workspace_dir: Path):
        """Initialize the refactoring manager.
        
        Args:
            workspace_dir: The directory to store workflows and backups.
        """
        self.workspace_dir = workspace_dir
        self.workflows_dir = workspace_dir / "workflows"
        self.backups_dir = workspace_dir / "backups"
        
        # Create directories if they don't exist
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing workflows
        self.workflows: Dict[str, RefactoringWorkflow] = {}
        self._load_workflows()
    
    def _load_workflows(self) -> None:
        """Load existing workflows from the workspace directory."""
        for file_path in self.workflows_dir.glob("*.json"):
            try:
                workflow = RefactoringWorkflow.load(file_path)
                self.workflows[workflow.id] = workflow
            except Exception as e:
                logger.error(f"Error loading workflow from {file_path}: {e}")
    
    def create_workflow(self, title: str, description: str) -> RefactoringWorkflow:
        """Create a new refactoring workflow.
        
        Args:
            title: The workflow title.
            description: The workflow description.
            
        Returns:
            The created workflow.
        """
        workflow_id = str(uuid.uuid4())
        workflow = RefactoringWorkflow(
            id=workflow_id,
            title=title,
            description=description
        )
        
        # Save the workflow
        self.workflows[workflow_id] = workflow
        self._save_workflow(workflow)
        
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[RefactoringWorkflow]:
        """Get a workflow by ID.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            The workflow, or None if not found.
        """
        return self.workflows.get(workflow_id)
    
    def get_workflows(self) -> List[RefactoringWorkflow]:
        """Get all workflows.
        
        Returns:
            A list of all workflows.
        """
        return list(self.workflows.values())
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            True if the workflow was deleted, False otherwise.
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        # Remove the workflow file
        workflow_file = self.workflows_dir / f"{workflow_id}.json"
        if workflow_file.exists():
            workflow_file.unlink()
        
        # Remove the workflow from memory
        del self.workflows[workflow_id]
        
        return True
    
    def add_step(
        self,
        workflow_id: str,
        title: str,
        description: str,
        file_paths: List[Path],
        dependencies: Optional[List[str]] = None
    ) -> Optional[RefactoringStep]:
        """Add a step to a workflow.
        
        Args:
            workflow_id: The workflow ID.
            title: The step title.
            description: The step description.
            file_paths: The file paths affected by the step.
            dependencies: The step dependencies (optional).
            
        Returns:
            The created step, or None if the workflow was not found.
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        step_id = str(uuid.uuid4())
        step = RefactoringStep(
            id=step_id,
            title=title,
            description=description,
            file_paths=file_paths,
            dependencies=dependencies or []
        )
        
        workflow.add_step(step)
        self._save_workflow(workflow)
        
        return step
    
    def add_checkpoint(
        self,
        workflow_id: str,
        title: str,
        description: str,
        step_ids: List[str],
        verification_criteria: Optional[List[str]] = None
    ) -> Optional[RefactoringCheckpoint]:
        """Add a checkpoint to a workflow.
        
        Args:
            workflow_id: The workflow ID.
            title: The checkpoint title.
            description: The checkpoint description.
            step_ids: The step IDs to include in the checkpoint.
            verification_criteria: The verification criteria (optional).
            
        Returns:
            The created checkpoint, or None if the workflow was not found.
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        checkpoint_id = str(uuid.uuid4())
        checkpoint = RefactoringCheckpoint(
            id=checkpoint_id,
            title=title,
            description=description,
            step_ids=step_ids,
            verification_criteria=verification_criteria or []
        )
        
        workflow.add_checkpoint(checkpoint)
        self._save_workflow(workflow)
        
        return checkpoint
    
    def update_step_status(self, workflow_id: str, step_id: str, status: StepStatus) -> bool:
        """Update the status of a step.
        
        Args:
            workflow_id: The workflow ID.
            step_id: The step ID.
            status: The new status.
            
        Returns:
            True if the step was updated, False otherwise.
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        workflow.update_step_status(step_id, status)
        self._save_workflow(workflow)
        
        return True
    
    def update_checkpoint_status(self, workflow_id: str, checkpoint_id: str, status: CheckpointStatus) -> bool:
        """Update the status of a checkpoint.
        
        Args:
            workflow_id: The workflow ID.
            checkpoint_id: The checkpoint ID.
            status: The new status.
            
        Returns:
            True if the checkpoint was updated, False otherwise.
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        workflow.update_checkpoint_status(checkpoint_id, status)
        self._save_workflow(workflow)
        
        return True
    
    def add_verification_result(self, workflow_id: str, checkpoint_id: str, criterion: str, result: bool) -> bool:
        """Add a verification result to a checkpoint.
        
        Args:
            workflow_id: The workflow ID.
            checkpoint_id: The checkpoint ID.
            criterion: The verification criterion.
            result: The verification result.
            
        Returns:
            True if the result was added, False otherwise.
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        checkpoint = workflow.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return False
        
        checkpoint.add_verification_result(criterion, result)
        self._save_workflow(workflow)
        
        return True
    
    def set_current_step(self, workflow_id: str, step_id: Optional[str]) -> bool:
        """Set the current step of a workflow.
        
        Args:
            workflow_id: The workflow ID.
            step_id: The step ID, or None to clear the current step.
            
        Returns:
            True if the current step was set, False otherwise.
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        workflow.set_current_step(step_id)
        self._save_workflow(workflow)
        
        return True
    
    def create_backup(self, file_path: Path) -> Path:
        """Create a backup of a file.
        
        Args:
            file_path: The file to back up.
            
        Returns:
            The path to the backup file.
        """
        # Create a backup file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backups_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        
        # Copy the file to the backup directory
        shutil.copy2(file_path, backup_file)
        
        return backup_file
    
    def restore_backup(self, backup_file: Path, target_file: Path) -> bool:
        """Restore a backup file.
        
        Args:
            backup_file: The backup file to restore.
            target_file: The target file to restore to.
            
        Returns:
            True if the backup was restored, False otherwise.
        """
        if not backup_file.exists():
            return False
        
        # Create a backup of the current file
        if target_file.exists():
            self.create_backup(target_file)
        
        # Copy the backup file to the target file
        shutil.copy2(backup_file, target_file)
        
        return True
    
    def _save_workflow(self, workflow: RefactoringWorkflow) -> None:
        """Save a workflow to a file.
        
        Args:
            workflow: The workflow to save.
        """
        workflow_file = self.workflows_dir / f"{workflow.id}.json"
        workflow.save(workflow_file)
