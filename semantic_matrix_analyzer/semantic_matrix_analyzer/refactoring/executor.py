"""
Executor module for interactive refactoring.

This module provides functionality for executing refactoring steps.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

from semantic_matrix_analyzer.refactoring.workflow import (
    RefactoringWorkflow, RefactoringStep, RefactoringCheckpoint,
    StepStatus, CheckpointStatus
)
from semantic_matrix_analyzer.refactoring.manager import RefactoringManager

logger = logging.getLogger(__name__)


class RefactoringExecutor:
    """Executes refactoring steps."""

    def __init__(self, manager: RefactoringManager):
        """Initialize the refactoring executor.

        Args:
            manager: The refactoring manager.
        """
        self.manager = manager
        self.handlers: Dict[str, Callable[[RefactoringStep], bool]] = {}

    def register_handler(self, step_type: str, handler: Callable[[RefactoringStep], bool]) -> None:
        """Register a handler for a step type.

        Args:
            step_type: The step type.
            handler: The handler function.
        """
        self.handlers[step_type] = handler

    def execute_step(self, workflow_id: str, step_id: str) -> bool:
        """Execute a refactoring step.

        Args:
            workflow_id: The workflow ID.
            step_id: The step ID.

        Returns:
            True if the step was executed successfully, False otherwise.
        """
        workflow = self.manager.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return False

        step = workflow.get_step(step_id)
        if not step:
            logger.error(f"Step not found: {step_id}")
            return False

        # Check if the step can be executed
        completed_step_ids = [s_id for s_id, s in workflow.steps.items() if s.status == StepStatus.COMPLETED]
        if not step.can_execute(completed_step_ids):
            logger.error(f"Step cannot be executed: {step_id} (dependencies not met)")
            return False

        # Update the step status
        self.manager.update_step_status(workflow_id, step_id, StepStatus.IN_PROGRESS)

        # Create backups of the files
        backups = {}
        for file_path in step.file_paths:
            if file_path.exists():
                backup_path = self.manager.create_backup(file_path)
                backups[str(file_path)] = backup_path

        # Store the backups in the step changes
        step.changes["backups"] = {str(file_path): str(backup_path) for file_path, backup_path in backups.items()}

        # Execute the step
        try:
            # Get the step type
            step_type = step.changes.get("type", "generic")

            # Execute the step using the appropriate handler
            handler = self.handlers.get(step_type)
            if handler:
                success = handler(step)
            else:
                # Default implementation
                success = self._execute_generic_step(step)

            # Update the step status
            if success:
                self.manager.update_step_status(workflow_id, step_id, StepStatus.COMPLETED)
            else:
                self.manager.update_step_status(workflow_id, step_id, StepStatus.FAILED)

            return success
        except Exception as e:
            logger.error(f"Error executing step {step_id}: {e}")
            self.manager.update_step_status(workflow_id, step_id, StepStatus.FAILED)
            return False

    def _execute_generic_step(self, step: RefactoringStep) -> bool:
        """Execute a generic refactoring step.

        Args:
            step: The step to execute.

        Returns:
            True if the step was executed successfully, False otherwise.
        """
        # This is a placeholder implementation
        # In a real implementation, this would perform the actual refactoring

        # Check if the step has code changes
        if "code_changes" not in step.changes:
            logger.error(f"Step {step.id} has no code changes")
            return False

        code_changes = step.changes["code_changes"]

        # Apply the code changes
        for file_path_str, changes in code_changes.items():
            file_path = Path(file_path_str)

            # Read the file
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Apply the changes
            for change in changes:
                change_type = change.get("type")

                if change_type == "replace":
                    # Replace a range of lines
                    start_line = change.get("start_line", 1) - 1  # 0-based index
                    end_line = change.get("end_line", 1) - 1  # 0-based index
                    old_text = change.get("old_text", "")
                    new_text = change.get("new_text", "")

                    lines = content.splitlines()

                    # Check if the old text matches
                    old_lines = lines[start_line:end_line + 1]
                    actual_text = "\n".join(old_lines)

                    # Normalize line endings for comparison
                    normalized_actual = actual_text.replace("\r\n", "\n").strip()
                    normalized_expected = old_text.replace("\r\n", "\n").strip()

                    if normalized_actual != normalized_expected:
                        logger.error(f"Old text does not match in file {file_path}")
                        logger.error(f"Expected: {repr(normalized_expected)}")
                        logger.error(f"Actual: {repr(normalized_actual)}")

                        # For debugging, show the first difference
                        for i, (a, b) in enumerate(zip(normalized_actual, normalized_expected)):
                            if a != b:
                                logger.error(f"First difference at position {i}: '{a}' vs '{b}'")
                                break

                        return False

                    # Replace the lines
                    new_lines = new_text.splitlines()
                    lines[start_line:end_line + 1] = new_lines

                    # Update the content
                    content = "\n".join(lines)

                elif change_type == "insert":
                    # Insert text at a specific line
                    line = change.get("line", 1) - 1  # 0-based index
                    text = change.get("text", "")

                    lines = content.splitlines()

                    # Insert the text
                    new_lines = text.splitlines()
                    lines[line:line] = new_lines

                    # Update the content
                    content = "\n".join(lines)

                elif change_type == "delete":
                    # Delete a range of lines
                    start_line = change.get("start_line", 1) - 1  # 0-based index
                    end_line = change.get("end_line", 1) - 1  # 0-based index

                    lines = content.splitlines()

                    # Delete the lines
                    lines[start_line:end_line + 1] = []

                    # Update the content
                    content = "\n".join(lines)

            # Write the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        return True

    def rollback_step(self, workflow_id: str, step_id: str) -> bool:
        """Roll back a refactoring step.

        Args:
            workflow_id: The workflow ID.
            step_id: The step ID.

        Returns:
            True if the step was rolled back successfully, False otherwise.
        """
        workflow = self.manager.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return False

        step = workflow.get_step(step_id)
        if not step:
            logger.error(f"Step not found: {step_id}")
            return False

        # Check if the step has backups
        if "backups" not in step.changes:
            logger.error(f"Step {step_id} has no backups")
            return False

        backups = step.changes["backups"]

        # Restore the backups
        for file_path_str, backup_path_str in backups.items():
            file_path = Path(file_path_str)
            backup_path = Path(backup_path_str)

            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Restore the backup
            self.manager.restore_backup(backup_path, file_path)

        # Update the step status
        self.manager.update_step_status(workflow_id, step_id, StepStatus.PENDING)

        return True

    def verify_checkpoint(self, workflow_id: str, checkpoint_id: str) -> bool:
        """Verify a refactoring checkpoint.

        Args:
            workflow_id: The workflow ID.
            checkpoint_id: The checkpoint ID.

        Returns:
            True if the checkpoint was verified successfully, False otherwise.
        """
        workflow = self.manager.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return False

        checkpoint = workflow.get_checkpoint(checkpoint_id)
        if not checkpoint:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False

        # Check if all steps in the checkpoint are completed
        for step_id in checkpoint.step_ids:
            step = workflow.get_step(step_id)
            if not step or step.status != StepStatus.COMPLETED:
                logger.error(f"Step {step_id} is not completed")
                return False

        # Verify the checkpoint
        if checkpoint.verification_criteria:
            # Check if all verification criteria are met
            if not checkpoint.is_verified():
                logger.error(f"Checkpoint {checkpoint_id} is not verified")
                return False

        # Update the checkpoint status
        self.manager.update_checkpoint_status(workflow_id, checkpoint_id, CheckpointStatus.VERIFIED)

        return True

    def reject_checkpoint(self, workflow_id: str, checkpoint_id: str) -> bool:
        """Reject a refactoring checkpoint.

        Args:
            workflow_id: The workflow ID.
            checkpoint_id: The checkpoint ID.

        Returns:
            True if the checkpoint was rejected successfully, False otherwise.
        """
        workflow = self.manager.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return False

        checkpoint = workflow.get_checkpoint(checkpoint_id)
        if not checkpoint:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False

        # Update the checkpoint status
        self.manager.update_checkpoint_status(workflow_id, checkpoint_id, CheckpointStatus.REJECTED)

        # Roll back all steps in the checkpoint
        for step_id in checkpoint.step_ids:
            self.rollback_step(workflow_id, step_id)

        return True
