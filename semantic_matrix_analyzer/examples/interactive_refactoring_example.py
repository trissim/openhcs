#!/usr/bin/env python3
"""
Example of using the interactive refactoring workflow system.

This script demonstrates how to use the interactive refactoring workflow system to break down
large changes into manageable steps, track progress, provide checkpoints for verification,
and support rollbacks if needed.
"""

import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_matrix_analyzer.refactoring import (
    RefactoringWorkflow, RefactoringStep, RefactoringCheckpoint,
    StepStatus, CheckpointStatus, RefactoringManager, RefactoringExecutor,
    ProgressTracker, ProgressEvent
)


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_sample_files(workspace_dir):
    """Create sample files for testing.

    Args:
        workspace_dir: The workspace directory.

    Returns:
        A list of file paths.
    """
    # Create a directory for the sample files
    sample_dir = workspace_dir / "sample"
    sample_dir.mkdir(exist_ok=True)

    # Create sample files
    file_paths = []

    # 1. Create a calculator module
    calculator_path = sample_dir / "calculator.py"
    with open(calculator_path, "w", encoding="utf-8") as f:
        f.write("""def add(a, b):
    \"\"\"Add two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.
    \"\"\"
    return a + b

def subtract(a, b):
    \"\"\"Subtract two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The difference between a and b.
    \"\"\"
    return a - b

def multiply(a, b):
    \"\"\"Multiply two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The product of a and b.
    \"\"\"
    return a * b

def divide(a, b):
    \"\"\"Divide two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The quotient of a and b.
    \"\"\"
    return a / b""")
    file_paths.append(calculator_path)

    # 2. Create a main module
    main_path = sample_dir / "main.py"
    with open(main_path, "w", encoding="utf-8") as f:
        f.write("""from calculator import add, subtract, multiply, divide

def main():
    \"\"\"Main entry point for the calculator application.\"\"\"
    print("Calculator Application")
    print("=====================")

    a = float(input("Enter the first number: "))
    b = float(input("Enter the second number: "))

    print(f"{a} + {b} = {add(a, b)}")
    print(f"{a} - {b} = {subtract(a, b)}")
    print(f"{a} * {b} = {multiply(a, b)}")
    print(f"{a} / {b} = {divide(a, b)}")

if __name__ == "__main__":
    main()""")
    file_paths.append(main_path)

    return file_paths


def create_refactoring_workflow(manager, file_paths):
    """Create a refactoring workflow.

    Args:
        manager: The refactoring manager.
        file_paths: The file paths to refactor.

    Returns:
        The created workflow.
    """
    # Create a workflow
    workflow = manager.create_workflow(
        title="Calculator Refactoring",
        description="Refactor the calculator application to use a class-based approach."
    )

    # Add steps

    # Step 1: Create a Calculator class
    step1 = manager.add_step(
        workflow_id=workflow.id,
        title="Create Calculator class",
        description="Create a Calculator class to encapsulate the calculator functionality.",
        file_paths=[file_paths[0]],
        dependencies=[]
    )

    # Read the actual content of the file
    with open(file_paths[0], "r", encoding="utf-8") as f:
        calculator_content = f.read()

    # Add code changes to the step
    step1.changes["type"] = "generic"
    step1.changes["code_changes"] = {
        str(file_paths[0]): [
            {
                "type": "replace",
                "start_line": 1,
                "end_line": 44,
                "old_text": calculator_content,
                "new_text": """
class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def add(self, a, b):
        \"\"\"Add two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The sum of a and b.
        \"\"\"
        return a + b

    def subtract(self, a, b):
        \"\"\"Subtract two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The difference between a and b.
        \"\"\"
        return a - b

    def multiply(self, a, b):
        \"\"\"Multiply two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The product of a and b.
        \"\"\"
        return a * b

    def divide(self, a, b):
        \"\"\"Divide two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The quotient of a and b.
        \"\"\"
        return a / b

# For backward compatibility
def add(a, b):
    return Calculator().add(a, b)

def subtract(a, b):
    return Calculator().subtract(a, b)

def multiply(a, b):
    return Calculator().multiply(a, b)

def divide(a, b):
    return Calculator().divide(a, b)"""
            }
        ]
    }

    # Step 2: Update main.py to use the Calculator class
    step2 = manager.add_step(
        workflow_id=workflow.id,
        title="Update main.py",
        description="Update main.py to use the Calculator class.",
        file_paths=[file_paths[1]],
        dependencies=[step1.id]
    )

    # Read the actual content of the file
    with open(file_paths[1], "r", encoding="utf-8") as f:
        main_content = f.read()

    # Add code changes to the step
    step2.changes["type"] = "generic"
    step2.changes["code_changes"] = {
        str(file_paths[1]): [
            {
                "type": "replace",
                "start_line": 1,
                "end_line": 19,
                "old_text": main_content,
                "new_text": """
from calculator import Calculator

def main():
    \"\"\"Main entry point for the calculator application.\"\"\"
    print("Calculator Application")
    print("=====================")

    calculator = Calculator()

    a = float(input("Enter the first number: "))
    b = float(input("Enter the second number: "))

    print(f"{a} + {b} = {calculator.add(a, b)}")
    print(f"{a} - {b} = {calculator.subtract(a, b)}")
    print(f"{a} * {b} = {calculator.multiply(a, b)}")
    print(f"{a} / {b} = {calculator.divide(a, b)}")

if __name__ == "__main__":
    main()"""
            }
        ]
    }

    # Step 3: Remove backward compatibility functions
    step3 = manager.add_step(
        workflow_id=workflow.id,
        title="Remove backward compatibility",
        description="Remove the backward compatibility functions from calculator.py.",
        file_paths=[file_paths[0]],
        dependencies=[step2.id]
    )

    # After step 1 is executed, we need to read the updated content
    # This is a placeholder for the expected content after step 1
    expected_backward_compatibility = """# For backward compatibility
def add(a, b):
    return Calculator().add(a, b)

def subtract(a, b):
    return Calculator().subtract(a, b)

def multiply(a, b):
    return Calculator().multiply(a, b)

def divide(a, b):
    return Calculator().divide(a, b)"""

    # Add code changes to the step
    step3.changes["type"] = "generic"
    step3.changes["code_changes"] = {
        str(file_paths[0]): [
            {
                "type": "replace",
                "start_line": 45,
                "end_line": 57,
                "old_text": expected_backward_compatibility,
                "new_text": ""
            }
        ]
    }

    # Add checkpoints

    # Checkpoint 1: After creating the Calculator class
    checkpoint1 = manager.add_checkpoint(
        workflow_id=workflow.id,
        title="Calculator class created",
        description="Verify that the Calculator class has been created correctly.",
        step_ids=[step1.id],
        verification_criteria=[
            "Calculator class exists",
            "All methods are implemented correctly"
        ]
    )

    # Checkpoint 2: After updating main.py
    checkpoint2 = manager.add_checkpoint(
        workflow_id=workflow.id,
        title="Main module updated",
        description="Verify that the main module has been updated to use the Calculator class.",
        step_ids=[step2.id],
        verification_criteria=[
            "Main module imports Calculator class",
            "Main module uses Calculator class correctly"
        ]
    )

    # Checkpoint 3: After removing backward compatibility
    checkpoint3 = manager.add_checkpoint(
        workflow_id=workflow.id,
        title="Backward compatibility removed",
        description="Verify that the backward compatibility functions have been removed.",
        step_ids=[step3.id],
        verification_criteria=[
            "Backward compatibility functions are removed"
        ]
    )

    return workflow


def execute_workflow(executor, workflow_id):
    """Execute a refactoring workflow.

    Args:
        executor: The refactoring executor.
        workflow_id: The workflow ID.

    Returns:
        True if the workflow was executed successfully, False otherwise.
    """
    workflow = executor.manager.get_workflow(workflow_id)
    if not workflow:
        return False

    # For this example, we'll simulate the execution instead of actually executing the steps
    print("Simulating workflow execution...")

    # Simulate executing each step
    for step_id, step in workflow.steps.items():
        print(f"Simulating step: {step.title}")

        # Mark the step as completed
        executor.manager.update_step_status(workflow_id, step_id, StepStatus.COMPLETED)

        # Verify the checkpoint
        checkpoint = workflow.get_checkpoint_for_step(step_id)
        if checkpoint:
            print(f"Simulating checkpoint verification: {checkpoint.title}")

            # Add verification results
            for criterion in checkpoint.verification_criteria:
                # In a real implementation, this would prompt the user for verification
                # For this example, we'll just assume all criteria are met
                executor.manager.add_verification_result(workflow_id, checkpoint.id, criterion, True)

            # Mark the checkpoint as verified
            executor.manager.update_checkpoint_status(workflow_id, checkpoint.id, CheckpointStatus.VERIFIED)

    print("Workflow simulation completed successfully!")
    return True


def main():
    """Main entry point for the script."""
    setup_logging()

    # Create a temporary workspace directory
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_dir = Path(temp_dir)
        print(f"Created workspace directory: {workspace_dir}")

        # Create sample files
        file_paths = create_sample_files(workspace_dir)
        print(f"Created {len(file_paths)} sample files:")
        for file_path in file_paths:
            print(f"  {file_path}")

        # Create a refactoring manager
        manager = RefactoringManager(workspace_dir)

        # Create a refactoring workflow
        workflow = create_refactoring_workflow(manager, file_paths)
        print(f"Created workflow: {workflow.title}")
        print(f"  Steps: {len(workflow.steps)}")
        print(f"  Checkpoints: {len(workflow.checkpoints)}")

        # Create a refactoring executor
        executor = RefactoringExecutor(manager)

        # Create a progress tracker
        tracker = ProgressTracker()

        # Execute the workflow
        print("\nExecuting workflow...")
        success = execute_workflow(executor, workflow.id)

        if success:
            print("\nWorkflow executed successfully!")

            # Print the workflow progress
            progress = workflow.get_progress()
            print(f"\nWorkflow progress: {progress[0]}/{progress[1]} steps completed")

            # Print the checkpoint status
            print("\nCheckpoint status:")
            for checkpoint_id, checkpoint in workflow.checkpoints.items():
                print(f"  {checkpoint.title}: {checkpoint.status.value}")

                # Print verification results
                if checkpoint.verification_results:
                    print("    Verification results:")
                    for criterion, result in checkpoint.verification_results.items():
                        print(f"      {criterion}: {'✓' if result else '✗'}")
        else:
            print("\nWorkflow execution failed!")


if __name__ == "__main__":
    main()
