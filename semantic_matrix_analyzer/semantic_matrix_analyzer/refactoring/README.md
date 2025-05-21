# Interactive Refactoring Workflow

The Interactive Refactoring Workflow system is a component of the Semantic Matrix Analyzer that provides functionality for interactive refactoring workflows that break down large changes into manageable steps, track progress, provide checkpoints for verification, and support rollbacks if needed.

## Overview

The Interactive Refactoring Workflow system consists of several key components:

1. **Workflow Management**: Defines and manages refactoring workflows
2. **Step Execution**: Executes refactoring steps
3. **Checkpoint Verification**: Verifies that refactoring steps have been completed correctly
4. **Progress Tracking**: Tracks progress in refactoring workflows

## Usage

### Basic Usage

```python
from semantic_matrix_analyzer.refactoring import (
    RefactoringWorkflow, RefactoringStep, RefactoringCheckpoint,
    StepStatus, CheckpointStatus, RefactoringManager, RefactoringExecutor,
    ProgressTracker, ProgressEvent
)

# Create a refactoring manager
manager = RefactoringManager(Path("workspace"))

# Create a workflow
workflow = manager.create_workflow(
    title="Refactoring Example",
    description="Example refactoring workflow"
)

# Add a step
step = manager.add_step(
    workflow_id=workflow.id,
    title="Example Step",
    description="An example refactoring step",
    file_paths=[Path("example.py")],
    dependencies=[]
)

# Add code changes to the step
step.changes["type"] = "generic"
step.changes["code_changes"] = {
    "example.py": [
        {
            "type": "replace",
            "start_line": 1,
            "end_line": 5,
            "old_text": "def example():\n    return 42",
            "new_text": "def example():\n    return 43"
        }
    ]
}

# Add a checkpoint
checkpoint = manager.add_checkpoint(
    workflow_id=workflow.id,
    title="Example Checkpoint",
    description="An example checkpoint",
    step_ids=[step.id],
    verification_criteria=["Code compiles", "Tests pass"]
)

# Create an executor
executor = RefactoringExecutor(manager)

# Execute the step
executor.execute_step(workflow.id, step.id)

# Add verification results
manager.add_verification_result(workflow.id, checkpoint.id, "Code compiles", True)
manager.add_verification_result(workflow.id, checkpoint.id, "Tests pass", True)

# Verify the checkpoint
executor.verify_checkpoint(workflow.id, checkpoint.id)
```

### Tracking Progress

```python
# Create a progress tracker
tracker = ProgressTracker()

# Add events
tracker.add_event(ProgressEvent(
    id=str(uuid.uuid4()),
    workflow_id=workflow.id,
    event_type="step_status_changed",
    timestamp=datetime.now(),
    details={
        "step_id": step.id,
        "status": StepStatus.COMPLETED.value
    }
))

# Get workflow progress
progress = tracker.get_workflow_progress(workflow.id)
print(f"Completion: {progress['completion_percentage']}%")

# Get workflow timeline
timeline = tracker.get_workflow_timeline(workflow.id)
for event in timeline:
    print(f"{event['timestamp']}: {event['event_type']}")
```

## Components

### Workflow Management

Defines and manages refactoring workflows:

- `RefactoringWorkflow`: A workflow for interactive refactoring
- `RefactoringStep`: A step in a refactoring workflow
- `RefactoringCheckpoint`: A checkpoint in a refactoring workflow
- `RefactoringManager`: Manages refactoring workflows

### Step Execution

Executes refactoring steps:

- `RefactoringExecutor`: Executes refactoring steps
- `execute_step`: Executes a refactoring step
- `rollback_step`: Rolls back a refactoring step
- `verify_checkpoint`: Verifies a refactoring checkpoint
- `reject_checkpoint`: Rejects a refactoring checkpoint

### Progress Tracking

Tracks progress in refactoring workflows:

- `ProgressEvent`: An event in a refactoring workflow
- `ProgressTracker`: Tracks progress in refactoring workflows
- `get_workflow_progress`: Gets the progress of a workflow
- `get_workflow_timeline`: Gets a timeline of events for a workflow

## Benefits

The Interactive Refactoring Workflow system provides several benefits:

1. **Manageable Steps**: Breaks down large refactoring tasks into manageable steps
2. **Progress Tracking**: Tracks progress and provides visibility into the refactoring process
3. **Verification**: Ensures that refactoring steps are completed correctly
4. **Rollback Support**: Allows rolling back changes if needed
5. **Collaboration**: Facilitates collaboration on refactoring tasks
