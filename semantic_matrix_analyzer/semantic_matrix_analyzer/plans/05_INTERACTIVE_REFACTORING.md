# Plan 05: Interactive Refactoring Workflow

## Objective

Develop an interactive workflow for complex refactoring tasks that breaks down large changes into manageable steps, tracks progress, provides checkpoints for verification, and supports rollbacks if needed.

## Rationale

Complex refactoring tasks can be overwhelming and error-prone. By implementing an interactive refactoring workflow:

1. **Manageable Steps**: Break down complex refactorings into smaller, verifiable steps
2. **Progress Tracking**: Keep track of progress through multi-stage refactorings
3. **Verification Points**: Allow users to verify intermediate results
4. **Rollback Support**: Enable recovery from failed refactoring steps
5. **Cognitive Load Reduction**: Reduce the mental effort required to manage complex refactorings

## Implementation Details

### 1. Refactoring Plan Generation

Create a system for generating refactoring plans:

```python
@dataclass
class RefactoringStep:
    """A step in a refactoring plan."""
    id: str
    description: str
    files: List[Path]
    dependencies: List[str]  # IDs of steps that must be completed first
    estimated_effort: str  # "low", "medium", "high"
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    changes: Dict[Path, str] = field(default_factory=dict)  # file_path -> change_description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the step to a dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "files": [str(f) for f in self.files],
            "dependencies": self.dependencies,
            "estimated_effort": self.estimated_effort,
            "status": self.status,
            "changes": {str(k): v for k, v in self.changes.items()}
        }

@dataclass
class RefactoringPlan:
    """A plan for refactoring code."""
    id: str
    title: str
    description: str
    steps: List[RefactoringStep]
    created_at: datetime
    updated_at: datetime
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the plan to a dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status
        }
    
    def get_next_steps(self) -> List[RefactoringStep]:
        """Get the next steps that can be executed."""
        completed_step_ids = {s.id for s in self.steps if s.status == "completed"}
        
        next_steps = []
        for step in self.steps:
            if step.status == "pending":
                # Check if all dependencies are completed
                if all(dep in completed_step_ids for dep in step.dependencies):
                    next_steps.append(step)
        
        return next_steps
    
    def get_step(self, step_id: str) -> Optional[RefactoringStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def update_step_status(self, step_id: str, status: str) -> None:
        """Update the status of a step."""
        step = self.get_step(step_id)
        if step:
            step.status = status
            self.updated_at = datetime.now()
            
            # Update the plan status based on step statuses
            if all(s.status == "completed" for s in self.steps):
                self.status = "completed"
            elif any(s.status == "failed" for s in self.steps):
                self.status = "failed"
            elif any(s.status == "in_progress" for s in self.steps):
                self.status = "in_progress"

class RefactoringPlanGenerator:
    """Generates refactoring plans."""
    
    def generate_plan(self, title: str, description: str, files: List[Path]) -> RefactoringPlan:
        """Generate a refactoring plan."""
        # Create a unique ID for the plan
        plan_id = f"refactoring_{uuid.uuid4().hex[:8]}"
        
        # Analyze the files to determine refactoring steps
        steps = self._analyze_files(files)
        
        # Create the plan
        plan = RefactoringPlan(
            id=plan_id,
            title=title,
            description=description,
            steps=steps,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return plan
    
    def _analyze_files(self, files: List[Path]) -> List[RefactoringStep]:
        """Analyze files to determine refactoring steps."""
        # This would implement analysis of files to determine refactoring steps
        # For now, we'll return a placeholder step
        return [
            RefactoringStep(
                id=f"step_{uuid.uuid4().hex[:8]}",
                description="Placeholder refactoring step",
                files=files,
                dependencies=[],
                estimated_effort="medium"
            )
        ]
```

### 2. Refactoring Execution

Create a system for executing refactoring steps:

```python
class RefactoringExecutor:
    """Executes refactoring steps."""
    
    def __init__(self, verifier):
        self.verifier = verifier
    
    def execute_step(self, step: RefactoringStep) -> bool:
        """Execute a refactoring step."""
        # Update the step status
        step.status = "in_progress"
        
        try:
            # Execute the changes for each file
            for file_path, change_description in step.changes.items():
                self._execute_change(file_path, change_description)
            
            # Verify the changes
            verification_result = self.verifier.verify_changes(step.files)
            
            if verification_result.is_valid:
                # Update the step status
                step.status = "completed"
                return True
            else:
                # Update the step status
                step.status = "failed"
                return False
        except Exception as e:
            # Update the step status
            step.status = "failed"
            return False
    
    def _execute_change(self, file_path: Path, change_description: str) -> None:
        """Execute a change to a file."""
        # This would implement execution of a change to a file
        pass
```

### 3. Checkpoint Management

Create a system for managing checkpoints:

```python
@dataclass
class Checkpoint:
    """A checkpoint in a refactoring process."""
    id: str
    description: str
    created_at: datetime
    files: Dict[Path, str]  # file_path -> file_content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the checkpoint to a dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "files": {str(k): v for k, v in self.files.items()}
        }

class CheckpointManager:
    """Manages checkpoints in a refactoring process."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.checkpoints = {}
        self._load_checkpoints()
    
    def _load_checkpoints(self) -> None:
        """Load checkpoints from storage."""
        # Load checkpoints from JSON files in the storage directory
        pass
    
    def create_checkpoint(self, description: str, files: List[Path]) -> Checkpoint:
        """Create a checkpoint."""
        # Create a unique ID for the checkpoint
        checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:8]}"
        
        # Save the current state of the files
        file_contents = {}
        for file_path in files:
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    file_contents[file_path] = f.read()
        
        # Create the checkpoint
        checkpoint = Checkpoint(
            id=checkpoint_id,
            description=description,
            created_at=datetime.now(),
            files=file_contents
        )
        
        # Save the checkpoint
        self.checkpoints[checkpoint_id] = checkpoint
        self._save_checkpoint(checkpoint)
        
        return checkpoint
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore a checkpoint."""
        checkpoint = self.checkpoints.get(checkpoint_id)
        if not checkpoint:
            return False
        
        # Restore the files
        for file_path, file_content in checkpoint.files.items():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)
        
        return True
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        return self.checkpoints.get(checkpoint_id)
    
    def get_all_checkpoints(self) -> List[Checkpoint]:
        """Get all checkpoints."""
        return list(self.checkpoints.values())
    
    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to storage."""
        # Save the checkpoint to a JSON file
        pass
```

### 4. Rollback Support

Create a system for rolling back changes:

```python
class RollbackManager:
    """Manages rollbacks in a refactoring process."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Roll back to a checkpoint."""
        return self.checkpoint_manager.restore_checkpoint(checkpoint_id)
    
    def rollback_step(self, step: RefactoringStep) -> bool:
        """Roll back a refactoring step."""
        # This would implement rolling back a refactoring step
        # For now, we'll return a placeholder result
        return True
```

### 5. Interactive Workflow

Create a system for managing the interactive workflow:

```python
class InteractiveRefactoringWorkflow:
    """Manages an interactive refactoring workflow."""
    
    def __init__(
        self,
        plan_generator: RefactoringPlanGenerator,
        executor: RefactoringExecutor,
        checkpoint_manager: CheckpointManager,
        rollback_manager: RollbackManager
    ):
        self.plan_generator = plan_generator
        self.executor = executor
        self.checkpoint_manager = checkpoint_manager
        self.rollback_manager = rollback_manager
        self.current_plan = None
    
    def start_refactoring(self, title: str, description: str, files: List[Path]) -> RefactoringPlan:
        """Start a refactoring process."""
        # Create a checkpoint before starting
        self.checkpoint_manager.create_checkpoint("Initial state", files)
        
        # Generate a refactoring plan
        self.current_plan = self.plan_generator.generate_plan(title, description, files)
        
        return self.current_plan
    
    def get_next_steps(self) -> List[RefactoringStep]:
        """Get the next steps that can be executed."""
        if not self.current_plan:
            return []
        
        return self.current_plan.get_next_steps()
    
    def execute_step(self, step_id: str) -> bool:
        """Execute a refactoring step."""
        if not self.current_plan:
            return False
        
        step = self.current_plan.get_step(step_id)
        if not step:
            return False
        
        # Create a checkpoint before executing the step
        self.checkpoint_manager.create_checkpoint(f"Before step {step_id}", step.files)
        
        # Execute the step
        result = self.executor.execute_step(step)
        
        # Update the plan
        self.current_plan.update_step_status(step_id, "completed" if result else "failed")
        
        return result
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Roll back to a checkpoint."""
        return self.rollback_manager.rollback_to_checkpoint(checkpoint_id)
    
    def get_plan_status(self) -> Dict[str, Any]:
        """Get the status of the current plan."""
        if not self.current_plan:
            return {"status": "no_plan"}
        
        return {
            "status": self.current_plan.status,
            "completed_steps": sum(1 for s in self.current_plan.steps if s.status == "completed"),
            "total_steps": len(self.current_plan.steps),
            "next_steps": [s.to_dict() for s in self.current_plan.get_next_steps()]
        }
```

## Success Criteria

1. Generation of refactoring plans with manageable steps
2. Execution of refactoring steps with verification
3. Checkpoint management for tracking progress
4. Rollback support for failed steps
5. Interactive workflow for guiding users through refactorings

## Dependencies

- Existing code analysis system
- Existing verification system

## Timeline

- Research and design: 1 week
- Refactoring plan generation: 2 weeks
- Refactoring execution: 2 weeks
- Checkpoint management: 1 week
- Rollback support: 1 week
- Interactive workflow: 2 weeks
- Testing and documentation: 1 week

Total: 10 weeks
