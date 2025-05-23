# Plan 01: TUI-Core Interface Separation and Decoupling

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: The `openhcs.tui` package currently exhibits high coupling with the `openhcs.core` modules. UI components and command logic in modules like `commands.py`, `menu_bar.py`, and `tui_architecture.py` directly import and interact with core classes such as `PipelineOrchestrator`, `AbstractStep`, `FunctionStep`, and `FUNC_REGISTRY`. This tight coupling violates fundamental architectural principles, leading to a brittle system where changes in the core can necessitate widespread changes in the TUI, and vice-versa. It also makes testing UI components in isolation difficult.

**Goal**: To establish a clear, well-defined boundary between the TUI (presentation and interaction mechanism) and the Core (business logic and data processing policy). This will be achieved by:
    1. Defining formal API contracts (Python `Protocol`s) for all TUI-to-Core interactions.
    2. Implementing an Adapter layer within the TUI (`openhcs.tui.core_adapters`) that conforms to these interfaces and mediates all communication with the Core.
    3. Refactoring existing TUI modules to use these adapters and interfaces, removing direct dependencies on Core implementation details.

**Architectural Principles**:
*   **Information Hiding**: Core implementation details will be hidden from the TUI.
*   **Law of Demeter (Principle of Least Knowledge)**: TUI components will only talk to their "immediate friends" (the adapter interfaces).
*   **Separation of Concerns**: UI logic will be distinctly separated from core processing logic.
*   **Dependency Inversion Principle**: TUI modules will depend on abstractions (interfaces/protocols), not on concrete core implementations.

## 2. Proposed Interfaces (to be created in `openhcs/tui/interfaces.py`)

This new file will house all protocols defining the contract between the TUI and its abstraction of the core.

```python
# openhcs/tui/interfaces.py
from typing import Protocol, Any, List, Optional, Dict, Coroutine, Union
from pathlib import Path

# Forward declare for type hints if needed, or use actual types if importable without circularity
if TYPE_CHECKING:
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.io.filemanager import FileManager
    # Add other core types that might appear in interface signatures if they don't create import cycles.
    # For data structures, prefer generic Dicts or Pydantic models if complex.


class CoreStepData(Protocol):
    """Data representation of a pipeline step for TUI consumption."""
    uid: str
    name: str
    step_type: str # e.g., "FunctionStep", "CompositeStep"
    func_display_name: Optional[str] # Name of the function/pattern
    params: Dict[str, Any]
    status: str # e.g., "new", "configured", "error"
    is_enabled: bool
    # Potentially other TUI-relevant metadata

    def to_dict(self) -> Dict[str, Any]: ...


class CorePlateData(Protocol):
    """Data representation of a plate/orchestrator for TUI consumption."""
    id: str # Unique identifier for the plate/orchestrator
    name: str # Display name, often derived from path
    path: str # Filesystem path to the plate data
    status: str # e.g., "new", "initialized", "compiled_ok", "error_init"
    backend_name: Optional[str]
    pipeline_definition_summary: Optional[str] # e.g., "5 steps" or list of step names


class CoreOrchestratorAdapterInterface(Protocol):
    """
    Interface for TUI interactions related to a single plate/pipeline orchestrator.
    Methods should be asynchronous if they involve I/O or potentially long-running core operations.
    """

    async def get_plate_data(self) -> CorePlateData: ...
    async def get_config(self) -> Dict[str, Any]: ...
    async def update_config(self, config_delta: Dict[str, Any]) -> None: ...

    async def initialize(self) -> None: ...
    async def get_pipeline_steps(self) -> List[CoreStepData]: ...
    async def add_step(self, step_type: str, func_identifier: Optional[str], default_name: str) -> Optional[CoreStepData]: ...
    async def update_step(self, step_uid: str, changes: Dict[str, Any]) -> Optional[CoreStepData]: ...
    async def remove_step(self, step_uid: str) -> bool: ...
    async def move_step(self, step_uid: str, direction: str) -> bool: ... # direction: "up" or "down"
    async def save_pipeline_definition_to_storage(self, path_override: Optional[Path] = None) -> str: ... # Returns actual save path
    async def load_pipeline_definition_from_storage(self, path: Path) -> List[CoreStepData]: ...

    async def compile_pipeline(self) -> bool: ... # Returns success status
    async def execute_compiled_pipeline(self) -> bool: ... # Returns success status
    async def get_last_compilation_error(self) -> Optional[str]: ...
    async def get_last_execution_error(self) -> Optional[str]: ...


class CoreApplicationAdapterInterface(Protocol):
    """
    Interface for TUI interactions related to global application state and general core functionalities.
    """

    async def get_global_config(self) -> 'GlobalPipelineConfig': ... # Use actual type if safe
    async def update_global_config(self, config_data: Dict[str, Any]) -> None: ... # Pass data, adapter handles conversion

    async def get_available_plates(self) -> List[CorePlateData]: ...
    async def add_new_plate(self, path: str, backend_name: str, plate_name: Optional[str] = None) -> Optional[str]: ... # Returns plate_id or None on failure
    async def remove_plate(self, plate_id: str) -> bool: ...
    async def get_orchestrator_adapter(self, plate_id: str) -> Optional[CoreOrchestratorAdapterInterface]: ...

    async def get_file_manager(self) -> 'FileManager': ... # Use actual type if safe
    async def get_func_registry_summary(self) -> Dict[str, List[str]]: ... # Backend -> List of func names
    async def get_function_details(self, backend_name: str, func_name: str) -> Optional[Dict[str, Any]]: ... # Pattern, params, etc.
    async def validate_function_pattern(self, pattern: Union[List, Dict]) -> bool: ...

    async def shutdown_core_services(self) -> None: ...
```

## 3. Adapter Implementation (New file: `openhcs/tui/core_adapters.py`)

This new module will contain the concrete adapter class(es).

```python
# openhcs/tui/core_adapters.py
import asyncio
from typing import Any, List, Optional, Dict, Coroutine, Union, Type
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.io.filemanager import FileManager
from openhcs.processing.func_registry import FUNC_REGISTRY
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.pipeline.funcstep_contract_validator import validate_pattern_structure # Example
from openhcs.constants.constants import Backend # Example

from .interfaces import (
    CoreApplicationAdapterInterface,
    CoreOrchestratorAdapterInterface,
    CoreStepData,
    CorePlateData
)

# Shared ThreadPoolExecutor for running synchronous core methods
# This can be defined here or passed in if a global executor is preferred.
SHARED_EXECUTOR = ThreadPoolExecutor(max_workers=5, thread_name_prefix="tui-core-adapter")

# Helper to convert core Step object to CoreStepData
def _core_step_to_tui_data(step: AbstractStep) -> CoreStepData:
    # This is a simplified conversion. Actual implementation will need more detail.
    func_display_name = None
    if isinstance(step, FunctionStep):
        # Attempt to get a display name from the function pattern if possible
        if isinstance(step.func_pattern, list) and step.func_pattern:
            first_call = step.func_pattern[0]
            if isinstance(first_call, dict) and 'func' in first_call:
                func_display_name = str(first_call['func'])
        elif isinstance(step.func_pattern, dict) and 'func' in step.func_pattern:
            func_display_name = str(step.func_pattern['func'])

    return { # type: ignore # Assuming CoreStepData is a TypedDict or similar
        "uid": step.uid,
        "name": step.name,
        "step_type": step.__class__.__name__,
        "func_display_name": func_display_name,
        "params": step.params.copy() if step.params else {}, # Ensure copy
        "status": getattr(step, 'status', 'unknown'), # Assuming steps might have a status
        "is_enabled": getattr(step, 'is_enabled', True),
    }

def _core_orchestrator_to_tui_plate_data(orchestrator: PipelineOrchestrator) -> CorePlateData:
    # Simplified conversion
    pipeline_def = orchestrator.pipeline_definition
    summary = f"{len(pipeline_def)} steps" if pipeline_def else "No pipeline"
    return { # type: ignore
        "id": orchestrator.plate_id,
        "name": orchestrator.plate_path.name,
        "path": str(orchestrator.plate_path),
        "status": getattr(orchestrator, 'status', 'unknown'), # Orchestrator needs a status attribute
        "backend_name": orchestrator.config.default_backend.value if orchestrator.config and orchestrator.config.default_backend else None,
        "pipeline_definition_summary": summary,
    }


class SingleOrchestratorAdapter(CoreOrchestratorAdapterInterface):
    def __init__(self, orchestrator: PipelineOrchestrator, loop: asyncio.AbstractEventLoop):
        self._orchestrator = orchestrator
        self._loop = loop

    async def _run_sync(self, func, *args, **kwargs):
        return await self._loop.run_in_executor(SHARED_EXECUTOR, lambda: func(*args, **kwargs))

    async def get_plate_data(self) -> CorePlateData:
        return await self._run_sync(_core_orchestrator_to_tui_plate_data, self._orchestrator)

    async def get_config(self) -> Dict[str, Any]:
        return await self._run_sync(lambda: self._orchestrator.config.model_dump() if self._orchestrator.config else {})

    async def update_config(self, config_delta: Dict[str, Any]) -> None:
        # Orchestrator config update logic needs to be defined in core
        # For now, assuming a method like `update_plate_config` exists
        await self._run_sync(getattr(self._orchestrator, 'update_plate_config', lambda cd: None), config_delta)

    async def initialize(self) -> None:
        await self._run_sync(self._orchestrator.initialize)
        # Update orchestrator status after initialization
        setattr(self._orchestrator, 'status', 'initialized')


    async def get_pipeline_steps(self) -> List[CoreStepData]:
        pipeline_def = await self._run_sync(lambda: self._orchestrator.pipeline_definition)
        return [_core_step_to_tui_data(step) for step in pipeline_def] if pipeline_def else []

    async def add_step(self, step_type: str, func_identifier: Optional[str], default_name: str) -> Optional[CoreStepData]:
        # This logic is complex and involves FUNC_REGISTRY.
        # For now, placeholder. Actual logic will create a step and add it.
        # The core orchestrator should have a method for this.
        # Example: new_step_core = await self._run_sync(self._orchestrator.add_new_step, step_type, func_identifier, default_name)
        # For now, let's assume it's handled by directly manipulating pipeline_definition for simplicity in this plan.
        # This part needs careful implementation in the orchestrator.
        
        # Simplified: Create a FunctionStep (assuming this is the primary type added)
        # This still uses FUNC_REGISTRY, which the adapter aims to abstract away from commands.
        # The adapter itself can use FUNC_REGISTRY.
        func_pattern = None
        if func_identifier and step_type == "FunctionStep":
            # func_identifier might be "backend_name.func_name"
            # This lookup should be more robust.
            parts = func_identifier.split('.', 1)
            if len(parts) == 2 and parts[0] in FUNC_REGISTRY and parts[1] in FUNC_REGISTRY[parts[0]]:
                 func_pattern = FUNC_REGISTRY[parts[0]][parts[1]]['pattern']

        new_core_step = FunctionStep(name=default_name, func_pattern=func_pattern)
        
        current_def = self._orchestrator.pipeline_definition
        if current_def is None: current_def = []
        current_def.append(new_core_step)
        self._orchestrator.pipeline_definition = current_def
        return _core_step_to_tui_data(new_core_step)


    async def update_step(self, step_uid: str, changes: Dict[str, Any]) -> Optional[CoreStepData]:
        # Find step by UID, update its attributes, return updated CoreStepData
        # This should ideally be a method on the orchestrator.
        pipeline_def = self._orchestrator.pipeline_definition
        if pipeline_def:
            for step in pipeline_def:
                if step.uid == step_uid:
                    # Apply changes. This is simplified.
                    # A real implementation would be more careful about what can be changed.
                    for key, value in changes.items():
                        if hasattr(step, key):
                            setattr(step, key, value)
                    # If func_pattern changed, it might need re-validation or re-creation of the step
                    if 'func_pattern' in changes and isinstance(step, FunctionStep):
                        step.func_pattern = changes['func_pattern'] # This is a direct update, might need more logic
                    return _core_step_to_tui_data(step)
        return None

    async def remove_step(self, step_uid: str) -> bool:
        pipeline_def = self._orchestrator.pipeline_definition
        if pipeline_def:
            original_len = len(pipeline_def)
            self._orchestrator.pipeline_definition = [s for s in pipeline_def if s.uid != step_uid]
            return len(self._orchestrator.pipeline_definition) < original_len
        return False

    async def move_step(self, step_uid: str, direction: str) -> bool:
        # Implement move logic within self._orchestrator.pipeline_definition
        # This is complex and needs careful index management.
        # Placeholder for now.
        return False # Placeholder

    async def save_pipeline_definition_to_storage(self, path_override: Optional[Path] = None) -> str:
        # Orchestrator needs a robust save method.
        # For now, mimic existing logic from MenuBar/Commands.
        # This still directly uses core types like AbstractStep.
        # The goal is for the adapter to handle this translation.
        pipeline_def = self._orchestrator.pipeline_definition
        if not pipeline_def:
            raise ValueError("Pipeline is empty, nothing to save.")

        default_filename = "pipeline_definition.json" # Simplified
        save_path = path_override or (self._orchestrator.plate_path / default_filename)

        pipeline_dicts = [step.to_dict() for step in pipeline_def] # Assumes to_dict() exists and is suitable

        # This should use FileManager from ProcessingContext if available
        # For simplicity, direct write:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            import json
            json.dump(pipeline_dicts, f, indent=2)
        setattr(self._orchestrator, 'status', 'saved') # Example status update
        return str(save_path)


    async def load_pipeline_definition_from_storage(self, path: Path) -> List[CoreStepData]:
        # Orchestrator needs a robust load method.
        # This should return List[CoreStepData]
        # For now, mimic existing logic.
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        with open(path, "rb") as f:
            import pickle # Assuming pickle for now, consistent with commands.py
            loaded_core_steps = pickle.load(f) # This loads actual AbstractStep instances

        if not isinstance(loaded_core_steps, list) or \
           not all(isinstance(item, AbstractStep) for item in loaded_core_steps):
            raise ValueError("Invalid pipeline file format.")

        self._orchestrator.pipeline_definition = loaded_core_steps
        setattr(self._orchestrator, 'status', 'loaded') # Example status update
        return [_core_step_to_tui_data(step) for step in loaded_core_steps]


    async def compile_pipeline(self) -> bool:
        # This needs to handle the pipeline_definition correctly.
        # The orchestrator's compile_pipelines might expect the actual step objects.
        if not self._orchestrator.pipeline_definition:
            setattr(self._orchestrator, 'status', 'error_compile')
            setattr(self._orchestrator, 'last_compile_error', 'Pipeline definition is empty.')
            return False
        try:
            compiled_data = await self._run_sync(
                self._orchestrator.compile_pipelines,
                self._orchestrator.pipeline_definition # Pass the list of AbstractStep objects
            )
            self._orchestrator.last_compiled_contexts = compiled_data # Store on orchestrator
            setattr(self._orchestrator, 'status', 'compiled_ok')
            setattr(self._orchestrator, 'last_compile_error', None)
            return True
        except Exception as e:
            setattr(self._orchestrator, 'status', 'error_compile')
            setattr(self._orchestrator, 'last_compile_error', str(e))
            return False

    async def execute_compiled_pipeline(self) -> bool:
        if getattr(self._orchestrator, 'status', '') != 'compiled_ok' or \
           not hasattr(self._orchestrator, 'last_compiled_contexts') or \
           not self._orchestrator.last_compiled_contexts:
            setattr(self._orchestrator, 'status', 'error_run')
            setattr(self._orchestrator, 'last_exec_error', 'Pipeline not compiled or compiled contexts missing.')
            return False
        try:
            await self._run_sync(
                self._orchestrator.execute_compiled_plate,
                self._orchestrator.pipeline_definition, # Stateless definition
                self._orchestrator.last_compiled_contexts
            )
            setattr(self._orchestrator, 'status', 'run_completed')
            setattr(self._orchestrator, 'last_exec_error', None)
            return True
        except Exception as e:
            setattr(self._orchestrator, 'status', 'error_run')
            setattr(self._orchestrator, 'last_exec_error', str(e))
            return False

    async def get_last_compilation_error(self) -> Optional[str]:
        return getattr(self._orchestrator, 'last_compile_error', None)

    async def get_last_execution_error(self) -> Optional[str]:
        return getattr(self._orchestrator, 'last_exec_error', None)


class TUICoreAdapter(CoreApplicationAdapterInterface):
    def __init__(self, initial_context: ProcessingContext, global_config: GlobalPipelineConfig):
        self._initial_context = initial_context
        self._global_config = global_config # This is the single, shared instance
        self._file_manager = initial_context.filemanager
        self._loop = asyncio.get_event_loop()
        self._orchestrators: Dict[str, PipelineOrchestrator] = {} # Manages core orchestrator instances

    async def _run_sync(self, func, *args, **kwargs):
        return await self._loop.run_in_executor(SHARED_EXECUTOR, lambda: func(*args, **kwargs))

    async def get_global_config(self) -> GlobalPipelineConfig:
        return self._global_config # Return the shared instance

    async def update_global_config(self, config_data: Dict[str, Any]) -> None:
        # The GlobalPipelineConfig is a Pydantic model.
        # It should be updated carefully. The launcher might be the sole owner.
        # This adapter method might just trigger a save if the config object itself is mutated elsewhere,
        # or it might update the shared instance.
        # For now, assume direct update of the shared instance.
        # This needs careful consideration of ownership.
        try:
            updated_config = GlobalPipelineConfig(**{**self._global_config.model_dump(), **config_data})
            # Update the shared instance fields.
            for field_name, value in updated_config.model_dump().items():
                setattr(self._global_config, field_name, value)
            # Persist changes if GlobalPipelineConfig has a save method
            if hasattr(self._global_config, 'save_to_default_location'):
                 await self._run_sync(self._global_config.save_to_default_location)
        except Exception as e:
            # Log error
            print(f"Error updating global config via adapter: {e}") # Replace with logger

    async def get_available_plates(self) -> List[CorePlateData]:
        # This implies the adapter needs to know about all active orchestrators
        # or query a central registry if one exists in core.
        # For now, assume self._orchestrators holds them.
        return [_core_orchestrator_to_tui_plate_data(orch) for orch in self._orchestrators.values()]

    async def add_new_plate(self, path: str, backend_name: str, plate_name: Optional[str] = None) -> Optional[str]:
        # This involves creating a new PipelineOrchestrator instance.
        # The orchestrator's constructor or a factory method should be used.
        # The ProcessingContext needs to be correctly configured for this new plate.
        plate_path = Path(path)
        name = plate_name or plate_path.name
        plate_id = str(plate_path.resolve()) # Use resolved path as a unique ID

        if plate_id in self._orchestrators:
            # Log warning: plate already exists
            return plate_id

        # Create a new context for this plate, possibly derived from initial_context
        # This is a simplification; context creation might be more involved.
        plate_specific_context = ProcessingContext(
            filemanager=self._file_manager, # Share filemanager
            global_config=self._global_config # Share global_config
        )
        # The orchestrator config needs to be created/loaded for this plate.
        # This is a major simplification.
        from openhcs.core.orchestrator.config import OrchestratorConfig # Example
        
        # Determine backend Enum member
        try:
            backend_enum_member = Backend[backend_name.upper()]
        except KeyError:
            # Log error: invalid backend name
            return None

        orch_config = OrchestratorConfig(
            plate_path=plate_path,
            default_backend=backend_enum_member,
            # other necessary config fields
        )
        try:
            orchestrator = await self._run_sync(
                PipelineOrchestrator,
                context=plate_specific_context,
                config_override=orch_config, # Pass specific config
                plate_id_override=plate_id
            )
            # Initialize status for new orchestrator
            setattr(orchestrator, 'status', 'new')
            self._orchestrators[plate_id] = orchestrator
            return plate_id
        except Exception as e:
            # Log error creating orchestrator
            return None


    async def remove_plate(self, plate_id: str) -> bool:
        if plate_id in self._orchestrators:
            # Any cleanup for the orchestrator?
            del self._orchestrators[plate_id]
            return True
        return False

    async def get_orchestrator_adapter(self, plate_id: str) -> Optional[CoreOrchestratorAdapterInterface]:
        orchestrator = self._orchestrators.get(plate_id)
        if orchestrator:
            return SingleOrchestratorAdapter(orchestrator, self._loop)
        return None

    async def get_file_manager(self) -> FileManager:
        return self._file_manager

    async def get_func_registry_summary(self) -> Dict[str, List[str]]:
        summary = {}
        for backend, funcs in FUNC_REGISTRY.items():
            summary[str(backend)] = list(funcs.keys())
        return summary

    async def get_function_details(self, backend_name: str, func_name: str) -> Optional[Dict[str, Any]]:
        # Convert backend_name string to Backend enum if necessary for FUNC_REGISTRY access
        try:
            backend_enum = Backend[backend_name.upper()]
            if backend_enum in FUNC_REGISTRY and func_name in FUNC_REGISTRY[backend_enum]:
                return FUNC_REGISTRY[backend_enum][func_name].copy() # Return a copy
        except KeyError:
            pass # Backend name not found
        return None

    async def validate_function_pattern(self, pattern: Union[List, Dict]) -> bool:
        # This should call the core validation logic.
        try:
            # Assuming validate_pattern_structure is synchronous
            await self._run_sync(validate_pattern_structure, pattern, FUNC_REGISTRY)
            return True
        except Exception:
            return False # Or raise a specific validation error

    async def shutdown_core_services(self) -> None:
        # Example: if orchestrators need explicit shutdown
        for orch in self._orchestrators.values():
            if hasattr(orch, 'shutdown') and callable(orch.shutdown):
                await self._run_sync(orch.shutdown)
        SHARED_EXECUTOR.shutdown(wait=True)
```

## 4. Refactoring TUI Modules

High-level strategy for refactoring key TUI modules:

*   **`openhcs.tui.TUIState`**:
    *   Will no longer hold direct instances of `PipelineOrchestrator`. Instead, it will store `plate_id` strings.
    *   `active_orchestrator` attribute will be removed or changed to `active_plate_id: Optional[str]`.
    *   `current_pipeline_definition` will store `List[CoreStepData]` (or `List[Dict]`) instead of `List[AbstractStep]`.
    *   `step_to_edit_config` will store `CoreStepData` (or `Dict`) instead of `FunctionStep`.

*   **`openhcs.tui.OpenHCSTUI`**:
    *   Will instantiate and hold a single `TUICoreAdapter` instance.
    *   All interactions that previously accessed `self.state.active_orchestrator` will now:
        1.  Get `active_plate_id` from `self.state`.
        2.  Call `self.core_adapter.get_orchestrator_adapter(active_plate_id)`.
        3.  Use the returned `CoreOrchestratorAdapterInterface` to perform operations.
    *   Example: `_handle_show_edit_plate_config_request` will pass the `CoreOrchestratorAdapterInterface` for the selected plate to the dialog/editor, or the dialog/editor will fetch it using `plate_id`.

*   **`openhcs.tui.commands.py`**:
    *   The `Command.execute` signature will change:
        `async def execute(self, app_adapter: CoreApplicationAdapterInterface, plate_adapter: Optional[CoreOrchestratorAdapterInterface], state: "TUIState", **kwargs: Any) -> None:`
        (Or pass `TUICoreAdapter` and let commands get specific adapters).
    *   Commands like `InitializePlatesCommand`, `CompilePlatesCommand`, `RunPlatesCommand` will use methods from `plate_adapter` (e.g., `plate_adapter.initialize()`).
    *   `AddStepCommand` will use `plate_adapter.add_step(...)`. It will no longer directly instantiate `FunctionStep` or access `FUNC_REGISTRY`.
    *   `LoadPipelineCommand` / `SavePipelineCommand` will use `plate_adapter.load_pipeline_definition_from_storage(...)` and `save_pipeline_definition_to_storage(...)`.
    *   Direct imports of core types like `PipelineOrchestrator`, `AbstractStep`, `FunctionStep`, `FUNC_REGISTRY`, `GlobalPipelineConfig` (for direct use, type hints for interfaces are okay) will be removed.
    *   The `SHARED_EXECUTOR` currently in `commands.py` will be moved to `core_adapters.py` or be a shared utility.

*   **`openhcs.tui.menu_bar.py` (and other UI components like `PlateManagerPane`, `PipelineEditorPane`, `DualStepFuncEditorPane`)**:
    *   Event handlers (e.g., `_on_compile`, `_on_run`) will primarily construct and dispatch `Command` objects.
    *   The `CommandRegistry` will execute these commands, passing the necessary adapter interfaces.
    *   Conditional enabling/disabling of menu items (`Condition` objects) will rely on data fetched from `TUIState`. `TUIState` itself will be updated via notifications originating from commands (after adapter calls) or directly from adapter methods if appropriate (e.g., after a polling update).
    *   Direct core imports will be removed.

## 5. Specific Import Changes (Illustrative)

*   **In `openhcs.tui.commands.py`**:
    *   **REMOVE**:
        ```python
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
        from openhcs.core.steps.abstract import AbstractStep
        from openhcs.core.steps.function_step import FunctionStep
        from openhcs.processing.func_registry import FUNC_REGISTRY
        from openhcs.core.config import GlobalPipelineConfig
        from openhcs.core.context.processing_context import ProcessingContext # If only used for type hints for core objects
        ```
    *   **ADD**:
        ```python
        from .interfaces import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface, CoreStepData
        # TYPE_CHECKING imports for TUIState, GlobalPipelineConfig (if used in signatures)
        ```

*   **In `openhcs.tui.tui_architecture.py` (`OpenHCSTUI` class)**:
    *   **REMOVE**: Direct instantiation of `PipelineOrchestrator`.
    *   **ADD**:
        ```python
        from .core_adapters import TUICoreAdapter
        from .interfaces import CoreStepData # For TUIState.current_pipeline_definition
        # TYPE_CHECKING for GlobalPipelineConfig, ProcessingContext for constructor
        ```

## 6. Addressing Dependency/Call Graph Findings

*   **Breaking Direct Dependencies**: This plan directly addresses and aims to eliminate the following critical dependencies identified in `reports/code_analysis/tui_comprehensive.md/module_dependency_graph_tui.md`:
    *   `openhcs.tui.commands` -> `openhcs.core.orchestrator.orchestrator`, `openhcs.core.steps.*`, `openhcs.processing.func_registry`, `openhcs.core.config`.
    *   `openhcs.tui.menu_bar` -> `openhcs.core.orchestrator.orchestrator`, `openhcs.core.steps.abstract`, `openhcs.core.config`.
    *   `openhcs.tui.tui_architecture` -> `openhcs.core.orchestrator.orchestrator`, `openhcs.core.steps.*`, `openhcs.processing.func_registry`.
    *   `openhcs.tui.plate_manager_core` -> `openhcs.core.orchestrator.orchestrator`.
    *   `openhcs.tui.pipeline_editor` -> `openhcs.core.orchestrator.orchestrator`, `openhcs.core.steps.*`.
*   **Resolution via Adapter**: The `TUICoreAdapter` and the defined interfaces act as an intermediary. TUI components will only know about these TUI-local abstractions, not the concrete core classes.

## 7. Verification Steps

1.  **Static Analysis**: After implementing changes, re-run `python tools/code_analysis/code_analyzer_cli.py dependencies openhcs/tui -o updated_tui_dependencies.md`. Verify that direct dependencies to `openhcs.core` from the refactored TUI modules are significantly reduced or eliminated, replaced by dependencies on `openhcs.tui.interfaces` and `openhcs.tui.core_adapters`.
2.  **Unit Tests**:
    *   Write unit tests for `TUICoreAdapter` and `SingleOrchestratorAdapter`. Mock the actual core objects (`PipelineOrchestrator`, `GlobalPipelineConfig`, etc.) and verify that adapter methods correctly delegate calls and handle data transformations.
    *   Update unit tests for `Command` subclasses. Mock the adapter interfaces and verify that commands call the correct adapter methods with appropriate arguments.
3.  **Integration Tests (TUI-Adapter)**: Test the interaction between TUI components (like `PlateManagerPane`, `PipelineEditorPane` through their commands) and the `TUICoreAdapter`, ensuring that UI actions correctly translate to adapter calls.
4.  **End-to-End Tests**: Existing TUI end-to-end tests (if any) should continue to pass, demonstrating that the refactoring has not broken overall functionality. New tests might be needed to cover specific interaction flows through the adapter.
5.  **Code Review**: Focus on ensuring that no direct core imports remain in the presentation/command layers of the TUI, and that all core interactions pass through the defined interfaces and adapter.

This plan provides a clear path to decouple the TUI from core logic, improving modularity, testability, and maintainability.