# plan_11_precise_backend_violation_matrix.md
## Component: Precise Backend Violation Matrix

### Architectural Principles

1. **Pattern Resolver & Discovery**: Backend agnostic. Must be explicitly told what backend to use and given the file_manager.
2. **Microscopes**: Don't own backends or file_managers. These must be provided.
3. **FileManager**: Guardian between normal paths and virtual paths. Allows normal paths to be written to different backends.
4. **Steps**: Should get their file_manager from the context.
5. **Pipeline**: Passes file_manager around, doesn't own it.
6. **Registry**: Should be moved from file_managers to orchestrator. Orchestrator initializes registry and passes it by reference to file_managers upon construction.
7. **Backend Parameters**: Must be positional-only in function definitions and call sites (Clause 306).
8. **Backend Registry**: Must support both shared and isolated modes, configurable at orchestrator level (Clause 308).
9. **Backend Method Purity**: All FileManager methods must only use the passed backend parameter with no internal resolution (Clause 305).
10. **Backend Propagation**: Any function accepting file_manager must also accept backend and propagate it to every I/O call (Clause 310).
11. **VirtualPath Encapsulation**: VirtualPath objects must be restricted to the I/O layer and not exposed in FileManager public methods.
12. **Step Backend Metadata**: All steps must declare read_backend and write_backend in metadata for traceability.

### Violation Detection Prompt

```
# Backend Contract Violation Scanner

Scan the codebase "ezstitcher/" for violations of the following backend-related contracts:

1. PATTERN_RESOLVER_BACKEND_AGNOSTIC: Pattern resolver modules must be backend agnostic. They must:
   - Accept a backend parameter in all functions that perform I/O
   - Not have default backends
   - Not infer backends from context
   - Pass the backend parameter to all called functions that require it
   - Backend parameter must be positional-only (not keyword)

2. FILEMANAGER_GUARDIAN: FileManager is the guardian between normal paths and virtual paths:
   - All path operations must go through FileManager
   - FileManager must validate backends
   - No direct manipulation of paths outside FileManager
   - All methods must only use the passed backend parameter
   - No internal backend resolution or default behavior

3. STEPS_CONTEXT_FILEMANAGER: Steps must get their file_manager from the context:
   - No creating FileManager instances in steps
   - Must use context.file_manager
   - Must pass backend explicitly to file_manager methods
   - Must declare read_backend and write_backend in step metadata

4. PIPELINE_FILEMANAGER_PASSING: Pipeline just passes file_manager around:
   - No creating FileManager instances in pipeline
   - Must pass file_manager to steps
   - Must not modify file_manager
   - Must propagate backend flags from step metadata

5. REGISTRY_IN_ORCHESTRATOR: Registry should be in orchestrator:
   - Registry should be initialized by orchestrator
   - Registry should be passed to file_managers
   - No creating registry instances outside orchestrator
   - Registry must support both shared and isolated modes

6. VIRTUALPATH_ENCAPSULATION: VirtualPath objects must be restricted to the I/O layer:
   - No VirtualPath objects exposed in FileManager public methods
   - FileManager public methods must accept and return standard path types (str, Path)
   - VirtualPath objects must only be used internally within the I/O layer
   - No VirtualPath imports outside of the I/O layer

7. BACKEND_DECLARATION_IN_PLANS: All step plans must declare read_backend and write_backend:
   - Backend declarations must be statically inspectable in the plan file or constructor
   - Backend must be passed into the step instance at initialization
   - Backend must never be inferred from memory type or runtime behavior
   - Purpose: Prevent silent I/O, ensure backend is part of static compilation

8. FUNCTION_BACKEND_PROPAGATION: Any function that accepts a file_manager must also accept a backend:
   - Must pass that backend to any downstream calls
   - Must not hold onto it as internal state
   - Must not resolve it from context, class attribute, or default
   - Pattern of enforcement: `def foo(file_manager, backend): helper(file_manager, backend)`

For each violation, report:
1. File path
2. Line number
3. Violation type
4. Current code
5. Exact fix
```

### Precise Violation Matrix (Unresolved from Original Plan)

| ID | File Path | Line | Violation Type | Current Code | Exact Fix |
|----|-----------|------|---------------|--------------|-----------|
| V11 | ezstitcher/io/file_manager/file_manager_core.py | e.g., 69 | REGISTRY_IN_ORCHESTRATOR | `storage_backend_registry.get_backend(...)` | Use `self.registry.get_backend_instance(...)` |
| V21 | ezstitcher/io/file_manager/file_manager_core.py | 171 | VIRTUALPATH_ENCAPSULATION | `def get_local_path(self, virtual_path: VirtualPath) -> 'Path':` | Change parameter type to `Union[str, Path]` and handle conversion internally |

### Newly Found Violations

| ID | File Path | Line | Violation Type | Current Code | Justification |
|----|-----------|------|---------------|--------------|---------------|
| N1 | ezstitcher/formats/pattern/pattern_discovery.py | 72, 76 | FILEMANAGER_GUARDIAN | `directory_path.exists()` | Direct FS access (Clause 17) |
| N2 | ezstitcher/tui/function_pattern_editor.py | 553 | FILEMANAGER_GUARDIAN | `open(temp_file_path, 'r')` | Direct FS access (Clause 17) |
| N3 | ezstitcher/tui/function_pattern_editor.py | 585 | FILEMANAGER_GUARDIAN | `os.unlink(temp_file_path)` | Direct FS access (Clause 17) |
| N4 | ezstitcher/microscopes/microscope_base.py | e.g., 103, 189 | FILEMANAGER_GUARDIAN | `os.path.basename`, `os.path.join` | Path manipulation outside FileManager (Clause 17) |
| N5 | ezstitcher/microscopes/opera_phenix.py | e.g., 83, 99 | FILEMANAGER_GUARDIAN | `os.path.join`, `os.path.basename` | Path manipulation outside FileManager (Clause 17) |
| N6 | ezstitcher/core/pipeline/path_planner.py | e.g., 148 | FILEMANAGER_GUARDIAN | `Path(...)` conversions | Path manipulation outside FileManager (Clause 17) |
| N7 | ezstitcher/core/steps/function_step.py | e.g., 392 | FILEMANAGER_GUARDIAN | `Path(...)` path construction | Path manipulation outside FileManager (Clause 17) |
| N8 | ezstitcher/core/orchestrator/orchestrator.py | 84 | FILEMANAGER_GUARDIAN | `Path(self.plate_path)` | Path manipulation outside FileManager (Clause 17) |
| N9 | ezstitcher/ez/api.py | 12 | VIRTUALPATH_ENCAPSULATION | `from ezstitcher.io.virtual_path import VirtualPath` | Import of VirtualPath outside io.virtual_path (Clause 16) |
| N10 | ezstitcher/ez/utils.py | 55 | VIRTUALPATH_ENCAPSULATION | `from ezstitcher.io.virtual_path import VirtualPath` | Import of VirtualPath outside io.virtual_path (Clause 16) |
| N11 | ezstitcher/ez/core.py | 49 | VIRTUALPATH_ENCAPSULATION | `from ezstitcher.io.virtual_path import VirtualPath` | Import of VirtualPath outside io.virtual_path (Clause 16) |
| N12 | ezstitcher/io/storage/storage_factory.py | 11 | VIRTUALPATH_ENCAPSULATION | `from ezstitcher.io.virtual_path import VirtualPath` | Import of VirtualPath outside io.virtual_path (Clause 16) |
| N13 | ezstitcher/io/file_manager/file_manager_utils.py | 16 | VIRTUALPATH_ENCAPSULATION | `from ezstitcher.io.virtual_path import VirtualPath` | Import of VirtualPath outside io.virtual_path (Clause 16) |
| N14 | ezstitcher/io/image_io/image_io.py | 48, 98 | VIRTUALPATH_ENCAPSULATION | `from ezstitcher.io.virtual_path import VirtualPath` | Import of VirtualPath outside io.virtual_path (Clause 16) |
| N15 | ezstitcher/io/materialization_utils.py | 15 | VIRTUALPATH_ENCAPSULATION | `from ezstitcher.io.virtual_path import VirtualPath` | Import of VirtualPath outside io.virtual_path (Clause 16) |
| N16 | ezstitcher/microscopes/imagexpress.py | 19 | VIRTUALPATH_ENCAPSULATION | `from ezstitcher.io.virtual_path import VirtualPath` | Import of VirtualPath outside io.virtual_path (Clause 16) |
| N17 | ezstitcher/core/steps/function_step.py | e.g., constructor | STEPS_CONTEXT_FILEMANAGER | `def __init__(...)` | Constructor does not accept read_backend, write_backend (Clause 44) |
| N18 | ezstitcher/core/steps/specialized/norm_step.py | e.g., constructor | STEPS_CONTEXT_FILEMANAGER | `def __init__(...)` | Constructor does not accept read_backend, write_backend (Clause 44) |
| N19 | ezstitcher/core/steps/specialized/zflat_step.py | e.g., constructor | STEPS_CONTEXT_FILEMANAGER | `def __init__(...)` | Constructor does not accept read_backend, write_backend (Clause 44) |
| N20 | ezstitcher/core/steps/specialized/composite_step.py | e.g., constructor | STEPS_CONTEXT_FILEMANAGER | `def __init__(...)` | Constructor does not accept read_backend, write_backend (Clause 44) |
