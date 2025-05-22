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

Scan the codebase for violations of the following backend-related contracts:

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

9. EXCESSIVE_MODULARIZATION: Avoid creating helper methods for trivial operations:
   - Methods should encapsulate meaningful logical units, not just 1-2 lines of code
   - Private helper methods should provide clear abstraction benefits
   - Inline simple operations rather than creating dedicated methods
   - Exception: Thread-safe operations that require lock acquisition

For each violation, report:
1. File path
2. Line number
3. Violation type
4. Current code
5. Exact fix
```

### Precise Violation Matrix

| ID | File Path | Line | Violation Type | Current Code | Exact Fix |
|----|-----------|------|---------------|--------------|-----------|
| V16 | ezstitcher/core/steps/function_step.py | 200-210 | STEPS_CONTEXT_FILEMANAGER | `file_manager = FileManager()` | `file_manager = context.file_manager` |
| V17 | ezstitcher/core/steps/function_step.py | 250-260 | STEPS_CONTEXT_FILEMANAGER | `file_manager.list_files(input_dir)` | `file_manager.list_files(input_dir, read_backend)` |
| V19 | ezstitcher/core/steps/step_base.py | 30-40 | STEPS_CONTEXT_FILEMANAGER | `class Step:` | `class Step:\n    def __init__(self, *args, read_backend=None, write_backend=None, **kwargs):\n        self.read_backend = read_backend\n        self.write_backend = write_backend\n        super().__init__(*args, **kwargs)` |
| V12 | ezstitcher/core/orchestrator/orchestrator.py | 30-40 | REGISTRY_IN_ORCHESTRATOR | `self.file_manager: Optional[FileManager] = FileManager()` | `self.registry = BackendRegistry()\nself.file_manager: Optional[FileManager] = FileManager(registry=self.registry, isolation_mode=self.isolation_mode)` |
| V13 | ezstitcher/core/orchestrator/orchestrator.py | 50-60 | REGISTRY_IN_ORCHESTRATOR | Missing registry initialization method | `def initialize_registry(self):\n    """Initialize backend registry."""\n    # Register all available backends\n    self.registry.register_backend("disk", DiskStorageBackend)\n    self.registry.register_backend("memory", MemoryStorageBackend)\n    self.registry.register_backend("zarr", ZarrStorageBackend)` |
| V18 | ezstitcher/io/storage/storage_backend_registry.py | 10-20 | REGISTRY_IN_ORCHESTRATOR | Missing BackendRegistry class | `class BackendRegistry:\n    """Registry for backend classes and instances."""\n\n    def __init__(self):\n        self._backend_classes = {}\n        self._backend_instances = {}\n        self._isolated_instances = {}\n\n    def register_backend(self, name, backend_class):\n        """Register a backend class."""\n        self._backend_classes[name] = backend_class\n\n    def get_backend(self, name):\n        """Get a backend class by name."""\n        return self._backend_classes.get(name)\n\n    def get_backend_instance(self, name, isolation_mode=False):\n        """Get or create a backend instance by name.\n        \n        Args:\n            name: Backend name\n            isolation_mode: If True, returns a new instance each time\n                           If False, returns a shared singleton instance\n        """\n        if isolation_mode:\n            # In isolation mode, create a new instance each time\n            backend_class = self.get_backend(name)\n            if backend_class:\n                return backend_class()\n            return None\n            \n        # In shared mode, use singleton instances\n        if name not in self._backend_instances:\n            backend_class = self.get_backend(name)\n            if backend_class:\n                self._backend_instances[name] = backend_class()\n        return self._backend_instances.get(name)` |
| V1 | ezstitcher/formats/pattern/pattern_resolver.py | 219 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `all_patterns = _extract_patterns_from_data(well_patterns_data, file_manager)` | `all_patterns = _extract_patterns_from_data(well_patterns_data, file_manager, backend)` |
| V2 | ezstitcher/formats/pattern/pattern_resolver.py | 25-36 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `def auto_detect_patterns(self, directory: Union[str, Path], well_filter: List[str], variable_components: List[str], group_by: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:` | `def auto_detect_patterns(self, directory: Union[str, Path], well_filter: List[str], variable_components: List[str], backend: str, /, *, group_by: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:` |
| V3 | ezstitcher/formats/pattern/pattern_resolver.py | 40-46 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `def path_list_from_pattern(self, directory: Union[str, Path], pattern: str) -> List[Union[str, Path]]:` | `def path_list_from_pattern(self, directory: Union[str, Path], pattern: str, backend: str, /) -> List[Union[str, Path]]:` |
| V4 | ezstitcher/formats/pattern/pattern_resolver.py | 50-58 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `def list_files(self, directory: Union[str, Path], recursive: bool = False, pattern: Optional[str] = None, extensions: Optional[Set[str]] = None) -> List[Union[str, Path]]:` | `def list_files(self, directory: Union[str, Path], backend: str, /, *, recursive: bool = False, pattern: Optional[str] = None, extensions: Optional[Set[str]] = None) -> List[Union[str, Path]]:` |
| V5 | ezstitcher/formats/pattern/pattern_resolver.py | 60-62 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `def is_dir(self, path: Union[str, Path]) -> bool:` | `def is_dir(self, path: Union[str, Path], backend: str, /) -> bool:` |
| V6 | ezstitcher/formats/pattern/pattern_resolver.py | 76-85 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `def auto_detect_patterns(self, directory: Union[str, Path], well_filter: List[str], variable_components: List[str], group_by: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:` | `def auto_detect_patterns(self, directory: Union[str, Path], well_filter: List[str], variable_components: List[str], backend: str, /, *, group_by: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:` |
| V7 | ezstitcher/formats/pattern/pattern_resolver.py | 207-211 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `patterns_by_well = detector.auto_detect_patterns(directory, well_filter=[well], variable_components=variable_components, recursive=recursive)` | `patterns_by_well = detector.auto_detect_patterns(directory, [well], variable_components, backend, recursive=recursive)` |
| V8 | ezstitcher/formats/pattern/pattern_discovery.py | 300-310 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `def _generate_patterns_for_files(self, files: List[Any], variable_components: List[str]) -> List['PatternPath']:` | `def _generate_patterns_for_files(self, files: List[Any], variable_components: List[str], backend: str, /) -> List['PatternPath']:` |
| V9 | ezstitcher/formats/pattern/pattern_discovery.py | 350-360 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `patterns = self._generate_patterns_for_files(files, variable_components)` | `patterns = self._generate_patterns_for_files(files, variable_components, backend)` |
| V14 | ezstitcher/microscopes/microscope_base.py | 100-110 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `patterns_by_well = self.pattern_engine.auto_detect_patterns(folder_path, well_filter=well_filter, extensions=extensions, group_by=group_by, variable_components=variable_components)` | `patterns_by_well = self.pattern_engine.auto_detect_patterns(folder_path, well_filter, extensions, variable_components, backend, group_by=group_by)` |
| V15 | ezstitcher/microscopes/imagexpress.py | 150-160 | PATTERN_RESOLVER_BACKEND_AGNOSTIC | `htd_files = self.file_manager.list_files(plate_path, pattern="*.HTD")` | `htd_files = self.file_manager.list_files(plate_path, backend, pattern="*.HTD")` |
| V10 | ezstitcher/io/file_manager/file_manager_core.py | 50-60 | FILEMANAGER_GUARDIAN | `def __init__(self):` | `def __init__(self, registry=None, isolation_mode=False):` |
| V11 | ezstitcher/io/file_manager/file_manager_core.py | 100-110 | FILEMANAGER_GUARDIAN | `backend_instance = backend_class()` | `backend_instance = registry.get_backend_instance(backend_name, isolation_mode=self.isolation_mode) if registry else backend_class()` |
| V20 | ezstitcher/io/file_manager/file_manager_io.py | 50-60 | FILEMANAGER_GUARDIAN | `def read_image(self, file_path, backend):` | `def read_image(self, file_path, backend, /):\n    """Read image from file.\n    \n    Args:\n        file_path: Path to file\n        backend: Backend to use (positional-only)\n    \n    Note: This method only uses the passed backend parameter.\n    No internal resolution or default behavior is permitted.\n    """` |
| V24 | ezstitcher/formats/pattern/pattern_utils.py | 50-60 | BACKEND_PROPAGATION | `def process_pattern_helper(file_manager, path):\n    return file_manager.list_files(path)` | `def process_pattern_helper(file_manager, path, backend):\n    return file_manager.list_files(path, backend)` |
| V25 | ezstitcher/formats/pattern/pattern_utils.py | 70-80 | BACKEND_PROPAGATION | `def do_transform(file_manager, path, backend):\n    return process_pattern_helper(file_manager, path)` | `def do_transform(file_manager, path, backend):\n    return process_pattern_helper(file_manager, path, backend)` |
| V21 | ezstitcher/io/file_manager/file_manager_core.py | 200-210 | VIRTUALPATH_ENCAPSULATION | `def get_path(self, path: Union[str, Path, VirtualPath]) -> VirtualPath:` | `def get_path(self, path: Union[str, Path]) -> str:\n    """Convert a path to a standardized string path.\n    \n    Args:\n        path: Path to convert (str or Path)\n        \n    Returns:\n        Standardized string path\n    """\n    # Convert to VirtualPath internally but return a string\n    virtual_path = self._to_virtual_path(path, self.backend)\n    return str(virtual_path)` |
| V26 | ezstitcher/io/file_manager/file_manager_io.py | 100-110 | VIRTUALPATH_ENCAPSULATION | `def list_files(self, directory: Union[str, Path, VirtualPath], backend: str, recursive: bool = False) -> List[VirtualPath]:` | `def list_files(self, directory: Union[str, Path], backend: str, /, *, recursive: bool = False) -> List[str]:\n    """List files in a directory.\n    \n    Args:\n        directory: Directory to list files from\n        backend: Backend to use (positional-only)\n        recursive: Whether to list files recursively\n        \n    Returns:\n        List of standardized string paths\n    """\n    # Convert to VirtualPath internally\n    virtual_dir = self._to_virtual_path(directory, backend)\n    # Get virtual paths from backend\n    virtual_paths = self._get_backend(backend).list_files(virtual_dir, recursive=recursive)\n    # Convert back to strings\n    return [str(p) for p in virtual_paths]` |
| V27 | ezstitcher/core/pipeline/pipeline_compiler.py | 150-160 | BACKEND_DECLARATION_IN_PLANS | `def compile_step(self, step_config):\n    step_class = get_step_class(step_config['type'])\n    step_instance = step_class(**step_config['params'])` | `def compile_step(self, step_config):\n    step_class = get_step_class(step_config['type'])\n    # Ensure backend is explicitly declared in step config\n    if 'read_backend' not in step_config:\n        raise ValueError(f"Step {step_config['type']} missing required read_backend declaration")\n    if 'write_backend' not in step_config:\n        raise ValueError(f"Step {step_config['type']} missing required write_backend declaration")\n    # Pass backends explicitly to step instance\n    step_instance = step_class(\n        read_backend=step_config['read_backend'],\n        write_backend=step_config['write_backend'],\n        **step_config['params']\n    )` |
| V28 | ezstitcher/core/pipeline/pipeline_schema.py | 50-60 | BACKEND_DECLARATION_IN_PLANS | `STEP_SCHEMA = {\n    "type": "object",\n    "properties": {\n        "type": {"type": "string"},\n        "params": {"type": "object"}\n    },\n    "required": ["type", "params"]\n}` | `STEP_SCHEMA = {\n    "type": "object",\n    "properties": {\n        "type": {"type": "string"},\n        "read_backend": {"type": "string"},\n        "write_backend": {"type": "string"},\n        "params": {"type": "object"}\n    },\n    "required": ["type", "read_backend", "write_backend", "params"]\n}` |
| V29 | ezstitcher/core/steps/step_utils.py | 100-110 | FUNCTION_BACKEND_PROPAGATION | `def process_image(file_manager, image_path):\n    # Process image using file_manager\n    return file_manager.read_image(image_path)` | `def process_image(file_manager, image_path, backend):\n    # Process image using file_manager and backend\n    return file_manager.read_image(image_path, backend)` |
| V30 | ezstitcher/core/steps/step_utils.py | 120-130 | FUNCTION_BACKEND_PROPAGATION | `class ImageProcessor:\n    def __init__(self, file_manager):\n        self.file_manager = file_manager\n        \n    def process(self, image_path):\n        # Uses internal file_manager\n        return self.file_manager.read_image(image_path)` | `class ImageProcessor:\n    def __init__(self, file_manager, backend):\n        self.file_manager = file_manager\n        self.backend = backend\n        \n    def process(self, image_path):\n        # Pass backend to file_manager methods\n        return self.file_manager.read_image(image_path, self.backend)` |

### Implementation Strategy

1. **Create BackendRegistry Class with Isolation Mode**
   - Implement the BackendRegistry class in storage_backend_registry.py (V18)
   - Add methods for registering backends and getting backend instances
   - Implement isolation_mode to support both shared and isolated modes (Clause 308)
   - Ensure thread-safe implementation for cross-thread memory sharing
   - Implement static analysis tool for detecting backend= keyword usage (V22)
   - Implement static analysis tool for detecting backend propagation issues (V23)

2. **Update Protocol Definitions with Positional-Only Parameters**
   - Fix all protocol definitions in pattern_resolver.py to include backend parameter (V2-V6)
   - Add backend parameter to _generate_patterns_for_files method (V8)
   - Use Python's positional-only parameter syntax (/) to enforce Clause 306
   - Ensure consistent parameter order across all protocols

3. **Update FileManager for Registry and Isolation Mode**
   - Modify FileManager to accept registry parameter (V10)
   - Add isolation_mode parameter to FileManager constructor
   - Update backend initialization to use registry if provided (V11)
   - Ensure backward compatibility for existing code

4. **Move Registry to Orchestrator**
   - Add registry initialization to orchestrator (V12-V13)
   - Update FileManager creation to pass registry and isolation_mode
   - Ensure all backends are registered in the registry

5. **Fix Function Calls with Positional Backend Parameters (Clause 306)**
   - Update all function calls to pass backend parameter positionally (V1, V7, V9, V14-V17)
   - Remove any `backend=backend` keyword arguments at all call sites
   - Reorder parameters to ensure backend is passed in the correct position
   - Ensure backend parameter is propagated through all call chains
   - Implement static analysis to detect and prevent `backend=` keyword usage

6. **Enforce Function Backend Propagation (Clause 310)**
   - Update utility functions to accept and propagate backend (V29)
   - Update classes that hold file_manager to also hold backend (V30)
   - Ensure backend is passed to all downstream calls
   - Prevent storing backend as internal state without propagation
   - Prevent resolving backend from context, class attribute, or default

7. **Enforce Backend Declaration in Plans**
   - Update Step base class to include read_backend and write_backend (V19)
   - Update pipeline compiler to require explicit backend declarations (V27)
   - Update pipeline schema to require read_backend and write_backend (V28)
   - Ensure all steps declare their backend requirements
   - Make backend metadata available for traceability and planning
   - Prevent backend inference from memory type or runtime behavior

8. **Enforce VirtualPath Encapsulation**
   - Update FileManager public methods to accept and return standard path types (V21, V26)
   - Ensure VirtualPath objects are only used internally within the I/O layer
   - Prevent VirtualPath imports outside of the I/O layer
   - Convert VirtualPath objects to strings before returning from public methods

9. **Enforce Backend Method Purity**
   - Add docstrings to FileManager methods to enforce backend purity (V20)
   - Add validation to ensure backend parameter is used correctly

### Implementation Order

1. **Phase 1: Infrastructure**
   - Implement BackendRegistry class with isolation_mode (V18)
   - Update StorageBackend base class for mutation locking (V21)
   - Update FileManager to accept registry and isolation_mode (V10-V11)
   - Add registry initialization to orchestrator (V12-V13)
   - Implement static analysis tool for detecting backend= keyword usage (V22)
   - Implement static analysis tool for detecting backend propagation issues (V23)

2. **Phase 2: Protocol and Interface Updates**
   - Update PatternDetector protocol with positional-only backend (V2)
   - Update PathListProvider protocol with positional-only backend (V3)
   - Update DirectoryLister protocol with positional-only backend (V4-V5)
   - Update ManualRecursivePatternDetector protocol with positional-only backend (V6)
   - Update _generate_patterns_for_files method signature with positional-only backend (V8)
   - Update FileManager method signatures with positional-only backend (V20)
   - Update Step base class with backend metadata (V19)

3. **Phase 3: Function Call Updates**
   - Fix _extract_patterns_from_data call to pass backend positionally (V1)
   - Fix auto_detect_patterns call in get_patterns_for_well to pass backend positionally (V7)
   - Fix _generate_patterns_for_files call to pass backend positionally (V9)
   - Fix auto_detect_patterns call in microscope_base to pass backend positionally (V14)
   - Fix list_files call in imagexpress to pass backend positionally (V15)
   - Fix FileManager creation in steps (V16)
   - Fix list_files call in steps to pass backend positionally (V17)
   - Fix mid-tier helper functions for backend propagation (V24-V25)
   - Update utility functions to accept and propagate backend (V29)
   - Update classes that hold file_manager to also hold backend (V30)

4. **Phase 4: Backend Declaration in Plans**
   - Update pipeline schema to require read_backend and write_backend (V28)
   - Update pipeline compiler to require explicit backend declarations (V27)
   - Ensure all step configurations include explicit backend declarations
   - Add validation to prevent backend inference from memory type or runtime behavior

5. **Phase 5: VirtualPath Encapsulation**
   - Update FileManager get_path method to return strings instead of VirtualPath (V21)
   - Update FileManager list_files method to return strings instead of VirtualPath (V26)
   - Ensure all public FileManager methods hide VirtualPath implementation details
   - Add static analysis to prevent VirtualPath imports outside I/O layer

### Validation Checklist

After implementing fixes, verify:

1. **All I/O operations explicitly declare backend as positional-only parameter (Clause 306)**
   - No I/O function calls without backend parameter
   - All backend parameters are passed positionally, not as keywords
   - No `backend=backend` keyword arguments at any call site
   - No default backends or inferred backends
   - All function signatures use the `/` syntax to enforce positional-only parameters

2. **FileManager properly handles registry and isolation mode (Clause 308)**
   - FileManager accepts registry parameter and isolation_mode flag
   - FileManager uses registry for backend creation if provided
   - Registry supports both shared and isolated modes
   - Registry enables cross-thread memory sharing when in shared mode

3. **All FileManager methods use only the passed backend parameter (Clause 305)**
   - No internal backend resolution or default behavior
   - No fallback to default backends
   - All methods document that they only use the passed backend

4. **All functions accepting file_manager also accept and propagate backend (FUNCTION_BACKEND_PROPAGATION, Clause 310)**
   - No function calls file_manager methods without passing backend
   - Backend parameter is propagated through all call chains, including mid-tier helpers
   - No breaks in the backend propagation chain
   - Any function with file_manager parameter must also have backend parameter
   - Helper functions called by functions with file_manager must receive backend
   - Classes that hold file_manager must also hold backend
   - No storing backend as internal state without propagation
   - No resolving backend from context, class attribute, or default
   - Static analysis tool enforces complete propagation through nested chains

5. **All VirtualPath objects are properly encapsulated**
   - No VirtualPath objects exposed in FileManager public methods
   - All FileManager public methods accept and return standard path types
   - VirtualPath objects only used internally within the I/O layer
   - No VirtualPath imports outside of the I/O layer

6. **All steps declare read_backend and write_backend in metadata (BACKEND_DECLARATION_IN_PLANS)**
   - Step base class includes read_backend and write_backend fields
   - All step instances set these fields
   - Pipeline schema requires read_backend and write_backend
   - Pipeline compiler validates explicit backend declarations
   - Backend declarations are statically inspectable in plan files
   - No backend inference from memory type or runtime behavior
   - Orchestrator uses these fields for planning and traceability

7. **Pattern resolver and discovery are properly backend agnostic**
   - All methods accept backend parameter positionally
   - No default backends or inferred backends
   - Backend parameter is propagated to all downstream calls
