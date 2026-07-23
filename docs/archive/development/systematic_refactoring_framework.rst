Systematic Refactoring Framework
=================================

Authoritative guide for OpenHCS architectural decisions and refactoring approaches.

Core Architectural Philosophy
-----------------------------

Pragmatic OOP/FP Balance
~~~~~~~~~~~~~~~~~~~~~~~~~

**Use OOP for:**

- Contracts and interfaces (ABCs)
- Stateful systems (configuration, I/O, UI)
- Polymorphism (multiple implementations)
- Encapsulation (complex state management)

**Use FP for:**

- Data transformation (image processing, math operations)
- Configuration resolution (lazy evaluation, hierarchical chains)
- Validation logic (stateless functions)
- Utility functions (pure functions, no side effects)

Code Quality Principles
~~~~~~~~~~~~~~~~~~~~~~~

**Declarative, terse, elegant:**

.. code-block:: python

   # Good: Declarative enum-driven configuration
   class ProcessingContract(Enum):
       PURE_3D = "_execute_pure_3d"
       PURE_2D = "_execute_pure_2d"
       
       def execute(self, registry, func, image, *args, **kwargs):
           method = getattr(registry, self.value)
           return method(func, image, *args, **kwargs)

   # Bad: Imperative conditional logic
   def execute_contract(contract_type, registry, func, image, *args, **kwargs):
       if contract_type == "pure_3d":
           return registry._execute_pure_3d(func, image, *args, **kwargs)
       elif contract_type == "pure_2d":
           return registry._execute_pure_2d(func, image, *args, **kwargs)

**Strict separation of concerns:**

- Business logic: Domain operations isolated from framework
- I/O operations: All file operations through FileManager
- Configuration: Declarative dataclass-based, separate from logic
- UI logic: Framework-agnostic service layer with UI adapters

Fundamental Refactoring Principles
----------------------------------

Fail-Loud Philosophy
~~~~~~~~~~~~~~~~~~~~

Eliminate defensive programming. Let Python's exceptions bubble up:

.. code-block:: python

   # Forbidden: Defensive programming with silent failures
   if hasattr(obj, 'method'):
       result = obj.method()
   else:
       result = default_value  # Masks bugs

   # Forbidden: getattr with fallbacks
   result = getattr(obj, 'attribute', default_value)  # Masks missing attributes

   # Required: Let Python fail naturally
   def process_data(data: Array) -> Array:
       if data.ndim != 3:
           raise ValueError(f"Expected 3D array, got {data.ndim}D")
       return transform(data)

   # Required: Error handling ONLY where errors are expected
   try:
       result = gpu_operation(data)
   except CudaError as e:
       raise MemoryConversionError(
           source_type="cupy", target_type="torch",
           method="GPU_conversion", reason=str(e)
       ) from e

Stateless Architecture
~~~~~~~~~~~~~~~~~~~~~~

Prefer pure functions over stateful classes:

.. code-block:: python

   # Good: Pure function with explicit dependencies
   def validate_pipeline_config(config: PipelineConfig, 
                               available_functions: Set[str]) -> ValidationResult:
       errors = []
       for step in config.steps:
           if step.function_name not in available_functions:
               errors.append(f"Unknown function: {step.function_name}")
       return ValidationResult(errors)

   # Bad: Stateful validator with hidden dependencies
   class PipelineValidator:
       def __init__(self):
           self.function_registry = get_global_registry()  # Hidden dependency
       
       def validate(self, config):
           # Stateful validation logic

Dataclass Patterns
~~~~~~~~~~~~~~~~~~

Use dataclasses for declarative configuration:

.. code-block:: python

   @dataclass(frozen=True)
   class StepConfig:
       function_name: str
       parameters: Dict[str, Any]
       memory_type: MemoryType = MemoryType.NUMPY
       
       def validate(self) -> List[str]:
           errors = []
           if not self.function_name:
               errors.append("function_name required")
           return errors

ABC Contract Enforcement
~~~~~~~~~~~~~~~~~~~~~~~~

Use ABCs to enforce explicit contracts:

.. code-block:: python

   class StorageBackend(ABC):
       @abstractmethod
       def load(self, path: str) -> bytes: pass
       
       @abstractmethod
       def save(self, data: bytes, path: str) -> None: pass

   class FileSystemBackend(StorageBackend):
       def load(self, path: str) -> bytes:
           with open(path, 'rb') as f:
               return f.read()
       
       def save(self, data: bytes, path: str) -> None:
           with open(path, 'wb') as f:
               f.write(data)

Enum-Driven Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Replace magic strings with enums:

.. code-block:: python

   # Good: Enum-driven behavior
   class MemoryType(Enum):
       NUMPY = "numpy"
       TORCH = "torch"
       CUPY = "cupy"
       
       def convert_array(self, array, target_type: 'MemoryType'):
           converter = getattr(self, f"_to_{target_type.value}")
           return converter(array)

   # Bad: Magic strings
   def convert_array(array, source_type: str, target_type: str):
       if source_type == "numpy" and target_type == "torch":
           return torch.from_numpy(array)
       elif source_type == "torch" and target_type == "numpy":
           return array.numpy()

OpenHCS-Specific Patterns
-------------------------

Lazy Configuration Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hierarchical resolution: step → pipeline → global:

.. code-block:: python

   def resolve_field_value(field_name: str, 
                          step_config: Optional[StepConfig],
                          pipeline_config: Optional[PipelineConfig],
                          global_config: GlobalConfig) -> Any:
       # Breadth-first resolution
       for config in [step_config, pipeline_config, global_config]:
           if config and hasattr(config, field_name):
               value = getattr(config, field_name)
               if value is not None:
                   return value
       return None

FileManager I/O Abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All I/O operations must go through FileManager:

.. code-block:: python

   # Good: FileManager abstraction
   def save_results(results: List[Array], 
                   output_paths: List[str], 
                   filemanager: FileManager) -> None:
       for result, path in zip(results, output_paths):
           filemanager.save_array(result, path)

   # Bad: Direct file system access
   def save_results(results: List[Array], output_paths: List[str]) -> None:
       for result, path in zip(results, output_paths):
           np.save(path, result)  # Bypasses backend system

3D→3D Function Contracts
~~~~~~~~~~~~~~~~~~~~~~~~

All OpenHCS functions maintain 3D→3D contracts:

.. code-block:: python

   @register_function("denoise_3d")
   def denoise_volume(volume: Array3D) -> Array3D:
       """Denoise 3D volume. Input and output must be 3D."""
       if volume.ndim != 3:
           raise ValueError(f"Expected 3D volume, got {volume.ndim}D")
       
       denoised = apply_denoising(volume)
       
       if denoised.ndim != 3:
           raise ValueError(f"Function violated 3D→3D contract")
       return denoised

Systematic Refactoring Process
------------------------------

Step 1: Identify Violations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code smells to look for:

- ``hasattr()`` checks with fallback logic
- ``getattr()`` calls with default values
- ``try/except`` blocks where errors aren't expected
- Magic strings instead of enums
- Direct file system access bypassing FileManager
- Defensive programming patterns
- Hardcoded lists that should use enums
- Stateful classes that could be pure functions

Step 2: Apply Patterns
~~~~~~~~~~~~~~~~~~~~~~

1. **Create ABCs** for similar functionality
2. **Extract dependencies** to constructor parameters
3. **Remove dispatch layers** in favor of direct method calls
4. **Use Generic[T]** and metaprogramming for true genericism
5. **Replace defensive code** with explicit error handling
6. **Standardize interfaces** across subsystems

Step 3: Validate Changes
~~~~~~~~~~~~~~~~~~~~~~~~

Refactoring validation checklist:

- [ ] All I/O operations go through FileManager
- [ ] No defensive programming patterns remain
- [ ] Error handling only where errors are expected
- [ ] Enums used instead of magic strings
- [ ] ABCs define clear contracts
- [ ] Functions maintain 3D→3D contracts where applicable
- [ ] Breadth-first traversal used for recursive operations
- [ ] Lazy resolution follows step → pipeline → global hierarchy

Decision Framework
------------------

When to Refactor
~~~~~~~~~~~~~~~~

**Refactor when:**

- Code violates OpenHCS architectural principles
- Defensive programming patterns are present
- Magic strings are used instead of enums
- Similar functionality lacks consistent interfaces
- Direct file system access bypasses FileManager

**Don't refactor when:**

- Code already follows OpenHCS patterns
- Changes would break existing contracts
- Refactoring adds complexity without clear benefit

Architectural Decision Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Identify the domain** - Configuration, I/O, processing, or UI?
2. **Choose paradigm** - OOP for contracts/state, FP for transformations
3. **Apply patterns** - Use established OpenHCS patterns for the domain
4. **Validate design** - Ensure fail-loud behavior and clear contracts
5. **Test integration** - Verify compatibility with existing systems

This framework ensures consistent, maintainable, and robust code across the OpenHCS codebase.
