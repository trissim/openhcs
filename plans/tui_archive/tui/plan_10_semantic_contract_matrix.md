# plan_10_semantic_contract_matrix.md
## Component: OpenHCS Semantic Contract Matrix

### Objective
Create a comprehensive semantic matrix of contracts in the OpenHCS architecture, with deterministic rules to flag violations, focusing on backend handling, parameter propagation, and interface consistency.

### Semantic Contract Matrix

| Contract ID | Contract Name | Contract Description | Violation Patterns | Deterministic Detection Rules | Severity |
|-------------|---------------|----------------------|-------------------|------------------------------|----------|
| C1 | Explicit Backend Declaration | Every I/O operation must explicitly declare its storage backend | Missing backend parameter; Default backend fallback; Inferred backend from context | `function_requires_io AND NOT has_backend_param` | Critical |
| C2 | Backend Parameter Propagation | Backend parameter must be propagated through all call chains | Missing backend in intermediate calls; Dropped backend parameters | `calls_io_function AND NOT passes_backend_param` | Critical |
| C3 | Backend Validation | Every function accepting a backend must validate it | Missing validation; Assumed backend validity | `has_backend_param AND NOT validates_backend` | High |
| C4 | Protocol-Implementation Consistency | Protocol definitions must match implementations | Protocol missing backend; Implementation requiring backend | `protocol_method_signature != implementation_method_signature` | Critical |
| C5 | Documentation Completeness | Documentation must reflect all required parameters | Missing backend in docs; Incomplete parameter descriptions | `has_backend_param AND NOT documented_backend_param` | Medium |
| C6 | Parameter Order Consistency | Parameter order must be consistent across related functions | Backend in different positions; Inconsistent parameter ordering | `param_position('backend', func1) != param_position('backend', func2)` | Medium |
| C7 | No Default Backends | No function should use default backends | Fallback to default backend; Implicit backend selection | `has_default_backend OR has_backend_fallback` | High |
| C8 | Explicit Failure | Functions must fail explicitly when backend is invalid | Silent failures; Fallback behavior | `handles_backend_error AND NOT raises_explicit_error` | High |
| C9 | Backend Type Safety | Backend parameters must have explicit type annotations | Missing type annotations; Loose typing (Any, object) | `has_backend_param AND NOT has_backend_type_annotation` | Medium |
| C10 | Backend Immutability | Backend must not be modified after initialization | Backend mutation; Backend reconfiguration | `modifies_backend_after_init` | High |
| C11 | Backend Lifecycle Management | Backend must be properly initialized and closed | Missing initialization; Resource leaks | `uses_backend AND (NOT initializes_backend OR NOT closes_backend)` | High |
| C12 | No Backend Inference | No function should infer backend capabilities | Runtime capability checking; hasattr/getattr on backend | `uses_hasattr_on_backend OR uses_getattr_on_backend` | Critical |
| C13 | Backend Capability Declaration | Backend capabilities must be declared explicitly | Missing capability declaration; Runtime capability discovery | `uses_backend_capability AND NOT declares_capability_requirement` | High |
| C14 | Backend Isolation | Functions should not mix backends | Multiple backends in same function; Backend conversion | `uses_multiple_backends_in_function` | Medium |
| C15 | Backend Serialization | Backend references must be serializable | Non-serializable backends; Backend closures | `backend_contains_non_serializable_state` | Medium |

### Detection Patterns

#### D1: Function Requires I/O
```python
def function_requires_io(func):
    """Determine if a function requires I/O operations."""
    # Check if function calls any I/O functions
    io_functions = [
        'read', 'write', 'open', 'close', 'list_files', 'list_image_files',
        'file_exists', 'is_dir', 'get_standard_path', 'auto_detect_patterns',
        'path_list_from_pattern'
    ]
    
    # Check if function has I/O-related parameters
    io_param_names = [
        'file', 'path', 'directory', 'folder', 'input_dir', 'output_dir',
        'file_manager', 'backend'
    ]
    
    # Check function body for I/O operations
    io_operations = [
        'open(', 'read(', 'write(', 'close(', 'listdir(', 'glob(',
        'file_manager.', 'backend.'
    ]
    
    return (
        any(io_func in get_function_calls(func) for io_func in io_functions) or
        any(io_param in get_function_params(func) for io_param in io_param_names) or
        any(io_op in get_function_body(func) for io_op in io_operations)
    )
```

#### D2: Has Backend Parameter
```python
def has_backend_param(func):
    """Check if a function has a backend parameter."""
    params = get_function_params(func)
    return 'backend' in params
```

#### D3: Calls I/O Function
```python
def calls_io_function(func):
    """Check if a function calls any I/O functions."""
    io_functions = [
        'read', 'write', 'open', 'close', 'list_files', 'list_image_files',
        'file_exists', 'is_dir', 'get_standard_path', 'auto_detect_patterns',
        'path_list_from_pattern'
    ]
    
    function_calls = get_function_calls(func)
    return any(io_func in function_calls for io_func in io_functions)
```

#### D4: Passes Backend Parameter
```python
def passes_backend_param(func, call):
    """Check if a function call passes a backend parameter."""
    # Check for explicit backend parameter
    if 'backend=' in call:
        return True
    
    # Check for positional backend parameter
    func_def = get_function_definition(get_function_name(call))
    if not func_def:
        return False
    
    params = get_function_params(func_def)
    if 'backend' not in params:
        return False
    
    backend_position = params.index('backend')
    call_args = get_call_args(call)
    
    return len(call_args) > backend_position
```

#### D5: Validates Backend
```python
def validates_backend(func):
    """Check if a function validates its backend parameter."""
    validation_patterns = [
        'isinstance(backend,', 'if backend is None:', 'if not backend:',
        'if backend not in', 'backend_instance = file_manager._get_backend(backend)',
        'assert isinstance(backend'
    ]
    
    function_body = get_function_body(func)
    return any(pattern in function_body for pattern in validation_patterns)
```

#### D6: Has Default Backend
```python
def has_default_backend(func):
    """Check if a function has a default backend."""
    default_patterns = [
        'backend = backend or', 'backend = backend if backend else',
        'backend = get_default_backend()', 'backend = DEFAULT_BACKEND',
        'backend: str = DEFAULT_BACKEND', 'backend=DEFAULT_BACKEND',
        'backend = None'
    ]
    
    function_body = get_function_body(func)
    function_def = get_function_definition(func)
    
    return (
        any(pattern in function_body for pattern in default_patterns) or
        'backend=None' in function_def or
        'backend = None' in function_def
    )
```

#### D7: Has Backend Fallback
```python
def has_backend_fallback(func):
    """Check if a function has backend fallback logic."""
    fallback_patterns = [
        'except: backend =', 'fallback_to_cpu', 'if backend is None:',
        'if not backend:', 'backend = backend or', 'backend = backend if backend else',
        'try: backend.', 'except: use_default_backend'
    ]
    
    function_body = get_function_body(func)
    return any(pattern in function_body for pattern in fallback_patterns)
```

#### D8: Handles Backend Error
```python
def handles_backend_error(func):
    """Check if a function handles backend errors."""
    error_handling_patterns = [
        'try:', 'except', 'if backend is None:', 'if not backend:',
        'if backend not in', 'if not isinstance(backend,'
    ]
    
    function_body = get_function_body(func)
    return any(pattern in function_body for pattern in error_handling_patterns)
```

#### D9: Raises Explicit Error
```python
def raises_explicit_error(func):
    """Check if a function raises explicit errors for backend issues."""
    error_raising_patterns = [
        'raise ValueError', 'raise TypeError', 'raise RuntimeError',
        'raise Exception', 'raise BackendError'
    ]
    
    function_body = get_function_body(func)
    return any(pattern in function_body for pattern in error_raising_patterns)
```

#### D10: Has Backend Type Annotation
```python
def has_backend_type_annotation(func):
    """Check if a function has a type annotation for its backend parameter."""
    function_def = get_function_definition(func)
    
    # Check for backend: str, backend: StorageBackend, etc.
    return 'backend:' in function_def and not 'backend: Any' in function_def
```

### Violation Matrix

| File Pattern | Contract Violations | Detection Rules | Fix Pattern |
|--------------|---------------------|-----------------|-------------|
| `**/pattern_resolver.py` | C1, C2, C4 | `function_requires_io AND NOT has_backend_param` | Add backend parameter to all I/O functions |
| `**/pattern_discovery.py` | C2, C3, C8 | `calls_io_function AND NOT passes_backend_param` | Propagate backend parameter to all I/O calls |
| `**/microscope_*.py` | C4, C6, C9 | `protocol_method_signature != implementation_method_signature` | Align protocol and implementation signatures |
| `**/file_manager*.py` | C3, C7, C8 | `has_backend_param AND NOT validates_backend` | Add backend validation to all functions |
| `**/steps/*.py` | C2, C12, C13 | `uses_backend_capability AND NOT declares_capability_requirement` | Declare backend capability requirements |
| `**/pipeline/*.py` | C2, C10, C11 | `uses_backend AND (NOT initializes_backend OR NOT closes_backend)` | Ensure proper backend lifecycle management |
| `**/io/*.py` | C7, C12, C14 | `uses_hasattr_on_backend OR uses_getattr_on_backend` | Replace runtime checks with static declarations |
| `**/docs/**/*.rst` | C5 | `has_backend_param AND NOT documented_backend_param` | Update documentation to include backend parameter |

### Implementation Strategy

1. **Fix Protocol Definitions First**
   - Update all protocol definitions to include backend parameter
   - Ensure consistent parameter order across all protocols
   - Add proper type annotations for backend parameters

2. **Fix Implementation Signatures**
   - Align implementation signatures with protocol definitions
   - Ensure consistent parameter order across all implementations
   - Add proper type annotations for backend parameters

3. **Fix Function Calls**
   - Update all function calls to pass backend parameter
   - Ensure backend parameter is propagated through call chains
   - Remove any default backend fallbacks

4. **Add Validation**
   - Add backend validation to all functions accepting backend parameter
   - Ensure explicit failure when backend is invalid or missing
   - Remove any silent failures or fallback behavior

5. **Update Documentation**
   - Update all documentation to include backend parameter
   - Ensure consistent parameter descriptions across all documentation
   - Add examples showing proper backend usage

### Automated Detection Tool

```python
def scan_for_violations(codebase_path):
    """Scan codebase for contract violations."""
    violations = []
    
    for file_path in get_python_files(codebase_path):
        for func in get_functions(file_path):
            # Check for Contract 1: Explicit Backend Declaration
            if function_requires_io(func) and not has_backend_param(func):
                violations.append({
                    'file': file_path,
                    'function': get_function_name(func),
                    'contract': 'C1',
                    'description': 'Function requires I/O but has no backend parameter'
                })
            
            # Check for Contract 2: Backend Parameter Propagation
            for call in get_function_calls(func):
                if calls_io_function(call) and not passes_backend_param(func, call):
                    violations.append({
                        'file': file_path,
                        'function': get_function_name(func),
                        'contract': 'C2',
                        'description': f'Function calls I/O function {call} but does not pass backend parameter'
                    })
            
            # Check for Contract 3: Backend Validation
            if has_backend_param(func) and not validates_backend(func):
                violations.append({
                    'file': file_path,
                    'function': get_function_name(func),
                    'contract': 'C3',
                    'description': 'Function accepts backend parameter but does not validate it'
                })
            
            # Continue with other contract checks...
    
    return violations
```
