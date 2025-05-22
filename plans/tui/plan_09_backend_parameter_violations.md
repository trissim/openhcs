# plan_09_backend_parameter_violations.md
## Component: Backend Parameter Interface Fraud Audit

### Objective
Document all instances of interface fraud related to the backend parameter in the codebase, focusing on functions that don't respect their contract by failing to pass required backend parameters or failing to fail loudly when backend parameters are missing.

### Findings

The OpenHCS architecture requires explicit backend declaration with no fallback logic or inferred capabilities. This audit identifies violations of Clause 42 (Ambiguity Resolution), Clause 88 (No Inferred Capabilities), and Clause 92 (No Interface Fraud) related to backend parameters.

#### Pattern 1: Missing Backend Parameter in Function Calls

| ID | File | Line | Function Call | Violation |
|----|------|------|--------------|-----------|
| P1-01 | ezstitcher/formats/pattern/pattern_resolver.py | ~225 | `all_patterns = _extract_patterns_from_data(well_patterns_data, file_manager)` | Missing required `backend` parameter in call to `_extract_patterns_from_data` |
| P1-02 | ezstitcher/formats/pattern/pattern_resolver.py | ~250 | `detector.auto_detect_patterns(directory, well_filter=[well], variable_components=variable_components, recursive=recursive)` | Missing required `backend` parameter in call to `auto_detect_patterns` |
| P1-03 | ezstitcher/formats/pattern/pattern_resolver.py | ~251 | `detector.file_manager.list_files(current_dir, recursive=False)` | Missing required `backend` parameter in call to `list_files` |
| P1-04 | ezstitcher/formats/pattern/pattern_resolver.py | ~252 | `detector.file_manager.is_dir(d)` | Missing required `backend` parameter in call to `is_dir` |

#### Pattern 2: Incorrect Protocol Definition Missing Backend Parameter

| ID | File | Line | Protocol Definition | Violation |
|----|------|------|---------------------|-----------|
| P2-01 | ezstitcher/formats/pattern/pattern_resolver.py | ~29 | `def auto_detect_patterns(self, directory: Union[str, Path], well_filter: List[str], variable_components: List[str], group_by: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:` | Protocol definition missing required `backend` parameter |
| P2-02 | ezstitcher/formats/pattern/pattern_resolver.py | ~43 | `def path_list_from_pattern(self, directory: Union[str, Path], pattern: str) -> List[Union[str, Path]]:` | Protocol definition missing required `backend` parameter |
| P2-03 | ezstitcher/formats/pattern/pattern_resolver.py | ~54 | `def list_files(self, directory: Union[str, Path], recursive: bool = False, pattern: Optional[str] = None, extensions: Optional[Set[str]] = None) -> List[Union[str, Path]]:` | Protocol definition missing required `backend` parameter |
| P2-04 | ezstitcher/formats/pattern/pattern_resolver.py | ~64 | `def is_dir(self, path: Union[str, Path]) -> bool:` | Protocol definition missing required `backend` parameter |
| P2-05 | ezstitcher/formats/pattern/pattern_resolver.py | ~80 | `def auto_detect_patterns(self, directory: Union[str, Path], well_filter: List[str], variable_components: List[str], group_by: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:` | Protocol definition missing required `backend` parameter |

#### Pattern 3: Documentation Missing Backend Parameter

| ID | File | Line | Documentation | Violation |
|----|------|------|---------------|-----------|
| P3-01 | docs/source/api/microscope_interfaces.rst | ~15 | `.. py:method:: auto_detect_patterns(folder_path, well_filter=None, extensions=None, group_by='channel', variable_components=None)` | Documentation missing required `backend` parameter |
| P3-02 | docs/source/api/microscope_interfaces.rst | ~40 | `.. py:method:: auto_detect_patterns(folder_path, well_filter=None, extensions=None, group_by='channel', variable_components=None)` | Documentation missing required `backend` parameter |

#### Pattern 4: Missing Backend Validation

| ID | File | Line | Code | Violation |
|----|------|------|------|-----------|
| P4-01 | ezstitcher/formats/pattern/pattern_discovery.py | ~300 | `def _generate_patterns_for_files(self, files: List[Any], variable_components: List[str]) -> List['PatternPath']:` | Method doesn't validate or use backend parameter for file operations |

#### Pattern 5: Inconsistent Backend Parameter Position

| ID | File | Line | Function Signature | Violation |
|----|------|------|-------------------|-----------|
| P5-01 | ezstitcher/formats/pattern/pattern_resolver.py | ~157 | `def get_patterns_for_well(well: str, directory: Union[str, Path], detector: PatternDetector, variable_components: List[str], file_manager: FileManager, backend: str, recursive: bool = False) -> List[str]:` | Inconsistent parameter order - `backend` should be after `file_manager` |
| P5-02 | ezstitcher/microscopes/microscope_base.py | ~150 | `def auto_detect_patterns(self, folder_path: Union[str, Path], backend: str, well_filter=None, extensions=None, group_by='channel', variable_components=None):` | Inconsistent parameter order - `backend` should be after other required parameters |

### Violation Patterns and Root Causes

1. **Missing Backend Parameter**
   - Root Cause: Functions call other functions without passing the required backend parameter
   - Impact: Runtime errors or silent failures when backend is needed but not provided
   - Fix Pattern: Add backend parameter to all function calls

2. **Incorrect Protocol Definitions**
   - Root Cause: Protocol definitions don't include backend parameter in their signatures
   - Impact: Type checking won't catch missing backend parameters
   - Fix Pattern: Update all protocol definitions to include backend parameter

3. **Documentation Gaps**
   - Root Cause: Documentation doesn't mention backend parameter
   - Impact: Developers won't know backend parameter is required
   - Fix Pattern: Update all documentation to include backend parameter

4. **Missing Backend Validation**
   - Root Cause: Functions don't validate backend parameter
   - Impact: Silent failures when backend is invalid
   - Fix Pattern: Add validation for backend parameter in all functions

5. **Inconsistent Parameter Order**
   - Root Cause: Inconsistent parameter order across functions
   - Impact: Confusion and potential errors when calling functions
   - Fix Pattern: Standardize parameter order across all functions

### Fix Matrix

| Violation Pattern | Fix Strategy | Priority |
|-------------------|--------------|----------|
| Missing Backend Parameter | Add backend parameter to all function calls | High |
| Incorrect Protocol Definitions | Update all protocol definitions to include backend parameter | High |
| Documentation Gaps | Update all documentation to include backend parameter | Medium |
| Missing Backend Validation | Add validation for backend parameter in all functions | High |
| Inconsistent Parameter Order | Standardize parameter order across all functions | Medium |

### Implementation Plan

1. **Update Protocol Definitions**
   - Update all protocol definitions to include backend parameter
   - Ensure consistent parameter order across all protocols

2. **Fix Function Calls**
   - Add backend parameter to all function calls
   - Ensure backend parameter is passed correctly

3. **Update Documentation**
   - Update all documentation to include backend parameter
   - Ensure consistent parameter order in documentation

4. **Add Backend Validation**
   - Add validation for backend parameter in all functions
   - Ensure consistent validation pattern across all functions

5. **Standardize Parameter Order**
   - Standardize parameter order across all functions
   - Ensure backend parameter is in consistent position

### Implementation Draft for Key Fixes

```python
# Update in ezstitcher/formats/pattern/pattern_resolver.py
class PatternDetector(Protocol):
    """Protocol compatible with MicroscopeHandler and PatternDiscoveryEngine."""

    def auto_detect_patterns(
        self,
        directory: Union[str, Path],
        well_filter: List[str],
        variable_components: List[str],
        backend: str,  # Added backend parameter
        group_by: Optional[str] = None,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """Detect patterns in the given directory."""
        ...

class PathListProvider(Protocol):
    """Protocol for objects that can list paths from a pattern."""
    def path_list_from_pattern(
        self,
        directory: Union[str, Path],
        pattern: str,
        backend: str  # Added backend parameter
    ) -> List[Union[str, Path]]:
        """List paths matching a pattern in a directory."""
        ...

class DirectoryLister(Protocol):
    """Protocol for objects that can list files in a directory."""
    def list_files(
        self,
        directory: Union[str, Path],
        backend: str,  # Added backend parameter
        recursive: bool = False,
        pattern: Optional[str] = None,
        extensions: Optional[Set[str]] = None
    ) -> List[Union[str, Path]]:
        """List files in a directory."""
        ...
    
    def is_dir(
        self,
        path: Union[str, Path],
        backend: str  # Added backend parameter
    ) -> bool:
        """Check if a path is a directory."""
        ...
```

```python
# Update in ezstitcher/formats/pattern/pattern_resolver.py
# Fix function call to pass backend parameter
patterns_by_well = detector.auto_detect_patterns(
    directory,
    well_filter=[well],
    variable_components=variable_components,
    backend=backend,  # Added backend parameter
    recursive=recursive
)

# Fix function call to _extract_patterns_from_data
all_patterns = _extract_patterns_from_data(
    well_patterns_data,
    file_manager,
    backend  # Added backend parameter
)

# Fix function calls to list_files and is_dir
subdirs = [
    d for d in detector.file_manager.list_files(
        current_dir,
        backend=backend,  # Added backend parameter
        recursive=False
    )
    if detector.file_manager.is_dir(d, backend=backend)  # Added backend parameter
]
```
