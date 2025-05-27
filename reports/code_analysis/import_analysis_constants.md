# Import Analysis

Found 3 files with import issues:

## openhcs/constants/__init__.py

### Unused Imports
These symbols are imported but not used:
```
- Backend
- CPU_MEMORY_TYPES
- Clause
- DEFAULT_ASSEMBLER_LOG_LEVEL
- DEFAULT_BACKEND
- DEFAULT_CPU_THREAD_COUNT
- DEFAULT_GROUP_BY
- DEFAULT_IMAGE_EXTENSION
- DEFAULT_IMAGE_EXTENSIONS
- DEFAULT_INTERPOLATION_MODE
- DEFAULT_INTERPOLATION_ORDER
- DEFAULT_MARGIN_RATIO
- DEFAULT_MAX_SHIFT
- DEFAULT_MICROSCOPE
- DEFAULT_NUM_WORKERS
- DEFAULT_OUT_DIR_SUFFIX
- DEFAULT_PIXEL_SIZE
- DEFAULT_POSITIONS_DIR_SUFFIX
- DEFAULT_RECURSIVE_PATTERN_SEARCH
- DEFAULT_SITE_PADDING
- DEFAULT_STITCHED_DIR_SUFFIX
- DEFAULT_TILE_OVERLAP
- DEFAULT_VARIABLE_COMPONENTS
- FORCE_DISK_WRITE
- GPU_MEMORY_TYPES
- GroupBy
- MEMORY_TYPE_CUPY
- MEMORY_TYPE_JAX
- MEMORY_TYPE_NUMPY
- MEMORY_TYPE_TENSORFLOW
- MEMORY_TYPE_TORCH
- MemoryType
- Microscope
- READ_BACKEND
- REQUIRES_DISK_READ
- REQUIRES_DISK_WRITE
- SUPPORTED_MEMORY_TYPES
- VALID_GPU_MEMORY_TYPES
- VALID_MEMORY_TYPES
- VariableComponents
- WRITE_BACKEND
```

## openhcs/constants/clauses.py

### Missing Imports
These symbols are used but not imported:
```
- Clause
- clause_numbers
- self
```

## openhcs/constants/constants.py

### Missing Imports
These symbols are used but not imported:
```
- Backend
- CPU_MEMORY_TYPES
- DEFAULT_IMAGE_EXTENSIONS
- GPU_MEMORY_TYPES
- GroupBy
- MemoryType
- Microscope
- VariableComponents
- mt
```

### Unused Imports
These symbols are imported but not used:
```
- Dict
- List
```
