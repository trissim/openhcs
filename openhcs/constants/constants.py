"""
Consolidated constants for OpenHCS.

This module defines all constants related to backends, defaults, I/O, memory, and pipeline.
These constants are governed by various doctrinal clauses.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Set, TypeVar

class VariableComponents(Enum):
    SITE = "site"
    CHANNEL = "channel"
    Z_INDEX = "z_index"
    WELL = "well"

class Microscope(Enum):
    AUTO = "auto"
    IMAGEXPRESS = "ImageXpress"
    OPERAPHENIX = "OperaPhenix"

class GroupBy(Enum):
    CHANNEL = VariableComponents.CHANNEL.value # Will be "channel"
    Z_INDEX = VariableComponents.Z_INDEX.value # Will be "z_index"
    SITE = VariableComponents.SITE.value     # Will be "site"
    WELL = VariableComponents.WELL.value     # Will be "well"
    NONE = "" # Added for allow_blank in Select

# I/O-related constants
DEFAULT_IMAGE_EXTENSION = ".tif"
DEFAULT_IMAGE_EXTENSIONS: Set[str] = {".tif", ".tiff", ".TIF", ".TIFF"}
DEFAULT_SITE_PADDING = 3
DEFAULT_RECURSIVE_PATTERN_SEARCH = False
DEFAULT_VARIABLE_COMPONENTS: VariableComponents = VariableComponents.SITE
DEFAULT_GROUP_BY: GroupBy = GroupBy.CHANNEL
DEFAULT_MICROSCOPE: Microscope = Microscope.AUTO

# Backend-related constants
class Backend(Enum):
    DISK = "disk"
    MEMORY = "memory"
    ZARR = "zarr"

class FileFormat(Enum):
    TIFF = list(DEFAULT_IMAGE_EXTENSIONS)
    NUMPY = [".npy"]
    TORCH = [".pt", ".torch", ".pth"]
    JAX = [".jax"]
    CUPY = [".cupy",".craw"]
    TENSORFLOW = [".tf"]
    TEXT = [".txt",".csv",".json",".py",".md"]

DEFAULT_BACKEND = Backend.MEMORY
REQUIRES_DISK_READ = "requires_disk_read"
REQUIRES_DISK_WRITE = "requires_disk_write"
FORCE_DISK_WRITE = "force_disk_write"
READ_BACKEND = "read_backend"
WRITE_BACKEND = "write_backend"

# Default values
DEFAULT_TILE_OVERLAP = 10.0
DEFAULT_MAX_SHIFT = 50
DEFAULT_MARGIN_RATIO = 0.1
DEFAULT_PIXEL_SIZE = 1.0
DEFAULT_ASSEMBLER_LOG_LEVEL = "INFO"
DEFAULT_INTERPOLATION_MODE = "nearest"
DEFAULT_INTERPOLATION_ORDER = 1
DEFAULT_CPU_THREAD_COUNT = 4
DEFAULT_PATCH_SIZE = 128
DEFAULT_SEARCH_RADIUS = 20
# Consolidated definition for CPU thread count


# Memory-related constants
T = TypeVar('T')
ConversionFunc = Callable[[Any], Any]

class MemoryType(Enum):
    NUMPY = "numpy"
    CUPY = "cupy"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    PYCLESPERANTO = "pyclesperanto"

CPU_MEMORY_TYPES: Set[MemoryType] = {MemoryType.NUMPY}
GPU_MEMORY_TYPES: Set[MemoryType] = {
    MemoryType.CUPY,
    MemoryType.TORCH,
    MemoryType.TENSORFLOW,
    MemoryType.JAX,
    MemoryType.PYCLESPERANTO
}
SUPPORTED_MEMORY_TYPES: Set[MemoryType] = CPU_MEMORY_TYPES | GPU_MEMORY_TYPES

VALID_MEMORY_TYPES = {mt.value for mt in MemoryType}
VALID_GPU_MEMORY_TYPES = {mt.value for mt in GPU_MEMORY_TYPES}

# Memory type constants for direct access
MEMORY_TYPE_NUMPY = MemoryType.NUMPY.value
MEMORY_TYPE_CUPY = MemoryType.CUPY.value
MEMORY_TYPE_TORCH = MemoryType.TORCH.value
MEMORY_TYPE_TENSORFLOW = MemoryType.TENSORFLOW.value
MEMORY_TYPE_JAX = MemoryType.JAX.value
MEMORY_TYPE_PYCLESPERANTO = MemoryType.PYCLESPERANTO.value

DEFAULT_NUM_WORKERS = 1
# Consolidated definition for number of workers
DEFAULT_OUT_DIR_SUFFIX = "_out"
DEFAULT_POSITIONS_DIR_SUFFIX = "_positions"
DEFAULT_STITCHED_DIR_SUFFIX = "_stitched"
DEFAULT_WORKSPACE_DIR_SUFFIX = "_workspace"