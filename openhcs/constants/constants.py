"""
Consolidated constants for OpenHCS.

This module defines all constants related to backends, defaults, I/O, memory, and pipeline.
These constants are governed by various doctrinal clauses.

Caching:
- Component enums (AllComponents, VariableComponents, GroupBy) are cached persistently
- Cache invalidated on OpenHCS version change or after 7 days
- Provides ~20x speedup on subsequent runs and in subprocesses
"""

from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Set, TypeVar, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Microscope(Enum):
    AUTO = "auto"
    OPENHCS = "openhcs"  # Added for the OpenHCS pre-processed format
    IMAGEXPRESS = "ImageXpress"
    OPERAPHENIX = "OperaPhenix"
    BBBC021 = "bbbc021"
    BBBC038 = "bbbc038"
    OMERO = "omero"  # Added for OMERO virtual filesystem backend
    BIOFORMATS = "bioformats"

class LiteralDtype(Enum):
    """Concrete numpy dtype literals (single source of truth)."""
    UINT8 = "uint8"
    UINT16 = "uint16"
    FLOAT32 = "float32"

    @property
    def numpy_dtype(self):
        """Get the corresponding numpy dtype."""
        import numpy as np
        return {
            self.UINT8: np.uint8,
            self.UINT16: np.uint16,
            self.FLOAT32: np.float32,
        }[self]


# Build DtypeConversion from LiteralDtype + execution-specific modes
DtypeConversion = Enum('DtypeConversion', {
    **{m.name: m.value for m in LiteralDtype},
    'PRESERVE_INPUT': 'preserve',
    'NATIVE_OUTPUT': 'native',
})
DtypeConversion.__doc__ = """Data type conversion modes for registered function execution."""
DtypeConversion.__module__ = __name__
DtypeConversion.__qualname__ = 'DtypeConversion'


def _add_numpy_dtype_property():
    """Add numpy_dtype property to DtypeConversion enum."""
    def numpy_dtype_getter(self):
        """Get the corresponding numpy dtype (None for PRESERVE/NATIVE)."""
        import numpy as np
        dtype_map = {
            DtypeConversion.UINT8: np.uint8,
            DtypeConversion.UINT16: np.uint16,
            DtypeConversion.FLOAT32: np.float32,
        }
        return dtype_map.get(self, None)

    DtypeConversion.numpy_dtype = property(numpy_dtype_getter)

_add_numpy_dtype_property()


def get_openhcs_config():
    """Get the OpenHCS configuration, initializing it if needed."""
    from openhcs.components.framework import ComponentConfigurationFactory
    return ComponentConfigurationFactory.create_openhcs_default_configuration()


# Lazy import cache manager to avoid circular dependencies
_component_enum_cache_manager = None


def _get_component_enum_cache_manager():
    """Lazy import of cache manager for component enums."""
    global _component_enum_cache_manager
    if _component_enum_cache_manager is None:
        try:
            from metaclass_registry.cache import RegistryCacheManager, CacheConfig

            def get_version():
                try:
                    import openhcs
                    return openhcs.__version__
                except:
                    return "unknown"

            # Serializer for component enum data
            # Note: RegistryCacheManager calls serializer(item) for each item in the dict
            # We store all three enums as a single item with key 'enums'
            def serialize_component_enums(enum_data: Dict[str, Any]) -> Dict[str, Any]:
                """Serialize the three component enum dicts to JSON."""
                return enum_data  # Already a dict of dicts

            # Deserializer for component enum data
            def deserialize_component_enums(data: Dict[str, Any]) -> Dict[str, Any]:
                """Deserialize component enum data from JSON."""
                return data  # Already a dict of dicts

            _component_enum_cache_manager = RegistryCacheManager(
                cache_name="component_enums",
                version_getter=get_version,
                serializer=serialize_component_enums,
                deserializer=deserialize_component_enums,
                config=CacheConfig(
                    max_age_days=7,
                    check_mtimes=False  # No file tracking needed for config-based enums
                )
            )
        except Exception as e:
            logger.debug(f"Failed to initialize component enum cache manager: {e}")
            _component_enum_cache_manager = False  # Mark as failed to avoid retrying

    return _component_enum_cache_manager if _component_enum_cache_manager is not False else None


def _add_groupby_methods(GroupBy: Enum) -> Enum:
    """Add custom methods to GroupBy enum."""
    GroupBy.component = property(lambda self: self.value)
    GroupBy.__eq__ = lambda self, other: self.value == getattr(other, 'value', other)
    GroupBy.__hash__ = lambda self: hash("GroupBy.NONE") if self.value is None else hash(self.value)
    GroupBy.__str__ = lambda self: f"GroupBy.{self.name}"
    GroupBy.__repr__ = lambda self: f"GroupBy.{self.name}"
    return GroupBy


# Simple lazy initialization - just defer the config call
@lru_cache(maxsize=1)
def _create_enums():
    """Create enums when first needed with persistent caching.

    CRITICAL: This function must create enums with proper __module__ and __qualname__
    attributes so they can be pickled correctly in multiprocessing contexts.
    The enums are stored in module globals() to ensure identity consistency.

    Caching provides ~20x speedup on subsequent runs and in subprocesses.
    """
    import os
    logger.debug("_create_enums() called in process %s", os.getpid())
    logger.debug("_create_enums() cache_info: %s", _create_enums.cache_info())
    if logger.isEnabledFor(logging.DEBUG):
        import traceback

        logger.debug(
            "_create_enums() stack trace:\n%s",
            "".join(traceback.format_stack()),
        )

    # Try to load from persistent cache first
    cache_manager = _get_component_enum_cache_manager()
    if cache_manager:
        try:
            cached_dict = cache_manager.load_cache()
            if cached_dict is not None and 'enums' in cached_dict:
                # Cache hit - reconstruct enums from cached data
                cached_data = cached_dict['enums']
                logger.debug("✅ Loading component enums from cache")

                all_components = Enum('AllComponents', cached_data['all_components'])
                all_components.__module__ = __name__
                all_components.__qualname__ = 'AllComponents'

                vc = Enum('VariableComponents', cached_data['variable_components'])
                vc.__module__ = __name__
                vc.__qualname__ = 'VariableComponents'

                GroupBy = Enum('GroupBy', cached_data['group_by'])
                GroupBy.__module__ = __name__
                GroupBy.__qualname__ = 'GroupBy'
                GroupBy = _add_groupby_methods(GroupBy)

                sc = Enum('SequentialComponents', cached_data['sequential_components'])
                sc.__module__ = __name__
                sc.__qualname__ = 'SequentialComponents'

                logger.debug(
                    "_create_enums() loaded from cache in process %s",
                    os.getpid(),
                )
                return all_components, vc, GroupBy, sc
        except Exception as e:
            logger.debug(f"Cache load failed for component enums: {e}")

    # Cache miss or disabled - create enums from config
    config = get_openhcs_config()
    remaining = config.get_remaining_components()

    # AllComponents: ALL possible dimensions (including multiprocessing axis)
    all_components_dict = {c.name: c.value for c in config.all_components}
    all_components = Enum('AllComponents', all_components_dict)
    all_components.__module__ = __name__
    all_components.__qualname__ = 'AllComponents'

    # VariableComponents: Components available for variable selection (excludes multiprocessing axis)
    vc_dict = {c.name: c.value for c in remaining}
    vc = Enum('VariableComponents', vc_dict)
    vc.__module__ = __name__
    vc.__qualname__ = 'VariableComponents'

    # GroupBy: Same as VariableComponents + NONE option (they're the same concept)
    gb_dict = {c.name: c.value for c in remaining}
    gb_dict['NONE'] = None
    GroupBy = Enum('GroupBy', gb_dict)
    GroupBy.__module__ = __name__
    GroupBy.__qualname__ = 'GroupBy'
    GroupBy = _add_groupby_methods(GroupBy)

    # SequentialComponents: Same as VariableComponents (for sequential processing)
    sc_dict = {c.name: c.value for c in remaining}
    sc = Enum('SequentialComponents', sc_dict)
    sc.__module__ = __name__
    sc.__qualname__ = 'SequentialComponents'

    # Save to persistent cache
    # Store all four enums as a single item with key 'enums'
    if cache_manager:
        try:
            enum_data = {
                'all_components': all_components_dict,
                'variable_components': vc_dict,
                'group_by': gb_dict,
                'sequential_components': sc_dict
            }
            cache_manager.save_cache({'enums': enum_data})
            logger.debug("💾 Saved component enums to cache")
        except Exception as e:
            logger.debug(f"Failed to save component enum cache: {e}")

    logger.debug(
        "_create_enums() returning in process %s: AllComponents=%s, "
        "VariableComponents=%s, GroupBy=%s, SequentialComponents=%s",
        os.getpid(),
        id(all_components),
        id(vc),
        id(GroupBy),
        id(sc),
    )
    logger.debug(
        "_create_enums() cache_info after return: %s",
        _create_enums.cache_info(),
    )
    return all_components, vc, GroupBy, sc


@lru_cache(maxsize=1)
def _create_streaming_components():
    """Create StreamingComponents enum from real filename components."""
    import os
    logger.debug("_create_streaming_components() called in process %s", os.getpid())

    # Import AllComponents (triggers lazy creation if needed)
    from openhcs.constants import AllComponents

    components_dict = {c.name: c.value for c in AllComponents}

    streaming_components = Enum('StreamingComponents', components_dict)
    streaming_components.__module__ = __name__
    streaming_components.__qualname__ = 'StreamingComponents'

    logger.debug(
        "_create_streaming_components() returning: StreamingComponents=%s",
        id(streaming_components),
    )
    return streaming_components


def __getattr__(name):
    """Lazy enum creation with identity guarantee.

    CRITICAL: Ensures enums are created exactly once per process and stored in globals()
    so that pickle identity checks pass in multiprocessing contexts.
    """
    if name in ('AllComponents', 'VariableComponents', 'GroupBy', 'SequentialComponents'):
        # Check if already created (handles race conditions)
        if name in globals():
            return globals()[name]

        # Create all enums at once and store in globals
        import os
        logger.debug(
            "Creating dynamic component enum %s in process %s",
            name,
            os.getpid(),
        )

        all_components, vc, gb, sc = _create_enums()
        globals()['AllComponents'] = all_components
        globals()['VariableComponents'] = vc
        globals()['GroupBy'] = gb
        globals()['SequentialComponents'] = sc

        logger.debug(
            "Created dynamic component enums in process %s: "
            "AllComponents=%s, VariableComponents=%s, GroupBy=%s, "
            "SequentialComponents=%s",
            os.getpid(),
            id(all_components),
            id(vc),
            id(gb),
            id(sc),
        )
        logger.debug(
            "VariableComponents.__module__=%s, __qualname__=%s",
            vc.__module__,
            vc.__qualname__,
        )

        return globals()[name]

    if name == 'StreamingComponents':
        # Check if already created
        if name in globals():
            return globals()[name]

        import os
        logger.debug(
            "Creating StreamingComponents in process %s",
            os.getpid(),
        )

        streaming_components = _create_streaming_components()
        globals()['StreamingComponents'] = streaming_components

        logger.debug(
            "Created StreamingComponents in process %s: StreamingComponents=%s",
            os.getpid(),
            id(streaming_components),
        )

        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Documentation URL
DOCUMENTATION_URL = "https://openhcs.readthedocs.io/en/latest/"


class OrchestratorState(Enum):
    """Simple orchestrator state tracking - no complex state machine."""
    CREATED = ("created", False, False)  # Object exists, not initialized
    READY = ("ready", True, True)  # Initialized, ready for compilation
    COMPILED = ("compiled", True, True)  # Compilation complete
    EXECUTING = ("executing", True, False)  # Execution in progress
    COMPLETED = ("completed", True, True)  # Execution completed successfully
    INIT_FAILED = ("init_failed", False, False)  # Initialization failed
    COMPILE_FAILED = ("compile_failed", True, False)  # Compilation failed
    EXEC_FAILED = ("exec_failed", True, False)  # Execution failed

    def __new__(
        cls,
        value: str,
        has_completed_initialization: bool,
        skips_initialization: bool,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.has_completed_initialization = has_completed_initialization
        obj.skips_initialization = skips_initialization
        return obj

# I/O-related constants
DEFAULT_IMAGE_EXTENSION = ".tif"
_TIFF_IMAGE_EXTENSIONS: Set[str] = {".tif", ".tiff"}
_RASTER_IMAGE_EXTENSIONS: Set[str] = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
}
DEFAULT_IMAGE_EXTENSIONS: Set[str] = set(_TIFF_IMAGE_EXTENSIONS)
LOADABLE_IMAGE_EXTENSIONS: Set[str] = _TIFF_IMAGE_EXTENSIONS | _RASTER_IMAGE_EXTENSIONS
DEFAULT_SITE_PADDING = 3
DEFAULT_RECURSIVE_PATTERN_SEARCH = False
# Lazy default resolution using lru_cache
@lru_cache(maxsize=1)
def get_default_variable_components():
    """Get default variable components from ComponentConfiguration."""
    _, vc, _, _ = _create_enums()  # Get the enum directly
    return [getattr(vc, c.name) for c in get_openhcs_config().default_variable]


@lru_cache(maxsize=1)
def get_default_group_by():
    """Get default group_by from ComponentConfiguration."""
    _, _, gb, _ = _create_enums()  # Get the enum directly
    config = get_openhcs_config()
    return getattr(gb, config.default_group_by.name) if config.default_group_by else None

@lru_cache(maxsize=1)
def get_multiprocessing_axis():
    """Get multiprocessing axis from ComponentConfiguration."""
    config = get_openhcs_config()
    return config.multiprocessing_axis

DEFAULT_MICROSCOPE: Microscope = Microscope.AUTO


# Backend-related constants
class Backend(Enum):
    AUTO = "auto"
    DISK = "disk"
    MEMORY = "memory"
    ZARR = "zarr"
    NAPARI_STREAM = "napari_stream"
    FIJI_STREAM = "fiji_stream"
    OMERO_LOCAL = "omero_local"
    VIRTUAL_WORKSPACE = "virtual_workspace"
    BIOFORMATS = "bioformats"

class FileFormat(Enum):
    TIFF = list(DEFAULT_IMAGE_EXTENSIONS)
    NUMPY = [".npy"]
    TORCH = [".pt", ".torch", ".pth"]
    JAX = [".jax"]
    CUPY = [".cupy",".craw"]
    TENSORFLOW = [".tf"]
    JSON = [".json"]
    CSV = [".csv"]
    TEXT = [".txt", ".py", ".md"]
    ROI = [".roi.zip"]

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

# ZMQ transport constants
# Note: Streaming port defaults are defined in NapariStreamingConfig and FijiStreamingConfig
CONTROL_PORT_OFFSET = 1000  # Control port = data port + 1000
DEFAULT_EXECUTION_SERVER_PORT = 7777
IPC_SOCKET_DIR_NAME = "ipc"  # ~/.openhcs/ipc/
IPC_SOCKET_PREFIX = "openhcs-zmq"  # ipc://openhcs-zmq-{port} or ~/.openhcs/ipc/openhcs-zmq-{port}.sock
IPC_SOCKET_EXTENSION = ".sock"  # Unix domain socket extension


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
