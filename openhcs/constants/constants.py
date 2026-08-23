"""
Consolidated constants for OpenHCS.

This module defines all constants related to backends, defaults, I/O, memory, and pipeline.
These constants are governed by various doctrinal clauses.

Component enums are created once per process from the declared component order.
"""

import logging
from enum import Enum
from functools import lru_cache
from typing import Any, Set

from arraybridge.types import (
    CPU_MEMORY_TYPES as CPU_MEMORY_TYPES,
)
from arraybridge.types import (
    GPU_MEMORY_TYPES as GPU_MEMORY_TYPES,
)
from arraybridge.types import (
    MEMORY_TYPE_CUPY as MEMORY_TYPE_CUPY,
)
from arraybridge.types import (
    MEMORY_TYPE_JAX as MEMORY_TYPE_JAX,
)
from arraybridge.types import (
    MEMORY_TYPE_NUMPY as MEMORY_TYPE_NUMPY,
)
from arraybridge.types import (
    MEMORY_TYPE_PYCLESPERANTO as MEMORY_TYPE_PYCLESPERANTO,
)
from arraybridge.types import (
    MEMORY_TYPE_TENSORFLOW as MEMORY_TYPE_TENSORFLOW,
)
from arraybridge.types import (
    MEMORY_TYPE_TORCH as MEMORY_TYPE_TORCH,
)
from arraybridge.types import (
    SUPPORTED_MEMORY_TYPES as SUPPORTED_MEMORY_TYPES,
)
from arraybridge.types import (
    VALID_GPU_MEMORY_TYPES as VALID_GPU_MEMORY_TYPES,
)
from arraybridge.types import (
    VALID_MEMORY_TYPES as VALID_MEMORY_TYPES,
)
from arraybridge.types import (
    MemoryType as MemoryType,
)
from polystore.constants import Backend

logger = logging.getLogger(__name__)


class Microscope(Enum):
    AUTO = "auto"
    OPENHCS = "openhcsdata"
    IMAGEXPRESS = "imagexpress"
    OPERAPHENIX = "opera_phenix"
    OMERO = "omero"  # Added for OMERO virtual filesystem backend
    BIOFORMATS = "bioformats"
    SOURCE_BINDINGS = "source_bindings"


def get_openhcs_config():
    """Get the OpenHCS configuration, initializing it if needed."""
    from openhcs.components.framework import ComponentConfigurationFactory

    return ComponentConfigurationFactory.create_openhcs_default_configuration()


def _enum_value_for_comparison(value: Any) -> Any:
    """Return enum value for equality hooks without reflective attribute access."""
    if isinstance(value, Enum):
        return value.value
    return value


def _add_groupby_methods(GroupBy: Enum) -> Enum:
    """Add custom methods to GroupBy enum."""

    def groupby_eq(self, other: Any) -> bool:
        # GroupBy.NONE is a concrete enum value in user/config state. It must
        # not compare equal to Python None, which is the lazy-inheritance
        # sentinel in ObjectState and lazy dataclass fields.
        if other is None:
            return False
        return self.value == _enum_value_for_comparison(other)

    GroupBy.component = property(lambda self: self.value)
    GroupBy.__eq__ = groupby_eq
    GroupBy.__hash__ = lambda self: (
        hash("GroupBy.NONE") if self.value is None else hash(self.value)
    )
    GroupBy.__str__ = lambda self: f"GroupBy.{self.name}"
    GroupBy.__repr__ = lambda self: f"GroupBy.{self.name}"
    return GroupBy


def _add_allcomponents_methods(AllComponents: Enum) -> Enum:
    """Add component-axis semantic methods to the dynamic component enum."""

    def from_value(cls, value: Any):
        for component in cls:
            if component.value == value:
                return component
        return None

    def ordered_names(cls) -> tuple[str, ...]:
        return tuple(component.value for component in cls)

    def is_multiprocessing_axis(self) -> bool:
        return self.value == get_multiprocessing_axis().value

    def is_variable_axis(self) -> bool:
        return not self.is_multiprocessing_axis()

    def is_default_group_by_axis(self) -> bool:
        group_by = get_default_group_by()
        return group_by is not None and self.value == group_by.value

    AllComponents.from_value = classmethod(from_value)
    AllComponents.ordered_names = classmethod(ordered_names)
    AllComponents.is_multiprocessing_axis = is_multiprocessing_axis
    AllComponents.is_variable_axis = is_variable_axis
    AllComponents.is_default_group_by_axis = is_default_group_by_axis
    return AllComponents


# Simple lazy initialization - just defer the config call
@lru_cache(maxsize=1)
def _create_enums():
    """Create process-local component enums when first needed.

    CRITICAL: This function must create enums with proper __module__ and __qualname__
    attributes so they can be pickled correctly in multiprocessing contexts.
    The enums are stored in module globals() to ensure identity consistency.

    The function-local cache preserves enum identity within the process.
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

    config = get_openhcs_config()
    remaining = config.get_remaining_components()

    # AllComponents: ALL possible dimensions (including multiprocessing axis)
    all_components_dict = {c.name: c.value for c in config.all_components}
    all_components = Enum("AllComponents", all_components_dict)
    all_components.__module__ = __name__
    all_components.__qualname__ = "AllComponents"
    all_components = _add_allcomponents_methods(all_components)

    # VariableComponents: Components available for variable selection (excludes multiprocessing axis)
    vc_dict = {c.name: c.value for c in remaining}
    vc = Enum("VariableComponents", vc_dict)
    vc.__module__ = __name__
    vc.__qualname__ = "VariableComponents"

    # GroupBy: Same as VariableComponents + NONE option (they're the same concept)
    gb_dict = {c.name: c.value for c in remaining}
    gb_dict["NONE"] = None
    GroupBy = Enum("GroupBy", gb_dict)
    GroupBy.__module__ = __name__
    GroupBy.__qualname__ = "GroupBy"
    GroupBy = _add_groupby_methods(GroupBy)

    # SequentialComponents: Same as VariableComponents (for sequential processing)
    sc_dict = {c.name: c.value for c in remaining}
    sc = Enum("SequentialComponents", sc_dict)
    sc.__module__ = __name__
    sc.__qualname__ = "SequentialComponents"

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

    components_dict = {c.name: c.value for c in AllComponents}

    streaming_components = Enum("StreamingComponents", components_dict)
    streaming_components.__module__ = __name__
    streaming_components.__qualname__ = "StreamingComponents"

    logger.debug(
        "_create_streaming_components() returning: StreamingComponents=%s",
        id(streaming_components),
    )
    return streaming_components


AllComponents, VariableComponents, GroupBy, SequentialComponents = _create_enums()
StreamingComponents = _create_streaming_components()


# Documentation URL
DOCUMENTATION_URL = "https://openhcs.readthedocs.io/en/latest/"


class OrchestratorState(Enum):
    """Simple orchestrator state tracking - no complex state machine."""

    has_completed_initialization: bool
    skips_initialization: bool
    allows_execution: bool
    status_prefix: str

    CREATED = ("created", False, False, False, "")  # Object exists, not initialized
    READY = ("ready", True, True, False, "✓ Init")  # Initialized, ready for compilation
    COMPILED = ("compiled", True, True, True, "✓ Compiled")  # Compilation complete
    EXECUTING = (
        "executing",
        True,
        False,
        False,
        "🔄 Executing",
    )  # Execution in progress
    COMPLETED = (
        "completed",
        True,
        True,
        True,
        "✅ Complete",
    )  # Execution completed successfully
    INIT_FAILED = (
        "init_failed",
        False,
        False,
        False,
        "❌ Init Failed",
    )  # Initialization failed
    COMPILE_FAILED = (
        "compile_failed",
        True,
        False,
        False,
        "❌ Compile Failed",
    )  # Compilation failed
    EXEC_FAILED = (
        "exec_failed",
        True,
        False,
        False,
        "❌ Exec Failed",
    )  # Execution failed

    def __new__(
        cls,
        value: str,
        has_completed_initialization: bool,
        skips_initialization: bool,
        allows_execution: bool,
        status_prefix: str,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.has_completed_initialization = has_completed_initialization
        obj.skips_initialization = skips_initialization
        obj.allows_execution = allows_execution
        obj.status_prefix = status_prefix
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
    return [vc.__members__[c.name] for c in get_openhcs_config().default_variable]


@lru_cache(maxsize=1)
def get_default_group_by():
    """Get default group_by from ComponentConfiguration."""
    _, _, gb, _ = _create_enums()  # Get the enum directly
    config = get_openhcs_config()
    if config.default_group_by is None:
        return gb.__members__["NONE"]
    return gb.__members__[config.default_group_by.name]


@lru_cache(maxsize=1)
def get_multiprocessing_axis():
    """Get multiprocessing axis from ComponentConfiguration."""
    config = get_openhcs_config()
    return config.multiprocessing_axis


DEFAULT_MICROSCOPE: Microscope = Microscope.AUTO


class FileFormat(Enum):
    TIFF = list(DEFAULT_IMAGE_EXTENSIONS)
    NUMPY = [".npy"]
    TORCH = [".pt", ".torch", ".pth"]
    JAX = [".jax"]
    CUPY = [".cupy", ".craw"]
    TENSORFLOW = [".tf"]
    JSON = [".json"]
    CSV = [".csv"]
    TEXT = [".txt", ".py", ".md", ".swc"]
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

DEFAULT_NUM_WORKERS = 1
