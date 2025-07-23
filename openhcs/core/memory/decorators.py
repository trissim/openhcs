"""
Memory type declaration decorators for OpenHCS.

This module provides decorators for explicitly declaring the memory interface
of pure functions, enforcing Clause 106-A (Declared Memory Types) and supporting
memory-type-aware dispatching and orchestration.

These decorators annotate functions with input_memory_type and output_memory_type
attributes and provide automatic thread-local CUDA stream management for GPU
frameworks to enable true parallelization across multiple threads.
"""

import functools
import logging
import threading
from typing import Any, Callable, Optional, TypeVar

from openhcs.constants.constants import VALID_MEMORY_TYPES
from openhcs.core.utils import optional_import
from openhcs.core.memory.oom_recovery import _execute_with_oom_recovery

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

# Dtype conversion enum and utilities for consistent dtype handling across all frameworks
from enum import Enum
import numpy as np

class DtypeConversion(Enum):
    """Data type conversion modes for all memory type functions."""

    PRESERVE_INPUT = "preserve"     # Keep input dtype (default)
    NATIVE_OUTPUT = "native"        # Use framework's native output
    UINT8 = "uint8"                # Force uint8 (0-255 range)
    UINT16 = "uint16"              # Force uint16 (microscopy standard)
    INT16 = "int16"                # Force int16 (signed microscopy data)
    INT32 = "int32"                # Force int32 (large integer values)
    FLOAT32 = "float32"            # Force float32 (GPU performance)
    FLOAT64 = "float64"            # Force float64 (maximum precision)

    @property
    def numpy_dtype(self):
        """Get the corresponding numpy dtype."""
        dtype_map = {
            self.UINT8: np.uint8,
            self.UINT16: np.uint16,
            self.INT16: np.int16,
            self.INT32: np.int32,
            self.FLOAT32: np.float32,
            self.FLOAT64: np.float64,
        }
        return dtype_map.get(self, None)


def _scale_and_convert_numpy(result, target_dtype):
    """Scale numpy results to target integer range and convert dtype."""
    if not hasattr(result, 'dtype'):
        return result

    # Check if result is floating point and target is integer
    result_is_float = np.issubdtype(result.dtype, np.floating)
    target_is_int = target_dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]

    if result_is_float and target_is_int:
        # Scale floating point results to integer range
        result_min = result.min()
        result_max = result.max()

        if result_max > result_min:  # Avoid division by zero
            # Normalize to [0, 1] range
            normalized = (result - result_min) / (result_max - result_min)

            # Scale to target dtype range
            if target_dtype == np.uint8:
                scaled = normalized * 255.0
            elif target_dtype == np.uint16:
                scaled = normalized * 65535.0
            elif target_dtype == np.uint32:
                scaled = normalized * 4294967295.0
            elif target_dtype == np.int16:
                scaled = normalized * 65535.0 - 32768.0
            elif target_dtype == np.int32:
                scaled = normalized * 4294967295.0 - 2147483648.0
            else:
                scaled = normalized

            return scaled.astype(target_dtype)
        else:
            # Constant image, just convert dtype
            return result.astype(target_dtype)
    else:
        # Direct conversion for compatible types
        return result.astype(target_dtype)


def _scale_and_convert_pyclesperanto(result, target_dtype):
    """Scale pyclesperanto results to target integer range and convert dtype."""
    try:
        import pyclesperanto as cle
    except ImportError:
        return result

    if not hasattr(result, 'dtype'):
        return result

    # Check if result is floating point and target is integer
    result_is_float = np.issubdtype(result.dtype, np.floating)
    target_is_int = target_dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]

    if result_is_float and target_is_int:
        # Get min/max of result for proper scaling
        result_min = float(cle.minimum_of_all_pixels(result))
        result_max = float(cle.maximum_of_all_pixels(result))

        if result_max > result_min:  # Avoid division by zero
            # Normalize to [0, 1] range
            normalized = cle.subtract_image_from_scalar(result, scalar=result_min)
            range_val = result_max - result_min
            normalized = cle.multiply_image_and_scalar(normalized, scalar=1.0/range_val)

            # Scale to target dtype range
            if target_dtype == np.uint8:
                scaled = cle.multiply_image_and_scalar(normalized, scalar=255.0)
            elif target_dtype == np.uint16:
                scaled = cle.multiply_image_and_scalar(normalized, scalar=65535.0)
            elif target_dtype == np.uint32:
                scaled = cle.multiply_image_and_scalar(normalized, scalar=4294967295.0)
            elif target_dtype == np.int16:
                scaled = cle.multiply_image_and_scalar(normalized, scalar=65535.0)
                scaled = cle.subtract_image_from_scalar(scaled, scalar=32768.0)
            elif target_dtype == np.int32:
                scaled = cle.multiply_image_and_scalar(normalized, scalar=4294967295.0)
                scaled = cle.subtract_image_from_scalar(scaled, scalar=2147483648.0)
            else:
                scaled = normalized

            # Convert to target dtype using push/pull method
            scaled_cpu = cle.pull(scaled).astype(target_dtype)
            return cle.push(scaled_cpu)
        else:
            # Constant image, just convert dtype
            result_cpu = cle.pull(result).astype(target_dtype)
            return cle.push(result_cpu)
    else:
        # Direct conversion for compatible types
        result_cpu = cle.pull(result).astype(target_dtype)
        return cle.push(result_cpu)


def _scale_and_convert_cupy(result, target_dtype):
    """Scale CuPy results to target integer range and convert dtype."""
    try:
        import cupy as cp
    except ImportError:
        return result

    if not hasattr(result, 'dtype'):
        return result

    # If result is floating point and target is integer, scale appropriately
    if cp.issubdtype(result.dtype, cp.floating) and not cp.issubdtype(target_dtype, cp.floating):
        # Clip to [0, 1] range and scale to integer range
        clipped = cp.clip(result, 0, 1)
        if target_dtype == cp.uint8:
            return (clipped * 255).astype(target_dtype)
        elif target_dtype == cp.uint16:
            return (clipped * 65535).astype(target_dtype)
        elif target_dtype == cp.uint32:
            return (clipped * 4294967295).astype(target_dtype)
        else:
            # For other integer types, just convert without scaling
            return result.astype(target_dtype)

    # Direct conversion for same numeric type families
    return result.astype(target_dtype)


# GPU frameworks imported lazily to prevent thread explosion
# These will be imported only when actually needed by functions
_gpu_frameworks_cache = {}

def _get_cupy():
    """Lazy import CuPy only when needed."""
    if 'cupy' not in _gpu_frameworks_cache:
        _gpu_frameworks_cache['cupy'] = optional_import("cupy")
        if _gpu_frameworks_cache['cupy'] is not None:
            logger.debug(f"🔧 Lazy imported CuPy in thread {threading.current_thread().name}")
    return _gpu_frameworks_cache['cupy']

def _get_torch():
    """Lazy import PyTorch only when needed."""
    if 'torch' not in _gpu_frameworks_cache:
        _gpu_frameworks_cache['torch'] = optional_import("torch")
        if _gpu_frameworks_cache['torch'] is not None:
            logger.debug(f"🔧 Lazy imported PyTorch in thread {threading.current_thread().name}")
    return _gpu_frameworks_cache['torch']

def _get_tensorflow():
    """Lazy import TensorFlow only when needed."""
    if 'tensorflow' not in _gpu_frameworks_cache:
        _gpu_frameworks_cache['tensorflow'] = optional_import("tensorflow")
        if _gpu_frameworks_cache['tensorflow'] is not None:
            logger.debug(f"🔧 Lazy imported TensorFlow in thread {threading.current_thread().name}")
    return _gpu_frameworks_cache['tensorflow']

def _get_jax():
    """Lazy import JAX only when needed."""
    if 'jax' not in _gpu_frameworks_cache:
        _gpu_frameworks_cache['jax'] = optional_import("jax")
        if _gpu_frameworks_cache['jax'] is not None:
            logger.debug(f"🔧 Lazy imported JAX in thread {threading.current_thread().name}")
    return _gpu_frameworks_cache['jax']

# Thread-local storage for GPU streams and contexts
_thread_gpu_contexts = threading.local()

class ThreadGPUContext:
    """Unified thread-local GPU context manager to prevent stream leaks."""

    def __init__(self):
        self._cupy_stream = None
        self._torch_stream = None
        self._thread_name = threading.current_thread().name

    def get_cupy_stream(self):
        """Get or create the single CuPy stream for this thread."""
        if self._cupy_stream is None:
            cp = _get_cupy()
            if cp is not None and hasattr(cp, 'cuda'):
                self._cupy_stream = cp.cuda.Stream()
                logger.debug(f"🔧 Created CuPy stream for thread {self._thread_name}")
        return self._cupy_stream

    def get_torch_stream(self):
        """Get or create the single PyTorch stream for this thread."""
        if self._torch_stream is None:
            torch = _get_torch()
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                self._torch_stream = torch.cuda.Stream()
                logger.debug(f"🔧 Created PyTorch stream for thread {self._thread_name}")
        return self._torch_stream

    def cleanup(self):
        """Clean up streams when thread exits."""
        if self._cupy_stream is not None:
            logger.debug(f"🔧 Cleaning up CuPy stream for thread {self._thread_name}")
            self._cupy_stream = None

        if self._torch_stream is not None:
            logger.debug(f"🔧 Cleaning up PyTorch stream for thread {self._thread_name}")
            self._torch_stream = None

def get_thread_gpu_context() -> ThreadGPUContext:
    """Get the unified GPU context for the current thread."""
    if not hasattr(_thread_gpu_contexts, 'gpu_context'):
        _thread_gpu_contexts.gpu_context = ThreadGPUContext()

        # Register cleanup for when thread exits
        import weakref
        def cleanup_on_thread_exit():
            if hasattr(_thread_gpu_contexts, 'gpu_context'):
                _thread_gpu_contexts.gpu_context.cleanup()

        # Use weakref to avoid circular references
        current_thread = threading.current_thread()
        if hasattr(current_thread, '_cleanup_funcs'):
            current_thread._cleanup_funcs.append(cleanup_on_thread_exit)
        else:
            current_thread._cleanup_funcs = [cleanup_on_thread_exit]

    return _thread_gpu_contexts.gpu_context


def memory_types(*, input_type: str, output_type: str) -> Callable[[F], F]:
    """
    Decorator that explicitly declares the memory types for a function's input and output.

    This decorator enforces Clause 106-A (Declared Memory Types) by requiring explicit
    memory type declarations for both input and output.

    Args:
        input_type: The memory type for the function's input (e.g., "numpy", "cupy")
        output_type: The memory type for the function's output (e.g., "numpy", "cupy")

    Returns:
        A decorator function that sets the memory type attributes

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    # 🔒 Clause 88 — No Inferred Capabilities
    # Validate memory types at decoration time, not runtime
    if not input_type:
        raise ValueError(
            "Clause 106-A Violation: input_type must be explicitly declared. "
            "No default or inferred memory types are allowed."
        )

    if not output_type:
        raise ValueError(
            "Clause 106-A Violation: output_type must be explicitly declared. "
            "No default or inferred memory types are allowed."
        )

    # Validate that memory types are supported
    if input_type not in VALID_MEMORY_TYPES:
        raise ValueError(
            f"Clause 106-A Violation: input_type '{input_type}' is not supported. "
            f"Supported types are: {', '.join(sorted(VALID_MEMORY_TYPES))}"
        )

    if output_type not in VALID_MEMORY_TYPES:
        raise ValueError(
            f"Clause 106-A Violation: output_type '{output_type}' is not supported. "
            f"Supported types are: {', '.join(sorted(VALID_MEMORY_TYPES))}"
        )

    def decorator(func: F) -> F:
        """
        Decorator function that sets memory type attributes on the function.

        Args:
            func: The function to decorate

        Returns:
            The decorated function with memory type attributes set

        Raises:
            ValueError: If the function already has different memory type attributes
        """
        # 🔒 Clause 66 — Immutability
        # Check if memory type attributes already exist
        if hasattr(func, 'input_memory_type') and func.input_memory_type != input_type:
            raise ValueError(
                f"Clause 66 Violation: Function '{func.__name__}' already has input_memory_type "
                f"'{func.input_memory_type}', cannot change to '{input_type}'."
            )

        if hasattr(func, 'output_memory_type') and func.output_memory_type != output_type:
            raise ValueError(
                f"Clause 66 Violation: Function '{func.__name__}' already has output_memory_type "
                f"'{func.output_memory_type}', cannot change to '{output_type}'."
            )

        # Set memory type attributes using canonical names
        # 🔒 Clause 106-A.2 — Canonical Memory Type Attributes
        func.input_memory_type = input_type
        func.output_memory_type = output_type

        # Return the function unchanged (no wrapper)
        return func

    return decorator


def numpy(
    func: Optional[F] = None,
    *,
    input_type: str = "numpy",
    output_type: str = "numpy"
) -> Any:
    """
    Decorator that declares a function as operating on numpy arrays.

    This is a convenience wrapper around memory_types with numpy defaults.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "numpy")
        output_type: The memory type for the function's output (default: "numpy")

    Returns:
        The decorated function with memory type attributes set

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator_with_dtype_preservation(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Apply dtype preservation wrapper
        func = _create_numpy_dtype_preserving_wrapper(func, func.__name__)

        return func

    # Handle both @numpy and @numpy(input_type=..., output_type=...) forms
    if func is None:
        return decorator_with_dtype_preservation

    return decorator_with_dtype_preservation(func)


def cupy(func: Optional[F] = None, *, input_type: str = "cupy", output_type: str = "cupy", oom_recovery: bool = True) -> Any:
    """
    Decorator that declares a function as operating on cupy arrays.

    This decorator provides automatic thread-local CUDA stream management for
    true parallelization across multiple threads. Each thread gets its own
    persistent CUDA stream that is reused for all CuPy operations.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "cupy")
        output_type: The memory type for the function's output (default: "cupy")
        oom_recovery: Enable automatic OOM recovery (default: True)

    Returns:
        The decorated function with memory type attributes and stream management

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Apply dtype preservation wrapper
        func = _create_cupy_dtype_preserving_wrapper(func, func.__name__)

        # Add CUDA stream wrapper if CuPy is available (lazy import)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cp = _get_cupy()
            if cp is not None and hasattr(cp, 'cuda'):
                # Get unified thread context and CuPy stream
                gpu_context = get_thread_gpu_context()
                cupy_stream = gpu_context.get_cupy_stream()

                def execute_with_stream():
                    if cupy_stream is not None:
                        # Execute function in stream context
                        with cupy_stream:
                            return func(*args, **kwargs)
                    else:
                        # No CUDA available, execute without stream
                        return func(*args, **kwargs)

                # Execute with OOM recovery if enabled
                if oom_recovery:
                    return _execute_with_oom_recovery(execute_with_stream, input_type)
                else:
                    return execute_with_stream()
            else:
                # CuPy not available, execute without stream
                return func(*args, **kwargs)

        # Preserve memory type attributes
        wrapper.input_memory_type = func.input_memory_type
        wrapper.output_memory_type = func.output_memory_type

        return wrapper

    # Handle both @cupy and @cupy(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def torch(
    func: Optional[F] = None,
    *,
    input_type: str = "torch",
    output_type: str = "torch",
    oom_recovery: bool = True
) -> Any:
    """
    Decorator that declares a function as operating on torch tensors.

    This decorator provides automatic thread-local CUDA stream management for
    true parallelization across multiple threads. Each thread gets its own
    persistent CUDA stream that is reused for all PyTorch operations.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "torch")
        output_type: The memory type for the function's output (default: "torch")
        oom_recovery: Enable automatic OOM recovery (default: True)

    Returns:
        The decorated function with memory type attributes and stream management

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Apply dtype preservation wrapper
        func = _create_torch_dtype_preserving_wrapper(func, func.__name__)

        # Add CUDA stream wrapper if PyTorch is available and CUDA is available (lazy import)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            torch = _get_torch()
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                # Get unified thread context and PyTorch stream
                gpu_context = get_thread_gpu_context()
                torch_stream = gpu_context.get_torch_stream()

                def execute_with_stream():
                    if torch_stream is not None:
                        # Execute function in stream context
                        with torch.cuda.stream(torch_stream):
                            return func(*args, **kwargs)
                    else:
                        # No CUDA available, execute without stream
                        return func(*args, **kwargs)

                # Execute with OOM recovery if enabled
                if oom_recovery:
                    return _execute_with_oom_recovery(execute_with_stream, input_type)
                else:
                    return execute_with_stream()
            else:
                # PyTorch not available or CUDA not available, execute without stream
                return func(*args, **kwargs)

        # Preserve memory type attributes
        wrapper.input_memory_type = func.input_memory_type
        wrapper.output_memory_type = func.output_memory_type

        return wrapper

    # Handle both @torch and @torch(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def tensorflow(
    func: Optional[F] = None,
    *,
    input_type: str = "tensorflow",
    output_type: str = "tensorflow",
    oom_recovery: bool = True
) -> Any:
    """
    Decorator that declares a function as operating on tensorflow tensors.

    This decorator provides automatic thread-local GPU device context management
    for parallelization across multiple threads. TensorFlow manages CUDA streams
    internally, so we use device contexts for thread isolation.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "tensorflow")
        output_type: The memory type for the function's output (default: "tensorflow")
        oom_recovery: Enable automatic OOM recovery (default: True)

    Returns:
        The decorated function with memory type attributes and device management

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Apply dtype preservation wrapper
        func = _create_tensorflow_dtype_preserving_wrapper(func, func.__name__)

        # Add device context wrapper if TensorFlow is available and GPU is available (lazy import)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tf = _get_tensorflow()
            if tf is not None and tf.config.list_physical_devices('GPU'):
                def execute_with_device():
                    # Use GPU device context for thread isolation
                    # TensorFlow manages internal CUDA streams automatically
                    with tf.device('/GPU:0'):
                        return func(*args, **kwargs)

                # Execute with OOM recovery if enabled
                if oom_recovery:
                    return _execute_with_oom_recovery(execute_with_device, input_type)
                else:
                    return execute_with_device()
            else:
                # TensorFlow not available or GPU not available, execute without device context
                return func(*args, **kwargs)

        # Preserve memory type attributes
        wrapper.input_memory_type = func.input_memory_type
        wrapper.output_memory_type = func.output_memory_type

        return wrapper

    # Handle both @tensorflow and @tensorflow(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def jax(
    func: Optional[F] = None,
    *,
    input_type: str = "jax",
    output_type: str = "jax",
    oom_recovery: bool = True
) -> Any:
    """
    Decorator that declares a function as operating on JAX arrays.

    This decorator provides automatic thread-local GPU device placement for
    parallelization across multiple threads. JAX/XLA manages CUDA streams
    internally, so we use device placement for thread isolation.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "jax")
        output_type: The memory type for the function's output (default: "jax")
        oom_recovery: Enable automatic OOM recovery (default: True)

    Returns:
        The decorated function with memory type attributes and device management

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Apply dtype preservation wrapper
        func = _create_jax_dtype_preserving_wrapper(func, func.__name__)

        # Add device placement wrapper if JAX is available and GPU is available (lazy import)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            jax_module = _get_jax()
            if jax_module is not None:
                devices = jax_module.devices()
                gpu_devices = [d for d in devices if d.platform == 'gpu']

                if gpu_devices:
                    def execute_with_device():
                        # Use GPU device placement for thread isolation
                        # JAX/XLA manages internal CUDA streams automatically
                        with jax_module.default_device(gpu_devices[0]):
                            return func(*args, **kwargs)

                    # Execute with OOM recovery if enabled
                    if oom_recovery:
                        return _execute_with_oom_recovery(execute_with_device, input_type)
                    else:
                        return execute_with_device()
                else:
                    # No GPU devices available, execute without device placement
                    return func(*args, **kwargs)
            else:
                # JAX not available, execute without device placement
                return func(*args, **kwargs)

        # Preserve memory type attributes
        wrapper.input_memory_type = func.input_memory_type
        wrapper.output_memory_type = func.output_memory_type

        return wrapper

    # Handle both @jax and @jax(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def pyclesperanto(
    func: Optional[F] = None,
    *,
    input_type: str = "pyclesperanto",
    output_type: str = "pyclesperanto",
    oom_recovery: bool = True
) -> Any:
    """
    Decorator that declares a function as operating on pyclesperanto GPU arrays.

    This decorator provides automatic OOM recovery for pyclesperanto functions.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "pyclesperanto")
        output_type: The memory type for the function's output (default: "pyclesperanto")
        oom_recovery: Enable automatic OOM recovery (default: True)

    Returns:
        The decorated function with memory type attributes and OOM recovery

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Apply dtype preservation wrapper
        func = _create_pyclesperanto_dtype_preserving_wrapper(func, func.__name__)

        # Add OOM recovery wrapper
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if oom_recovery:
                return _execute_with_oom_recovery(lambda: func(*args, **kwargs), input_type)
            else:
                return func(*args, **kwargs)

        # Preserve memory type attributes
        wrapper.input_memory_type = func.input_memory_type
        wrapper.output_memory_type = func.output_memory_type

        # Make wrapper pickleable by preserving original function identity
        wrapper.__module__ = getattr(func, '__module__', wrapper.__module__)
        wrapper.__qualname__ = getattr(func, '__qualname__', wrapper.__qualname__)

        # Store reference to original function for pickle support
        wrapper.__wrapped__ = func

        return wrapper

    # Handle both @pyclesperanto and @pyclesperanto(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


# ============================================================================
# Dtype Preservation Wrapper Functions
# ============================================================================

def _create_numpy_dtype_preserving_wrapper(original_func, func_name):
    """
    Create a wrapper that preserves input data type and adds slice_by_slice parameter for NumPy functions.

    Many scikit-image functions return float64 regardless of input type.
    This wrapper ensures the output has the same dtype as the input and adds
    a slice_by_slice parameter to avoid cross-slice contamination in 3D arrays.
    """
    import numpy as np
    import inspect
    from functools import wraps

    @wraps(original_func)
    def numpy_dtype_and_slice_preserving_wrapper(image, *args, dtype_conversion=None, slice_by_slice: bool = False, **kwargs):
        # Set default dtype_conversion if not provided and DtypeConversion is available
        if dtype_conversion is None and DtypeConversion is not None:
            dtype_conversion = DtypeConversion.PRESERVE_INPUT

        try:
            # Store original dtype
            original_dtype = image.dtype

            # Handle slice_by_slice processing for 3D arrays using OpenHCS stack utilities
            if slice_by_slice and hasattr(image, 'ndim') and image.ndim == 3:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type

                # Detect memory type and use proper OpenHCS utilities
                memory_type = _detect_memory_type(image)
                gpu_id = 0  # Default GPU ID for slice processing

                # Unstack 3D array into 2D slices
                slices_2d = unstack_slices(image, memory_type, gpu_id)

                # Process each slice and handle special outputs
                main_outputs = []
                special_outputs_list = []

                for slice_2d in slices_2d:
                    slice_result = original_func(slice_2d, *args, **kwargs)

                    # Check if result is a tuple (indicating special outputs)
                    if isinstance(slice_result, tuple):
                        main_outputs.append(slice_result[0])  # First element is main output
                        special_outputs_list.append(slice_result[1:])  # Rest are special outputs
                    else:
                        main_outputs.append(slice_result)  # Single output

                # Stack main outputs back into 3D array
                result = stack_slices(main_outputs, memory_type, gpu_id)

                # If we have special outputs, combine them and return tuple
                if special_outputs_list:
                    # Combine special outputs from all slices
                    combined_special_outputs = []
                    num_special_outputs = len(special_outputs_list[0])

                    for i in range(num_special_outputs):
                        # Collect the i-th special output from all slices
                        special_output_values = [slice_outputs[i] for slice_outputs in special_outputs_list]
                        combined_special_outputs.append(special_output_values)

                    # Return tuple: (stacked_main_output, combined_special_output1, combined_special_output2, ...)
                    result = (result, *combined_special_outputs)
            else:
                # Call the original function normally
                result = original_func(image, *args, **kwargs)

            # Apply dtype conversion based on enum value
            if hasattr(result, 'dtype') and dtype_conversion is not None:
                if dtype_conversion == DtypeConversion.PRESERVE_INPUT:
                    # Preserve input dtype
                    if result.dtype != original_dtype:
                        result = _scale_and_convert_numpy(result, original_dtype)
                elif dtype_conversion == DtypeConversion.NATIVE_OUTPUT:
                    # Return NumPy's native output dtype
                    pass  # No conversion needed
                else:
                    # Force specific dtype
                    target_dtype = dtype_conversion.numpy_dtype
                    if target_dtype is not None:
                        result = _scale_and_convert_numpy(result, target_dtype)

            return result
        except Exception as e:
            logger.error(f"Error in NumPy dtype/slice preserving wrapper for {func_name}: {e}")
            # Return original result on error
            return original_func(image, *args, **kwargs)

    # Update function signature to include new parameters
    try:
        original_sig = inspect.signature(original_func)
        new_params = list(original_sig.parameters.values())

        # Check if slice_by_slice parameter already exists
        param_names = [p.name for p in new_params]
        # Add dtype_conversion parameter first (before slice_by_slice)
        if 'dtype_conversion' not in param_names:
            dtype_param = inspect.Parameter(
                'dtype_conversion',
                inspect.Parameter.KEYWORD_ONLY,
                default=DtypeConversion.PRESERVE_INPUT,
                annotation=DtypeConversion
            )
            new_params.append(dtype_param)

        if 'slice_by_slice' not in param_names:
            # Add slice_by_slice parameter as keyword-only (after dtype_conversion)
            slice_param = inspect.Parameter(
                'slice_by_slice',
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool
            )
            new_params.append(slice_param)

        # Create new signature and override the @wraps signature
        new_sig = original_sig.replace(parameters=new_params)
        numpy_dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        numpy_dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        numpy_dtype_and_slice_preserving_wrapper.__annotations__['dtype_conversion'] = DtypeConversion
        numpy_dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool

    except Exception:
        # If signature modification fails, continue without it
        pass

    # Update docstring to mention slice_by_slice parameter
    original_doc = numpy_dtype_and_slice_preserving_wrapper.__doc__ or ""
    additional_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.

    dtype_conversion : DtypeConversion, optional (default: PRESERVE_INPUT)
        Controls output data type conversion:

        - PRESERVE_INPUT: Keep input dtype (uint16 → uint16)
        - NATIVE_OUTPUT: Use NumPy's native output dtype
        - UINT8: Force 8-bit unsigned integer (0-255 range)
        - UINT16: Force 16-bit unsigned integer (microscopy standard)
        - INT16: Force 16-bit signed integer
        - INT32: Force 32-bit signed integer
        - FLOAT32: Force 32-bit float (GPU performance)
        - FLOAT64: Force 64-bit float (maximum precision)
    """
    numpy_dtype_and_slice_preserving_wrapper.__doc__ = original_doc + additional_doc

    return numpy_dtype_and_slice_preserving_wrapper


def _create_cupy_dtype_preserving_wrapper(original_func, func_name):
    """
    Create a wrapper that preserves input data type and adds slice_by_slice parameter for CuPy functions.

    This uses the SAME pattern as scikit-image for consistency. CuPy functions generally preserve
    dtypes better than scikit-image, but this wrapper ensures consistent behavior and adds
    slice_by_slice parameter to avoid cross-slice contamination in 3D arrays.
    """
    import inspect
    from functools import wraps

    @wraps(original_func)
    def cupy_dtype_and_slice_preserving_wrapper(image, *args, dtype_conversion=None, slice_by_slice: bool = False, **kwargs):
        # Set default dtype_conversion if not provided and DtypeConversion is available
        if dtype_conversion is None and DtypeConversion is not None:
            dtype_conversion = DtypeConversion.PRESERVE_INPUT

        try:
            cupy = optional_import("cupy")
            if cupy is None:
                return original_func(image, *args, **kwargs)

            # Store original dtype
            original_dtype = image.dtype

            # Handle slice_by_slice processing for 3D arrays using OpenHCS stack utilities
            if slice_by_slice and hasattr(image, 'ndim') and image.ndim == 3:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type

                # Detect memory type and use proper OpenHCS utilities
                memory_type = _detect_memory_type(image)
                gpu_id = image.device.id if hasattr(image, 'device') else 0

                # Unstack 3D array into 2D slices
                slices_2d = unstack_slices(image, memory_type, gpu_id)

                # Process each slice and handle special outputs
                main_outputs = []
                special_outputs_list = []

                for slice_2d in slices_2d:
                    slice_result = original_func(slice_2d, *args, **kwargs)

                    # Check if result is a tuple (indicating special outputs)
                    if isinstance(slice_result, tuple):
                        main_outputs.append(slice_result[0])  # First element is main output
                        special_outputs_list.append(slice_result[1:])  # Rest are special outputs
                    else:
                        main_outputs.append(slice_result)  # Single output

                # Stack main outputs back into 3D array
                result = stack_slices(main_outputs, memory_type, gpu_id)

                # If we have special outputs, combine them and return tuple
                if special_outputs_list:
                    # Combine special outputs from all slices
                    combined_special_outputs = []
                    num_special_outputs = len(special_outputs_list[0])

                    for i in range(num_special_outputs):
                        # Collect the i-th special output from all slices
                        special_output_values = [slice_outputs[i] for slice_outputs in special_outputs_list]
                        combined_special_outputs.append(special_output_values)

                    # Return tuple: (stacked_main_output, combined_special_output1, combined_special_output2, ...)
                    result = (result, *combined_special_outputs)
            else:
                # Call the original function normally
                result = original_func(image, *args, **kwargs)

            # Apply dtype conversion based on enum value
            if hasattr(result, 'dtype') and dtype_conversion is not None:
                if dtype_conversion == DtypeConversion.PRESERVE_INPUT:
                    # Preserve input dtype
                    if result.dtype != original_dtype:
                        result = _scale_and_convert_cupy(result, original_dtype)
                elif dtype_conversion == DtypeConversion.NATIVE_OUTPUT:
                    # Return CuPy's native output dtype
                    pass  # No conversion needed
                else:
                    # Force specific dtype
                    target_dtype = dtype_conversion.numpy_dtype
                    if target_dtype is not None:
                        result = _scale_and_convert_cupy(result, target_dtype)

            return result
        except Exception as e:
            logger.error(f"Error in CuPy dtype/slice preserving wrapper for {func_name}: {e}")
            # Return original result on error
            return original_func(image, *args, **kwargs)

    # Update function signature to include new parameters
    try:
        original_sig = inspect.signature(original_func)
        new_params = list(original_sig.parameters.values())

        # Check if slice_by_slice parameter already exists
        param_names = [p.name for p in new_params]
        # Add dtype_conversion parameter first (before slice_by_slice)
        if 'dtype_conversion' not in param_names:
            dtype_param = inspect.Parameter(
                'dtype_conversion',
                inspect.Parameter.KEYWORD_ONLY,
                default=DtypeConversion.PRESERVE_INPUT,
                annotation=DtypeConversion
            )
            new_params.append(dtype_param)

        if 'slice_by_slice' not in param_names:
            # Add slice_by_slice parameter as keyword-only (after dtype_conversion)
            slice_param = inspect.Parameter(
                'slice_by_slice',
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool
            )
            new_params.append(slice_param)

        # Create new signature and override the @wraps signature
        new_sig = original_sig.replace(parameters=new_params)
        cupy_dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        cupy_dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        cupy_dtype_and_slice_preserving_wrapper.__annotations__['dtype_conversion'] = DtypeConversion
        cupy_dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool

    except Exception:
        # If signature modification fails, continue without it
        pass

    # Update docstring to mention slice_by_slice parameter
    original_doc = cupy_dtype_and_slice_preserving_wrapper.__doc__ or ""
    additional_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.

    dtype_conversion : DtypeConversion, optional (default: PRESERVE_INPUT)
        Controls output data type conversion:

        - PRESERVE_INPUT: Keep input dtype (uint16 → uint16)
        - NATIVE_OUTPUT: Use CuPy's native output dtype
        - UINT8: Force 8-bit unsigned integer (0-255 range)
        - UINT16: Force 16-bit unsigned integer (microscopy standard)
        - INT16: Force 16-bit signed integer
        - INT32: Force 32-bit signed integer
        - FLOAT32: Force 32-bit float (GPU performance)
        - FLOAT64: Force 64-bit float (maximum precision)
    """
    cupy_dtype_and_slice_preserving_wrapper.__doc__ = original_doc + additional_doc

    return cupy_dtype_and_slice_preserving_wrapper


def _create_torch_dtype_preserving_wrapper(original_func, func_name):
    """
    Create a wrapper that preserves input data type and adds slice_by_slice parameter for PyTorch functions.

    This follows the same pattern as existing dtype preservation wrappers for consistency.
    PyTorch functions generally preserve dtypes well, but this wrapper ensures consistent behavior
    and adds slice_by_slice parameter to avoid cross-slice contamination in 3D arrays.
    """
    import inspect
    from functools import wraps

    @wraps(original_func)
    def torch_dtype_and_slice_preserving_wrapper(image, *args, dtype_conversion=None, slice_by_slice: bool = False, **kwargs):
        # Set default dtype_conversion if not provided
        if dtype_conversion is None:
            dtype_conversion = DtypeConversion.PRESERVE_INPUT

        try:
            torch = optional_import("torch")
            if torch is None:
                return original_func(image, *args, **kwargs)

            # Store original dtype
            original_dtype = image.dtype if hasattr(image, 'dtype') else None

            # Handle slice_by_slice processing for 3D arrays
            if slice_by_slice and hasattr(image, 'ndim') and image.ndim == 3:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type

                # Detect memory type and use proper OpenHCS utilities
                memory_type = _detect_memory_type(image)
                gpu_id = image.device.index if hasattr(image, 'device') and image.device.type == 'cuda' else 0

                # Unstack 3D array into 2D slices
                slices_2d = unstack_slices(image, memory_type=memory_type, gpu_id=gpu_id)

                # Process each slice and handle special outputs
                main_outputs = []
                special_outputs_list = []

                for slice_2d in slices_2d:
                    slice_result = original_func(slice_2d, *args, **kwargs)

                    # Check if result is a tuple (indicating special outputs)
                    if isinstance(slice_result, tuple):
                        main_outputs.append(slice_result[0])  # First element is main output
                        special_outputs_list.append(slice_result[1:])  # Rest are special outputs
                    else:
                        main_outputs.append(slice_result)  # Single output

                # Stack main outputs back into 3D array
                result = stack_slices(main_outputs, memory_type=memory_type, gpu_id=gpu_id)

                # If we have special outputs, combine them and return tuple
                if special_outputs_list:
                    # Combine special outputs from all slices
                    combined_special_outputs = []
                    num_special_outputs = len(special_outputs_list[0])

                    for i in range(num_special_outputs):
                        # Collect the i-th special output from all slices
                        special_output_values = [slice_outputs[i] for slice_outputs in special_outputs_list]
                        combined_special_outputs.append(special_output_values)

                    # Return tuple: (stacked_main_output, combined_special_output1, combined_special_output2, ...)
                    result = (result, *combined_special_outputs)
            else:
                # Process normally
                result = original_func(image, *args, **kwargs)

            # Apply dtype conversion if result is a tensor and we have dtype conversion info
            if (hasattr(result, 'dtype') and hasattr(result, 'shape') and
                original_dtype is not None and dtype_conversion is not None):

                if dtype_conversion == DtypeConversion.PRESERVE_INPUT:
                    # Preserve input dtype
                    if result.dtype != original_dtype:
                        result = result.to(original_dtype)
                elif dtype_conversion == DtypeConversion.NATIVE_OUTPUT:
                    # Return PyTorch's native output dtype
                    pass  # No conversion needed
                else:
                    # Force specific dtype
                    target_dtype = dtype_conversion.numpy_dtype
                    if target_dtype is not None:
                        # Map numpy dtypes to torch dtypes
                        import numpy as np
                        numpy_to_torch = {
                            np.uint8: torch.uint8,
                            np.uint16: torch.int32,  # PyTorch doesn't have uint16, use int32
                            np.int16: torch.int16,
                            np.int32: torch.int32,
                            np.float32: torch.float32,
                            np.float64: torch.float64,
                        }
                        torch_dtype = numpy_to_torch.get(target_dtype)
                        if torch_dtype is not None:
                            result = result.to(torch_dtype)

            return result

        except Exception as e:
            logger.error(f"Error in PyTorch dtype/slice preserving wrapper for {func_name}: {e}")
            # Return original result on error
            return original_func(image, *args, **kwargs)

    # Update function signature to include new parameters
    try:
        original_sig = inspect.signature(original_func)
        new_params = list(original_sig.parameters.values())

        # Add dtype_conversion parameter first (before slice_by_slice)
        param_names = [p.name for p in new_params]
        if 'dtype_conversion' not in param_names:
            dtype_param = inspect.Parameter(
                'dtype_conversion',
                inspect.Parameter.KEYWORD_ONLY,
                default=DtypeConversion.PRESERVE_INPUT,
                annotation=DtypeConversion
            )
            new_params.append(dtype_param)

        # Add slice_by_slice parameter after dtype_conversion
        if 'slice_by_slice' not in param_names:
            slice_param = inspect.Parameter(
                'slice_by_slice',
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool
            )
            new_params.append(slice_param)

        new_sig = original_sig.replace(parameters=new_params)
        torch_dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        torch_dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        if DtypeConversion is not None:
            torch_dtype_and_slice_preserving_wrapper.__annotations__['dtype_conversion'] = DtypeConversion
        torch_dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool

    except Exception:
        # If signature modification fails, continue without it
        pass

    # Update docstring to mention new parameters
    original_doc = torch_dtype_and_slice_preserving_wrapper.__doc__ or ""
    additional_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.

    dtype_conversion : DtypeConversion, optional (default: PRESERVE_INPUT)
        Controls output data type conversion:

        - PRESERVE_INPUT: Keep input dtype (uint16 → uint16)
        - NATIVE_OUTPUT: Use PyTorch's native output dtype
        - UINT8: Force 8-bit unsigned integer (0-255 range)
        - UINT16: Force 16-bit unsigned integer (mapped to int32 in PyTorch)
        - INT16: Force 16-bit signed integer
        - INT32: Force 32-bit signed integer
        - FLOAT32: Force 32-bit float (GPU performance)
        - FLOAT64: Force 64-bit float (maximum precision)
    """
    torch_dtype_and_slice_preserving_wrapper.__doc__ = original_doc + additional_doc

    return torch_dtype_and_slice_preserving_wrapper


def _create_tensorflow_dtype_preserving_wrapper(original_func, func_name):
    """
    Create a wrapper that preserves input data type and adds slice_by_slice parameter for TensorFlow functions.

    This follows the same pattern as existing dtype preservation wrappers for consistency.
    TensorFlow functions generally preserve dtypes well, but this wrapper ensures consistent behavior
    and adds slice_by_slice parameter to avoid cross-slice contamination in 3D arrays.
    """
    import inspect
    from functools import wraps

    @wraps(original_func)
    def tensorflow_dtype_and_slice_preserving_wrapper(image, *args, dtype_conversion=None, slice_by_slice: bool = False, **kwargs):
        # Set default dtype_conversion if not provided
        if dtype_conversion is None:
            dtype_conversion = DtypeConversion.PRESERVE_INPUT

        try:
            tf = optional_import("tensorflow")
            if tf is None:
                return original_func(image, *args, **kwargs)

            # Store original dtype
            original_dtype = image.dtype if hasattr(image, 'dtype') else None

            # Handle slice_by_slice processing for 3D arrays
            if slice_by_slice and hasattr(image, 'ndim') and image.ndim == 3:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type

                # Detect memory type and use proper OpenHCS utilities
                memory_type = _detect_memory_type(image)
                gpu_id = 0  # TensorFlow manages GPU placement internally

                # Unstack 3D array into 2D slices
                slices_2d = unstack_slices(image, memory_type=memory_type, gpu_id=gpu_id)

                # Process each slice and handle special outputs
                main_outputs = []
                special_outputs_list = []

                for slice_2d in slices_2d:
                    slice_result = original_func(slice_2d, *args, **kwargs)

                    # Check if result is a tuple (indicating special outputs)
                    if isinstance(slice_result, tuple):
                        main_outputs.append(slice_result[0])  # First element is main output
                        special_outputs_list.append(slice_result[1:])  # Rest are special outputs
                    else:
                        main_outputs.append(slice_result)  # Single output

                # Stack main outputs back into 3D array
                result = stack_slices(main_outputs, memory_type=memory_type, gpu_id=gpu_id)

                # If we have special outputs, combine them and return tuple
                if special_outputs_list:
                    # Combine special outputs from all slices
                    combined_special_outputs = []
                    num_special_outputs = len(special_outputs_list[0])

                    for i in range(num_special_outputs):
                        # Collect the i-th special output from all slices
                        special_output_values = [slice_outputs[i] for slice_outputs in special_outputs_list]
                        combined_special_outputs.append(special_output_values)

                    # Return tuple: (stacked_main_output, combined_special_output1, combined_special_output2, ...)
                    result = (result, *combined_special_outputs)
            else:
                # Process normally
                result = original_func(image, *args, **kwargs)

            # Apply dtype conversion if result is a tensor and we have dtype conversion info
            if (hasattr(result, 'dtype') and hasattr(result, 'shape') and
                original_dtype is not None and dtype_conversion is not None):

                if dtype_conversion == DtypeConversion.PRESERVE_INPUT:
                    # Preserve input dtype
                    if result.dtype != original_dtype:
                        result = tf.cast(result, original_dtype)
                elif dtype_conversion == DtypeConversion.NATIVE_OUTPUT:
                    # Return TensorFlow's native output dtype
                    pass  # No conversion needed
                else:
                    # Force specific dtype
                    target_dtype = dtype_conversion.numpy_dtype
                    if target_dtype is not None:
                        # Convert numpy dtype to tensorflow dtype
                        import numpy as np
                        numpy_to_tf = {
                            np.uint8: tf.uint8,
                            np.uint16: tf.uint16,
                            np.int16: tf.int16,
                            np.int32: tf.int32,
                            np.float32: tf.float32,
                            np.float64: tf.float64,
                        }
                        tf_dtype = numpy_to_tf.get(target_dtype)
                        if tf_dtype is not None:
                            result = tf.cast(result, tf_dtype)

            return result

        except Exception as e:
            logger.error(f"Error in TensorFlow dtype/slice preserving wrapper for {func_name}: {e}")
            # Return original result on error
            return original_func(image, *args, **kwargs)

    # Update function signature to include new parameters
    try:
        original_sig = inspect.signature(original_func)
        new_params = list(original_sig.parameters.values())

        # Add slice_by_slice parameter if not already present
        param_names = [p.name for p in new_params]
        if 'slice_by_slice' not in param_names:
            slice_param = inspect.Parameter(
                'slice_by_slice',
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool
            )
            new_params.append(slice_param)

        # Add dtype_conversion parameter if DtypeConversion is available
        if DtypeConversion is not None and 'dtype_conversion' not in param_names:
            dtype_param = inspect.Parameter(
                'dtype_conversion',
                inspect.Parameter.KEYWORD_ONLY,
                default=DtypeConversion.PRESERVE_INPUT,
                annotation=DtypeConversion
            )
            new_params.append(dtype_param)

        new_sig = original_sig.replace(parameters=new_params)
        tensorflow_dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        tensorflow_dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        tensorflow_dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool
        if DtypeConversion is not None:
            tensorflow_dtype_and_slice_preserving_wrapper.__annotations__['dtype_conversion'] = DtypeConversion

    except Exception:
        # If signature modification fails, continue without it
        pass

    # Update docstring to mention new parameters
    original_doc = tensorflow_dtype_and_slice_preserving_wrapper.__doc__ or ""
    additional_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.

    dtype_conversion : DtypeConversion, optional (default: PRESERVE_INPUT)
        Controls output data type conversion:

        - PRESERVE_INPUT: Keep input dtype (uint16 → uint16)
        - NATIVE_OUTPUT: Use TensorFlow's native output dtype
        - UINT8: Force 8-bit unsigned integer (0-255 range)
        - UINT16: Force 16-bit unsigned integer (microscopy standard)
        - INT16: Force 16-bit signed integer
        - INT32: Force 32-bit signed integer
        - FLOAT32: Force 32-bit float (GPU performance)
        - FLOAT64: Force 64-bit float (maximum precision)
    """
    tensorflow_dtype_and_slice_preserving_wrapper.__doc__ = original_doc + additional_doc

    return tensorflow_dtype_and_slice_preserving_wrapper


def _create_jax_dtype_preserving_wrapper(original_func, func_name):
    """
    Create a wrapper that preserves input data type and adds slice_by_slice parameter for JAX functions.

    This follows the same pattern as existing dtype preservation wrappers for consistency.
    JAX functions generally preserve dtypes well, but this wrapper ensures consistent behavior
    and adds slice_by_slice parameter to avoid cross-slice contamination in 3D arrays.
    """
    import inspect
    from functools import wraps

    @wraps(original_func)
    def jax_dtype_and_slice_preserving_wrapper(image, *args, dtype_conversion=None, slice_by_slice: bool = False, **kwargs):
        # Set default dtype_conversion if not provided
        if dtype_conversion is None:
            dtype_conversion = DtypeConversion.PRESERVE_INPUT

        try:
            jax = optional_import("jax")
            jnp = optional_import("jax.numpy") if jax is not None else None
            if jax is None or jnp is None:
                return original_func(image, *args, **kwargs)

            # Store original dtype
            original_dtype = image.dtype if hasattr(image, 'dtype') else None

            # Handle slice_by_slice processing for 3D arrays
            if slice_by_slice and hasattr(image, 'ndim') and image.ndim == 3:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type

                # Detect memory type and use proper OpenHCS utilities
                memory_type = _detect_memory_type(image)
                gpu_id = 0  # JAX manages GPU placement internally

                # Unstack 3D array into 2D slices
                slices_2d = unstack_slices(image, memory_type=memory_type, gpu_id=gpu_id)

                # Process each slice and handle special outputs
                main_outputs = []
                special_outputs_list = []

                for slice_2d in slices_2d:
                    slice_result = original_func(slice_2d, *args, **kwargs)

                    # Check if result is a tuple (indicating special outputs)
                    if isinstance(slice_result, tuple):
                        main_outputs.append(slice_result[0])  # First element is main output
                        special_outputs_list.append(slice_result[1:])  # Rest are special outputs
                    else:
                        main_outputs.append(slice_result)  # Single output

                # Stack main outputs back into 3D array
                result = stack_slices(main_outputs, memory_type=memory_type, gpu_id=gpu_id)

                # If we have special outputs, combine them and return tuple
                if special_outputs_list:
                    # Combine special outputs from all slices
                    combined_special_outputs = []
                    num_special_outputs = len(special_outputs_list[0])

                    for i in range(num_special_outputs):
                        # Collect the i-th special output from all slices
                        special_output_values = [slice_outputs[i] for slice_outputs in special_outputs_list]
                        combined_special_outputs.append(special_output_values)

                    # Return tuple: (stacked_main_output, combined_special_output1, combined_special_output2, ...)
                    result = (result, *combined_special_outputs)
            else:
                # Process normally
                result = original_func(image, *args, **kwargs)

            # Apply dtype conversion if result is an array and we have dtype conversion info
            if (hasattr(result, 'dtype') and hasattr(result, 'shape') and
                original_dtype is not None and dtype_conversion is not None):

                if dtype_conversion == DtypeConversion.PRESERVE_INPUT:
                    # Preserve input dtype
                    if result.dtype != original_dtype:
                        result = result.astype(original_dtype)
                elif dtype_conversion == DtypeConversion.NATIVE_OUTPUT:
                    # Return JAX's native output dtype
                    pass  # No conversion needed
                else:
                    # Force specific dtype
                    target_dtype = dtype_conversion.numpy_dtype
                    if target_dtype is not None:
                        # JAX uses numpy-compatible dtypes
                        result = result.astype(target_dtype)

            return result

        except Exception as e:
            logger.error(f"Error in JAX dtype/slice preserving wrapper for {func_name}: {e}")
            # Return original result on error
            return original_func(image, *args, **kwargs)

    # Update function signature to include new parameters
    try:
        original_sig = inspect.signature(original_func)
        new_params = list(original_sig.parameters.values())

        # Add slice_by_slice parameter if not already present
        param_names = [p.name for p in new_params]
        if 'slice_by_slice' not in param_names:
            slice_param = inspect.Parameter(
                'slice_by_slice',
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool
            )
            new_params.append(slice_param)

        # Add dtype_conversion parameter if DtypeConversion is available
        if DtypeConversion is not None and 'dtype_conversion' not in param_names:
            dtype_param = inspect.Parameter(
                'dtype_conversion',
                inspect.Parameter.KEYWORD_ONLY,
                default=DtypeConversion.PRESERVE_INPUT,
                annotation=DtypeConversion
            )
            new_params.append(dtype_param)

        new_sig = original_sig.replace(parameters=new_params)
        jax_dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        jax_dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        jax_dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool
        if DtypeConversion is not None:
            jax_dtype_and_slice_preserving_wrapper.__annotations__['dtype_conversion'] = DtypeConversion

    except Exception:
        # If signature modification fails, continue without it
        pass

    # Update docstring to mention new parameters
    original_doc = jax_dtype_and_slice_preserving_wrapper.__doc__ or ""
    additional_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.

    dtype_conversion : DtypeConversion, optional (default: PRESERVE_INPUT)
        Controls output data type conversion:

        - PRESERVE_INPUT: Keep input dtype (uint16 → uint16)
        - NATIVE_OUTPUT: Use JAX's native output dtype
        - UINT8: Force 8-bit unsigned integer (0-255 range)
        - UINT16: Force 16-bit unsigned integer (microscopy standard)
        - INT16: Force 16-bit signed integer
        - INT32: Force 32-bit signed integer
        - FLOAT32: Force 32-bit float (GPU performance)
        - FLOAT64: Force 64-bit float (maximum precision)
    """
    jax_dtype_and_slice_preserving_wrapper.__doc__ = original_doc + additional_doc

    return jax_dtype_and_slice_preserving_wrapper


def _create_pyclesperanto_dtype_preserving_wrapper(original_func, func_name):
    """
    Create a wrapper that ensures array-in/array-out compliance and dtype preservation for pyclesperanto functions.

    All OpenHCS functions must:
    1. Take 3D pyclesperanto array as first argument
    2. Return 3D pyclesperanto array as first output
    3. Additional outputs (values, coordinates) as 2nd, 3rd, etc. returns
    4. Preserve input dtype when appropriate
    """
    import inspect
    from functools import wraps

    @wraps(original_func)
    def pyclesperanto_dtype_and_slice_preserving_wrapper(image_3d, *args, dtype_conversion=None, slice_by_slice: bool = False, **kwargs):
        # Set default dtype_conversion if not provided
        if dtype_conversion is None:
            dtype_conversion = DtypeConversion.PRESERVE_INPUT

        try:
            cle = optional_import("pyclesperanto")
            if cle is None:
                return original_func(image_3d, *args, **kwargs)

            # Store original dtype for preservation
            original_dtype = image_3d.dtype

            # Handle slice_by_slice processing for 3D arrays using OpenHCS stack utilities
            if slice_by_slice and hasattr(image_3d, 'ndim') and image_3d.ndim == 3:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type

                # Detect memory type and use proper OpenHCS utilities
                memory_type = _detect_memory_type(image_3d)
                gpu_id = 0  # pyclesperanto manages GPU internally

                # Process each slice and handle special outputs
                slices = unstack_slices(image_3d, memory_type, gpu_id)
                main_outputs = []
                special_outputs_list = []

                for slice_2d in slices:
                    # Apply function to 2D slice
                    result_slice = original_func(slice_2d, *args, **kwargs)

                    # Check if result is a tuple (indicating special outputs)
                    if isinstance(result_slice, tuple):
                        main_outputs.append(result_slice[0])  # First element is main output
                        special_outputs_list.append(result_slice[1:])  # Rest are special outputs
                    else:
                        main_outputs.append(result_slice)  # Single output

                # Stack main outputs back into 3D array
                result = stack_slices(main_outputs, memory_type, gpu_id)

                # If we have special outputs, combine them and return tuple
                if special_outputs_list:
                    # Combine special outputs from all slices
                    combined_special_outputs = []
                    num_special_outputs = len(special_outputs_list[0])

                    for i in range(num_special_outputs):
                        # Collect the i-th special output from all slices
                        special_output_values = [slice_outputs[i] for slice_outputs in special_outputs_list]
                        combined_special_outputs.append(special_output_values)

                    # Return tuple: (stacked_main_output, combined_special_output1, combined_special_output2, ...)
                    result = (result, *combined_special_outputs)
            else:
                # Normal 3D processing
                result = original_func(image_3d, *args, **kwargs)

            # Check if result is 2D and needs expansion to 3D
            if hasattr(result, 'ndim') and result.ndim == 2:
                # Expand 2D result to 3D single slice
                try:
                    # Concatenate with itself to create 3D, then take first slice
                    temp_3d = cle.concatenate_along_z(result, result)  # Creates (2, Y, X)
                    result = temp_3d[0:1, :, :]  # Take first slice to get (1, Y, X)
                except Exception:
                    # If expansion fails, return original 2D result
                    # This maintains backward compatibility
                    pass

            # Apply dtype conversion based on enum value
            if hasattr(result, 'dtype') and hasattr(result, 'shape') and dtype_conversion is not None:
                if dtype_conversion == DtypeConversion.PRESERVE_INPUT:
                    # Preserve input dtype
                    if result.dtype != original_dtype:
                        return _scale_and_convert_pyclesperanto(result, original_dtype)
                    return result

                elif dtype_conversion == DtypeConversion.NATIVE_OUTPUT:
                    # Return pyclesperanto's native output dtype
                    return result

                else:
                    # Force specific dtype
                    target_dtype = dtype_conversion.numpy_dtype
                    if target_dtype is not None and result.dtype != target_dtype:
                        return _scale_and_convert_pyclesperanto(result, target_dtype)
                    return result
            else:
                # Non-array result, return as-is
                return result

        except Exception as e:
            logger.error(f"Error in pyclesperanto dtype/slice preserving wrapper for {func_name}: {e}")
            # If anything goes wrong, fall back to original function
            return original_func(image_3d, *args, **kwargs)

    # Update function signature to include new parameters
    try:
        original_sig = inspect.signature(original_func)
        new_params = list(original_sig.parameters.values())

        # Add slice_by_slice parameter if not already present
        param_names = [p.name for p in new_params]
        if 'slice_by_slice' not in param_names:
            slice_param = inspect.Parameter(
                'slice_by_slice',
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool
            )
            new_params.append(slice_param)

        # Add dtype_conversion parameter if DtypeConversion is available
        if DtypeConversion is not None and 'dtype_conversion' not in param_names:
            dtype_param = inspect.Parameter(
                'dtype_conversion',
                inspect.Parameter.KEYWORD_ONLY,
                default=DtypeConversion.PRESERVE_INPUT,
                annotation=DtypeConversion
            )
            new_params.append(dtype_param)

        new_sig = original_sig.replace(parameters=new_params)
        pyclesperanto_dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        pyclesperanto_dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        pyclesperanto_dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool
        if DtypeConversion is not None:
            pyclesperanto_dtype_and_slice_preserving_wrapper.__annotations__['dtype_conversion'] = DtypeConversion

    except Exception:
        # If signature modification fails, continue without it
        pass

    # Update docstring to mention additional parameters
    original_doc = pyclesperanto_dtype_and_slice_preserving_wrapper.__doc__ or ""
    additional_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.

    dtype_conversion : DtypeConversion, optional (default: PRESERVE_INPUT)
        Controls output data type conversion:

        - PRESERVE_INPUT: Keep input dtype (uint16 → uint16)
        - NATIVE_OUTPUT: Use pyclesperanto's native output (often float32)
        - UINT8: Force 8-bit unsigned integer (0-255 range)
        - UINT16: Force 16-bit unsigned integer (microscopy standard)
        - INT16: Force 16-bit signed integer
        - INT32: Force 32-bit signed integer
        - FLOAT32: Force 32-bit float (GPU performance)
        - FLOAT64: Force 64-bit float (maximum precision)
    """
    pyclesperanto_dtype_and_slice_preserving_wrapper.__doc__ = original_doc + additional_doc

    return pyclesperanto_dtype_and_slice_preserving_wrapper


def _scale_and_convert_pyclesperanto(result, target_dtype):
    """
    Scale and convert pyclesperanto array to target dtype.
    This is a simplified version of the helper function from pyclesperanto_registry.py
    """
    try:
        cle = optional_import("pyclesperanto")
        if cle is None:
            return result

        import numpy as np

        # If result is floating point and target is integer, scale appropriately
        if np.issubdtype(result.dtype, np.floating) and not np.issubdtype(target_dtype, np.floating):
            # Convert to numpy for scaling, then back to pyclesperanto
            result_np = cle.pull(result)

            # Clip to [0, 1] range and scale to integer range
            clipped = np.clip(result_np, 0, 1)
            if target_dtype == np.uint8:
                scaled = (clipped * 255).astype(target_dtype)
            elif target_dtype == np.uint16:
                scaled = (clipped * 65535).astype(target_dtype)
            elif target_dtype == np.uint32:
                scaled = (clipped * 4294967295).astype(target_dtype)
            else:
                # For other integer types, just convert without scaling
                scaled = clipped.astype(target_dtype)

            # Push back to GPU
            return cle.push(scaled)
        else:
            # Direct conversion for same numeric type families
            result_np = cle.pull(result)
            converted = result_np.astype(target_dtype)
            return cle.push(converted)

    except Exception:
        # If conversion fails, return original result
        return result
