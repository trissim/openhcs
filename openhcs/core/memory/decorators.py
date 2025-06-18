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

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

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
    decorator = memory_types(input_type=input_type, output_type=output_type)

    # Handle both @numpy and @numpy(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def cupy(func: Optional[F] = None, *, input_type: str = "cupy", output_type: str = "cupy") -> Any:
    """
    Decorator that declares a function as operating on cupy arrays.

    This decorator provides automatic thread-local CUDA stream management for
    true parallelization across multiple threads. Each thread gets its own
    persistent CUDA stream that is reused for all CuPy operations.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "cupy")
        output_type: The memory type for the function's output (default: "cupy")

    Returns:
        The decorated function with memory type attributes and stream management

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Add CUDA stream wrapper if CuPy is available (lazy import)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cp = _get_cupy()
            if cp is not None and hasattr(cp, 'cuda'):
                # Get unified thread context and CuPy stream
                gpu_context = get_thread_gpu_context()
                cupy_stream = gpu_context.get_cupy_stream()

                if cupy_stream is not None:
                    # Execute function in stream context
                    with cupy_stream:
                        return func(*args, **kwargs)
                else:
                    # No CUDA available, execute without stream
                    return func(*args, **kwargs)
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
    output_type: str = "torch"
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

    Returns:
        The decorated function with memory type attributes and stream management

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Add CUDA stream wrapper if PyTorch is available and CUDA is available (lazy import)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            torch = _get_torch()
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                # Get unified thread context and PyTorch stream
                gpu_context = get_thread_gpu_context()
                torch_stream = gpu_context.get_torch_stream()

                if torch_stream is not None:
                    # Execute function in stream context
                    with torch.cuda.stream(torch_stream):
                        return func(*args, **kwargs)
                else:
                    # No CUDA available, execute without stream
                    return func(*args, **kwargs)
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
    output_type: str = "tensorflow"
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

    Returns:
        The decorated function with memory type attributes and device management

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Add device context wrapper if TensorFlow is available and GPU is available (lazy import)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tf = _get_tensorflow()
            if tf is not None and tf.config.list_physical_devices('GPU'):
                # Use GPU device context for thread isolation
                # TensorFlow manages internal CUDA streams automatically
                with tf.device('/GPU:0'):
                    return func(*args, **kwargs)
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
    output_type: str = "jax"
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

    Returns:
        The decorated function with memory type attributes and device management

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    def decorator(func: F) -> F:
        # Set memory type attributes
        memory_decorator = memory_types(input_type=input_type, output_type=output_type)
        func = memory_decorator(func)

        # Add device placement wrapper if JAX is available and GPU is available (lazy import)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            jax_module = _get_jax()
            if jax_module is not None:
                devices = jax_module.devices()
                gpu_devices = [d for d in devices if d.platform == 'gpu']

                if gpu_devices:
                    # Use GPU device placement for thread isolation
                    # JAX/XLA manages internal CUDA streams automatically
                    with jax_module.default_device(gpu_devices[0]):
                        return func(*args, **kwargs)
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
    output_type: str = "pyclesperanto"
) -> Any:
    """
    Decorator that declares a function as operating on pyclesperanto GPU arrays.

    This is a convenience wrapper around memory_types with pyclesperanto defaults.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "pyclesperanto")
        output_type: The memory type for the function's output (default: "pyclesperanto")

    Returns:
        The decorated function with memory type attributes set

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    decorator = memory_types(input_type=input_type, output_type=output_type)

    # Handle both @pyclesperanto and @pyclesperanto(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)
