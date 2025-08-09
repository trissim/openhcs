"""
Utility functions for the OpenHCS package.
"""

import functools
import logging
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class _ModulePlaceholder:
    """
    Placeholder for missing optional modules that allows attribute access
    for type annotations while still being falsy and failing on actual use.
    """
    def __init__(self, module_name: str):
        self._module_name = module_name

    def __bool__(self):
        return False

    def __getattr__(self, name):
        # Return another placeholder for chained attribute access
        # This allows things like cp.ndarray in type annotations to work
        return _ModulePlaceholder(f"{self._module_name}.{name}")

    def __call__(self, *args, **kwargs):
        # If someone tries to actually call a function, fail loudly
        raise ImportError(f"Module '{self._module_name}' is not available. Please install the required dependency.")

    def __repr__(self):
        return f"<ModulePlaceholder for '{self._module_name}'>"


def optional_import(module_name: str) -> Optional[Any]:
    """
    Import a module if available, otherwise return a placeholder that handles
    attribute access gracefully for type annotations but fails on actual use.

    This function allows for graceful handling of optional dependencies.
    It can be used to import libraries that may not be installed,
    particularly GPU-related libraries like torch, tensorflow, and cupy.

    Args:
        module_name: Name of the module to import

    Returns:
        The imported module if available, a placeholder otherwise

    Example:
        ```python
        # Import torch if available
        torch = optional_import("torch")

        # Check if torch is available before using it
        if torch:
            # Use torch
            tensor = torch.tensor([1, 2, 3])
        else:
            # Handle the case where torch is not available
            raise ImportError("PyTorch is required for this function")
        ```
    """
    try:
        # Use importlib.import_module which handles dotted names properly
        import importlib
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Return a placeholder that handles attribute access gracefully
        return _ModulePlaceholder(module_name)

# Global thread activity tracking
thread_activity = defaultdict(list)
active_threads = set()
thread_lock = threading.Lock()

def get_thread_activity() -> Dict[int, List[Dict[str, Any]]]:
    """
    Get the current thread activity data.

    Returns:
        Dict mapping thread IDs to lists of activity records
    """
    return thread_activity

def get_active_threads() -> set:
    """
    Get the set of currently active thread IDs.

    Returns:
        Set of active thread IDs
    """
    return active_threads

def clear_thread_activity():
    """Clear all thread activity data."""
    with thread_lock:
        thread_activity.clear()
        active_threads.clear()

def track_thread_activity(func: Optional[Callable] = None, *, log_level: str = "info"):
    """
    Decorator to track thread activity for a function.

    Args:
        func: The function to decorate
        log_level: Logging level to use ("debug", "info", "warning", "error")

    Returns:
        Decorated function that tracks thread activity
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get thread information
            thread_id = threading.get_ident()
            thread_name = threading.current_thread().name

            # Record thread start time
            start_time = time.time()

            # Extract function name and arguments for context
            func_name = f.__name__
            # Get the first argument if it's a method (self or cls)
            context = ""
            if args and hasattr(args[0], "__class__"):
                if hasattr(args[0].__class__, func_name):
                    # It's likely a method, extract class name
                    context = f"{args[0].__class__.__name__}."

            # Extract well information if present in kwargs or args
            well = kwargs.get('well', None)
            if well is None and len(args) > 1 and isinstance(args[1], str):
                # Assume second argument might be well in methods like process_well(self, well, ...)
                well = args[1]

            # Add this thread to active threads
            with thread_lock:
                active_threads.add(thread_id)
                # Record the number of active threads at this moment
                thread_activity[thread_id].append({
                    'well': well,
                    'thread_name': thread_name,
                    'time': time.time(),
                    'action': 'start',
                    'function': f"{context}{func_name}",
                    'active_threads': len(active_threads)
                })

            # Log the start of the function
            log_func = getattr(logger, log_level.lower())
            log_func(f"Thread {thread_name} (ID: {thread_id}) started {context}{func_name} for well {well}")
            log_func(f"Active threads: {len(active_threads)}")

            try:
                # Call the original function
                result = f(*args, **kwargs)
                return result
            finally:
                # Record thread end time
                end_time = time.time()
                duration = end_time - start_time

                # Remove this thread from active threads
                with thread_lock:
                    active_threads.remove(thread_id)
                    # Record the number of active threads at this moment
                    thread_activity[thread_id].append({
                        'well': well,
                        'thread_name': thread_name,
                        'time': time.time(),
                        'action': 'end',
                        'function': f"{context}{func_name}",
                        'duration': duration,
                        'active_threads': len(active_threads)
                    })

                log_func(f"Thread {thread_name} (ID: {thread_id}) finished {context}{func_name} for well {well} in {duration:.2f} seconds")
                log_func(f"Active threads: {len(active_threads)}")

        return wrapper

    # Handle both @track_thread_activity and @track_thread_activity(log_level="debug")
    if func is None:
        return decorator
    return decorator(func)

def analyze_thread_activity():
    """
    Analyze thread activity data and return a report.

    Returns:
        Dict containing analysis results
    """
    max_concurrent = 0
    thread_starts = []
    thread_ends = []

    for thread_id, activities in thread_activity.items():
        for activity in activities:
            max_concurrent = max(max_concurrent, activity['active_threads'])
            if activity['action'] == 'start':
                thread_starts.append((
                    activity.get('well'),
                    activity['thread_name'],
                    activity['time'],
                    activity.get('function', '')
                ))
            else:  # 'end'
                thread_ends.append((
                    activity.get('well'),
                    activity['thread_name'],
                    activity['time'],
                    activity.get('duration', 0),
                    activity.get('function', '')
                ))

    # Sort by time
    thread_starts.sort(key=lambda x: x[2])
    thread_ends.sort(key=lambda x: x[2])

    # Find overlapping time periods
    overlaps = []
    for i, (well1, thread1, start1, func1) in enumerate(thread_starts):
        # Find the end time for this thread
        end1 = None
        for w, t, end, d, f in thread_ends:
            if t == thread1 and w == well1 and f == func1:
                end1 = end
                break

        if end1 is None:
            continue  # Skip if we can't find the end time

        # Check for overlaps with other threads
        for j, (well2, thread2, start2, func2) in enumerate(thread_starts):
            if i == j or thread1 == thread2:  # Skip same thread
                continue

            # Find the end time for the other thread
            end2 = None
            for w, t, end, d, f in thread_ends:
                if t == thread2 and w == well2 and f == func2:
                    end2 = end
                    break

            if end2 is None:
                continue  # Skip if we can't find the end time

            # Check if there's an overlap
            if start1 < end2 and start2 < end1:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0:
                    overlaps.append({
                        'thread1': thread1,
                        'well1': well1,
                        'function1': func1,
                        'thread2': thread2,
                        'well2': well2,
                        'function2': func2,
                        'duration': overlap_duration
                    })

    return {
        'max_concurrent': max_concurrent,
        'thread_starts': thread_starts,
        'thread_ends': thread_ends,
        'overlaps': overlaps
    }

def print_thread_activity_report():
    """Print a detailed report of thread activity."""
    analysis = analyze_thread_activity()

    print("\n" + "=" * 80)
    print("Thread Activity Report")
    print("=" * 80)

    print("\nThread Start Events:")
    for well, thread_name, time_val, func in analysis['thread_starts']:
        print(f"Thread {thread_name} started {func} for well {well} at {time_val:.2f}")

    print("\nThread End Events:")
    for well, thread_name, time_val, duration, func in analysis['thread_ends']:
        print(f"Thread {thread_name} finished {func} for well {well} at {time_val:.2f} (duration: {duration:.2f}s)")

    print("\nOverlap Analysis:")
    for overlap in analysis['overlaps']:
        print(f"Threads {overlap['thread1']} and {overlap['thread2']} overlapped for {overlap['duration']:.2f}s")
        print(f"  {overlap['thread1']} was processing {overlap['function1']} for well {overlap['well1']}")
        print(f"  {overlap['thread2']} was processing {overlap['function2']} for well {overlap['well2']}")

    print(f"\nFound {len(analysis['overlaps'])} thread overlaps")
    print(f"Maximum concurrent threads: {analysis['max_concurrent']}")
    print("=" * 80)

    return analysis


# Natural sorting utilities
def natural_sort_key(text: Union[str, Path]) -> List[Union[str, int]]:
    """
    Generate a natural sorting key for a string or Path.

    This function converts a string into a list of strings and integers
    that can be used as a sorting key to achieve natural (human-friendly)
    sorting order.

    Args:
        text: String or Path to generate sorting key for

    Returns:
        List of strings and integers for natural sorting

    Examples:
        >>> natural_sort_key("file10.txt")
        ['file', 10, '.txt']
        >>> natural_sort_key("A01_s001_w1_z001.tif")
        ['A', 1, '_s', 1, '_w', 1, '_z', 1, '.tif']
    """
    text = str(text)

    # Split on sequences of digits, keeping the digits
    parts = re.split(r'(\d+)', text)

    # Convert digit sequences to integers, leave other parts as strings
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part)

    return result


def natural_sort(items: List[Union[str, Path]]) -> List[Union[str, Path]]:
    """
    Sort a list of strings or Paths using natural sorting.

    Args:
        items: List of strings or Paths to sort

    Returns:
        New list sorted in natural order

    Examples:
        >>> natural_sort(["file1.txt", "file10.txt", "file2.txt"])
        ['file1.txt', 'file2.txt', 'file10.txt']
        >>> natural_sort(["A01_s001.tif", "A01_s010.tif", "A01_s002.tif"])
        ['A01_s001.tif', 'A01_s002.tif', 'A01_s010.tif']
    """
    return sorted(items, key=natural_sort_key)


def natural_sort_inplace(items: List[Union[str, Path]]) -> None:
    """
    Sort a list of strings or Paths using natural sorting in-place.

    Args:
        items: List of strings or Paths to sort in-place

    Examples:
        >>> files = ["file1.txt", "file10.txt", "file2.txt"]
        >>> natural_sort_inplace(files)
        >>> files
        ['file1.txt', 'file2.txt', 'file10.txt']
    """
    items.sort(key=natural_sort_key)


