"""
Utility functions for the OpenHCS package.
"""

import functools
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

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




