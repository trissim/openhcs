#!/usr/bin/env python3
"""
Debug script to check memory backend state during pipeline execution.
"""

import sys
sys.path.insert(0, '.')

from openhcs.io.base import storage_registry
from openhcs.constants.constants import Backend

def debug_memory_backend():
    """Debug the current state of the memory backend."""
    memory_backend = storage_registry[Backend.MEMORY.value]
    
    print("=== MEMORY BACKEND DEBUG ===")
    print(f"Total entries: {len(memory_backend._memory_store)}")
    
    if not memory_backend._memory_store:
        print("Memory backend is empty!")
        return
    
    print("\nAll entries:")
    for key, value in memory_backend._memory_store.items():
        if value is None:
            print(f"  DIR:  {key}")
        else:
            value_type = type(value).__name__
            if hasattr(value, 'shape'):
                print(f"  FILE: {key} ({value_type}, shape: {value.shape})")
            else:
                print(f"  FILE: {key} ({value_type})")
    
    print("\nFiles by well:")
    files_by_well = {}
    for key, value in memory_backend._memory_store.items():
        if value is not None:  # Only files, not directories
            # Try to extract well from path
            parts = key.split('/')
            for part in parts:
                if len(part) == 3 and part[0].isalpha() and part[1:].isdigit():
                    # Looks like a well (e.g., B02, B03)
                    well = part
                    if well not in files_by_well:
                        files_by_well[well] = []
                    files_by_well[well].append(key)
                    break
    
    for well, files in sorted(files_by_well.items()):
        print(f"  {well}: {len(files)} files")
        for file in files[:3]:  # Show first 3 files
            print(f"    {file}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")

if __name__ == "__main__":
    debug_memory_backend()
