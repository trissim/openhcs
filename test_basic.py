#!/usr/bin/env python3
"""
Basic test to verify Python environment.
"""

print("Testing basic Python functionality...")
print("Python is working!")

try:
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[0]}")
except Exception as e:
    print(f"Error: {e}")

print("Basic test complete.")
