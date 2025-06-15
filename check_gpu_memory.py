#!/usr/bin/env python3
"""
GPU Memory Checker for OpenHCS

Quick utility to check GPU memory usage and perform cleanup if needed.
Run this after your pipeline to see if VRAM is being properly cleared.

Usage:
    python check_gpu_memory.py                    # Just check memory
    python check_gpu_memory.py --cleanup          # Check + cleanup
    python check_gpu_memory.py --force-cleanup    # Nuclear cleanup option
"""

import argparse
import sys
import os

# Add OpenHCS to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Check GPU memory usage and optionally clean up")
    parser.add_argument("--cleanup", action="store_true", help="Perform standard GPU cleanup")
    parser.add_argument("--force-cleanup", action="store_true", help="Perform comprehensive cleanup (nuclear option)")
    
    args = parser.parse_args()
    
    try:
        from openhcs.core.memory.gpu_cleanup import (
            check_gpu_memory_usage, 
            cleanup_all_gpu_frameworks, 
            force_comprehensive_cleanup
        )
        
        print("🔍 GPU Memory Usage Check")
        print("=" * 50)
        check_gpu_memory_usage()
        
        if args.force_cleanup:
            print("\n🧹 Performing FORCE comprehensive cleanup...")
            force_comprehensive_cleanup()
            
        elif args.cleanup:
            print("\n🧹 Performing standard GPU cleanup...")
            cleanup_all_gpu_frameworks()
            
            # Check again after cleanup
            print("\n🔍 After cleanup:")
            check_gpu_memory_usage()
            
        else:
            print("\n💡 Tip: Use --cleanup or --force-cleanup to clear GPU memory")
            
    except ImportError as e:
        print(f"❌ Error importing OpenHCS modules: {e}")
        print("Make sure you're running this from the OpenHCS directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
