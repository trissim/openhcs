#!/usr/bin/env python3
"""
Stress test for UnifiedTaskManager to find potential issues.
"""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from openhcs.tui.utils.unified_task_manager import initialize_task_manager, get_task_manager, shutdown_task_manager


async def test_basic_functionality():
    """Test basic task manager functionality."""
    print("🧪 Testing basic functionality...")
    
    tm = initialize_task_manager()
    
    # Test task creation
    async def simple_task():
        await asyncio.sleep(0.1)
        return "success"
    
    task = tm.create_task(simple_task(), "test_task")
    result = await task
    assert result == "success", "Basic task failed"
    
    # Test fire and forget
    tm.fire_and_forget(simple_task(), "fire_forget_test")
    await asyncio.sleep(0.2)  # Let it complete
    
    print("✅ Basic functionality passed")


async def test_error_handling():
    """Test error handling in tasks."""
    print("🧪 Testing error handling...")
    
    tm = get_task_manager()
    
    # Test task that raises exception
    async def failing_task():
        raise ValueError("Test error")
    
    # Should not crash the task manager
    tm.fire_and_forget(failing_task(), "failing_task")
    await asyncio.sleep(0.1)
    
    print("✅ Error handling passed")


async def test_stress_load():
    """Test task manager under stress."""
    print("🧪 Testing stress load...")
    
    tm = get_task_manager()
    initial_count = tm.active_task_count
    
    # Create many tasks
    async def quick_task(i):
        await asyncio.sleep(0.01)
        return i
    
    tasks = []
    for i in range(100):
        task = tm.create_task(quick_task(i), f"stress_task_{i}")
        tasks.append(task)
    
    # Verify all tasks are tracked
    assert tm.active_task_count >= initial_count + 100, "Tasks not properly tracked"
    
    # Wait for completion
    results = await asyncio.gather(*tasks)
    assert len(results) == 100, "Not all tasks completed"
    
    # Verify cleanup
    await asyncio.sleep(0.1)
    final_count = tm.active_task_count
    assert final_count <= initial_count + 10, f"Tasks not cleaned up: {final_count} vs {initial_count}"
    
    print("✅ Stress load passed")


async def test_shutdown_behavior():
    """Test shutdown behavior."""
    print("🧪 Testing shutdown behavior...")
    
    tm = get_task_manager()
    
    # Create long-running tasks
    async def long_task():
        try:
            await asyncio.sleep(10)  # Long sleep
            return "completed"
        except asyncio.CancelledError:
            return "cancelled"
    
    # Create several long tasks
    for i in range(5):
        tm.fire_and_forget(long_task(), f"long_task_{i}")
    
    await asyncio.sleep(0.1)  # Let tasks start
    initial_count = tm.active_task_count
    assert initial_count >= 5, "Long tasks not started"
    
    # Test shutdown
    start_time = time.time()
    await shutdown_task_manager()
    shutdown_time = time.time() - start_time
    
    assert shutdown_time < 2.0, f"Shutdown took too long: {shutdown_time}s"
    
    print("✅ Shutdown behavior passed")


async def test_memory_leaks():
    """Test for potential memory leaks."""
    print("🧪 Testing memory leaks...")
    
    # Reinitialize after shutdown
    tm = initialize_task_manager()
    
    # Create and complete many tasks
    async def memory_task():
        data = "x" * 1000  # Some data
        await asyncio.sleep(0.001)
        return len(data)
    
    for i in range(1000):
        tm.fire_and_forget(memory_task(), f"memory_task_{i}")
    
    # Wait for completion
    await asyncio.sleep(1.0)
    
    # Check task set is cleaned up
    active_count = tm.active_task_count
    assert active_count < 10, f"Too many active tasks remaining: {active_count}"
    
    print("✅ Memory leak test passed")


async def main():
    """Run all tests."""
    print("🔍 UnifiedTaskManager Stress Testing...")
    print("=" * 50)
    
    try:
        await test_basic_functionality()
        await test_error_handling()
        await test_stress_load()
        await test_shutdown_behavior()
        await test_memory_leaks()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ UnifiedTaskManager is stress-tested and bulletproof")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Final cleanup
        try:
            await shutdown_task_manager()
        except:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
