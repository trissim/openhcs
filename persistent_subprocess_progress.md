# Persistent Subprocess Runner - Progress Summary

## Original Goal
Make the OpenHCS subprocess runner persistent so it doesn't need to reinitialize (GPU registry, function registry, etc.) each time the user presses "Run". Instead of exiting after completion, the subprocess should stay alive and be ready to handle the next run.

## Current State: **FAILED IMPLEMENTATION**

### What We Tried (Chronologically)

#### Attempt 1: Complex JSON Communication System
- **What**: Added `PersistentSubprocessManager` class with JSON stdin/stdout communication
- **Implementation**: 
  - Created `openhcs/textual_tui/services/persistent_subprocess.py`
  - Added JSON protocol: `{"command": "run"}`, `{"status": "ready"}`, etc.
  - Modified subprocess_runner.py to support `--persistent` mode
  - Added background threading for execution
- **Problems**: 
  - Overcomplicated the existing working system
  - JSON parsing issues with log output mixed in
  - Created dual code paths (persistent vs legacy)
  - Communication failures and timeouts

#### Attempt 2: Signal-Based Cancellation
- **What**: Added SIGTERM handling for cancellation without subprocess exit
- **Implementation**:
  - Added cancel signal handlers
  - Process group management for worker cleanup
  - Log-based ready markers
- **Problems**:
  - SIGTERM killed the TUI process instead of just subprocess
  - Process group killing errors (`[Errno 3] No such process`)
  - Still maintained dual code paths

#### Attempt 3: "Minimal" Infinite Loop (CURRENT - BROKEN)
- **What**: Added infinite `while True: sleep(10)` loop after subprocess completion
- **Implementation**: 6 lines added to subprocess_runner.py after successful completion
- **Problems**:
  - **Completely useless**: Subprocess just sits there doing nothing
  - **Doesn't solve the problem**: TUI still creates NEW subprocess for each run
  - **Zombie process**: Wastes resources with no benefit
  - **Not actually persistent**: Can't receive or process new work

## Key Issues Identified

### 1. **Fundamental Misunderstanding**
- I kept trying to maintain the existing subprocess creation pattern while making it "persistent"
- The real issue: TUI creates a NEW subprocess each time via `subprocess.Popen()`
- A truly persistent subprocess needs to be created ONCE and reused

### 2. **Communication Challenge**
- Current system uses one-way communication: TUI → subprocess via pickle files
- Persistent system needs bidirectional communication for new work
- Existing log-based monitoring works well for completion detection

### 3. **Overengineering**
- Added complex JSON protocols when simpler solutions exist
- Created dual code paths instead of replacing the existing system
- Ignored the working log-based communication system

## What Actually Needs to Happen

### The Real Solution
1. **TUI should create subprocess ONCE** (not every run)
2. **Subprocess should wait for new work** (not exit after completion)
3. **Use simple communication** (file-based, stdin, or signals)
4. **Reuse existing log monitoring** (don't reinvent)

### Minimal Implementation Approach
1. **Modify TUI**: Check if subprocess is alive before creating new one
2. **Modify subprocess**: Wait for new pickle files instead of exiting
3. **Use existing cleanup**: Keep current signal handling for stop button
4. **Keep existing logs**: Don't change the working log communication

## Current Code State
- **subprocess_runner.py**: Has useless infinite loop after completion
- **plate_manager.py**: Unchanged (still creates new subprocess each run)
- **persistent_subprocess.py**: Deleted (was overengineered)

## Next Steps (If Continuing)
1. **Remove the infinite loop** from subprocess_runner.py
2. **Implement simple file polling** in subprocess for new work
3. **Modify TUI** to reuse existing subprocess if alive
4. **Test with existing log monitoring** system

## Lessons Learned
- **Don't overcomplicate working systems**
- **Understand the problem before implementing solutions**
- **Infinite loops are not "persistence"**
- **The user's preference for simplicity was correct**

---

**Status**: Implementation failed. Need to start over with simpler approach or abandon feature.
