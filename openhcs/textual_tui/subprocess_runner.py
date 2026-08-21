#!/usr/bin/env python3
"""
OpenHCS Subprocess Runner

DEPRECATED: This subprocess runner is deprecated in favor of the ZMQ execution pattern.
For new code, use ZMQExecutionClient from openhcs.runtime.zmq_execution_client.

The ZMQ execution pattern provides:
- Bidirectional communication (vs fire-and-forget)
- Real-time progress streaming
- Graceful cancellation
- Process reuse
- Location-transparent execution (local or remote)

To use ZMQ execution, set environment variable:
    export OPENHCS_USE_ZMQ_EXECUTION=true

This file is maintained for backward compatibility only.

Standalone script that runs OpenHCS plate processing in a clean subprocess environment.
This mimics the integration test pattern from test_main.py but runs independently.

Usage:
    python subprocess_runner.py <data_file.pkl> <log_file_base> [unique_id]
"""

import sys
import dill as pickle
import logging
import traceback
import atexit
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

# Enable subprocess mode - this single variable controls all subprocess behavior
os.environ['OPENHCS_SUBPROCESS_MODE'] = '1'

# Prevent GPU library imports in subprocess runner - workers will handle GPU initialization
os.environ['OPENHCS_SUBPROCESS_NO_GPU'] = '1'
os.environ['POLYSTORE_SUBPROCESS_NO_GPU'] = '1'

# Subprocess runner doesn't need function registry at all
# Workers will initialize their own function registries when needed
logger.info("Subprocess runner: No function registry needed - workers will handle initialization")

def setup_subprocess_logging(log_file_path: str):
    """Set up dedicated logging for the subprocess - all logs go to the specified file."""

    # Configure root logger to capture ALL logs from subprocess and OpenHCS modules
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear any existing handlers

    # Create file handler for subprocess logs
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # Ensure all OpenHCS module logs are captured
    logging.getLogger("openhcs").setLevel(logging.INFO)

    # Prevent console output - everything goes to file
    logging.basicConfig = lambda *args, **kwargs: None

    # Get subprocess logger
    logger = logging.getLogger("openhcs.subprocess")
    logger.info("SUBPROCESS: Logging configured")

    return logger

# Status and result files removed - log file is single source of truth

def run_single_plate(
    plate_path: str,
    execution_bundle,
    logger,
    log_file_base: str = None,
):
    """
    Run a single plate using pre-compiled contexts from UI.

    This creates the orchestrator and executes the pre-compiled plate.
    """
    import psutil
    import os

    def log_thread_count(step_name):
        thread_count = psutil.Process(os.getpid()).num_threads()
        return thread_count

    # NUCLEAR ERROR DETECTION: Wrap EVERYTHING in try/except
    def force_error_detection(func_name, func, *args, **kwargs):
        """Wrapper that forces any error to be visible and logged."""
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error_msg = f"Error in {func_name}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(f"Failed: {func_name} failed: {e}") from e
        except BaseException as e:
            error_msg = f"Critical error in {func_name}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(f"Critical error: {func_name} failed: {e}") from e

    # DEATH DETECTION: Progress markers to find where process dies
    def death_marker(location, details=""):
        """Mark progress to detect where process dies (log file only)."""
        pass

    try:
        death_marker("FUNCTION_START", f"plate_path={plate_path}")
        log_thread_count("function start")

        logger.info(f"Starting plate {plate_path}")

        log_thread_count("after status write")
        
        log_thread_count("before orchestrator import")

        # NUCLEAR WRAP: Orchestrator import
        def import_orchestrator():
            from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
            return PipelineOrchestrator
        PipelineOrchestrator = force_error_detection("import_orchestrator", import_orchestrator)

        log_thread_count("after orchestrator import")

        # NUCLEAR WRAP: Storage registry import (lazy initialization for subprocess mode)
        def import_storage_registry():
            from polystore.base import storage_registry, ensure_storage_registry
            ensure_storage_registry()  # Ensure registry is created in subprocess mode
            return storage_registry
        storage_registry = force_error_detection("import_storage_registry", import_storage_registry)

        log_thread_count("after storage registry import")

        # Skip function registry import in subprocess runner - workers will initialize their own
        logger.info("SUBPROCESS: Function registry initialization deferred to workers")



        log_thread_count("before orchestrator creation")

        # NUCLEAR WRAP: Orchestrator creation
        orchestrator = force_error_detection("PipelineOrchestrator_creation", PipelineOrchestrator,
            plate_path=plate_path,
            storage_registry=storage_registry  # Use default registry
        )
        log_thread_count("after orchestrator creation")

        # NUCLEAR WRAP: Orchestrator initialization
        force_error_detection("orchestrator_initialize", orchestrator.initialize)
        log_thread_count("after orchestrator initialization")

        pipeline_definition = list(execution_bundle.pipeline_definition)
        compiled_contexts = dict(execution_bundle.runtime_contexts)

        # Step 3: Use wells from compiled bundle contexts (not rediscovery)
        # The UI already compiled contexts for the specific wells in this plate
        wells = list(compiled_contexts.keys())

        # AGGRESSIVE VALIDATION: Check wells from compiled contexts
        if not wells:
            error_msg = f"No wells found in compiled contexts for plate {plate_path}!"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        if not isinstance(wells, list):
            error_msg = f"Wells is not a list: {type(wells)} = {wells}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Create a custom progress callback to see exactly where it hangs
        def progress_callback(well_id, step_name, status):
            logger.info(f"🔥 SUBPROCESS: PROGRESS - Well {well_id}, Step '{step_name}', Status: {status}")

        # Add monitoring without timeout
        import threading

        # Start monitoring thread
        monitoring_active = threading.Event()
        monitoring_active.set()

        def monitor_execution():
            count = 0
            while monitoring_active.is_set():
                count += 1
                logger.info(f"🔥 SUBPROCESS: MONITOR #{count} - Still executing...")

                # Skip GPU memory monitoring in subprocess runner
                logger.info("🔥 SUBPROCESS: GPU memory monitoring deferred to workers")

                # Check if we can get thread info
                try:
                    import threading
                    active_threads = threading.active_count()
                    logger.info(f"🔥 SUBPROCESS: Active threads: {active_threads}")
                except:
                    pass

                # Log progress every 30 seconds with system info
                if count % 6 == 0:
                    logger.info(f"🔥 SUBPROCESS: PROGRESS - Been running for {count*5} seconds, still executing...")
                    print(f"🔥 SUBPROCESS STDOUT: PROGRESS - {count*5} seconds elapsed")

                    # Add system monitoring to catch resource issues
                    try:
                        import psutil
                        import os

                        # Memory info
                        memory = psutil.virtual_memory()
                        swap = psutil.swap_memory()
                        process = psutil.Process(os.getpid())

                        logger.info(f"🔥 SUBPROCESS: SYSTEM - RAM: {memory.percent:.1f}% used, {memory.available/1024**3:.1f}GB free")
                        logger.info(f"🔥 SUBPROCESS: SYSTEM - Swap: {swap.percent:.1f}% used")
                        logger.info(f"🔥 SUBPROCESS: SYSTEM - Process RAM: {process.memory_info().rss/1024**3:.1f}GB")
                        logger.info(f"🔥 SUBPROCESS: SYSTEM - Process threads: {process.num_threads()}")

                        print(f"🔥 SUBPROCESS STDOUT: RAM {memory.percent:.1f}%, Process {process.memory_info().rss/1024**3:.1f}GB, Threads {process.num_threads()}")

                        # Check for memory pressure
                        if memory.percent > 90:
                            logger.error(f"🔥 SUBPROCESS: WARNING - High memory usage: {memory.percent:.1f}%")
                            print(f"🔥 SUBPROCESS STDOUT: HIGH MEMORY WARNING: {memory.percent:.1f}%")

                        if process.memory_info().rss > 16 * 1024**3:  # 16GB
                            logger.error(f"🔥 SUBPROCESS: WARNING - Process using {process.memory_info().rss/1024**3:.1f}GB")
                            print(f"🔥 SUBPROCESS STDOUT: HIGH PROCESS MEMORY: {process.memory_info().rss/1024**3:.1f}GB")

                    except Exception as e:
                        logger.debug(f"Could not get system info: {e}")

                threading.Event().wait(5)  # Wait 5 seconds (more frequent)

        monitor_thread = threading.Thread(target=monitor_execution, daemon=True)
        monitor_thread.start()

        try:
            logger.info("🔥 SUBPROCESS: About to call execute_compiled_plate...")
            logger.info(f"🔥 SUBPROCESS: Pipeline has {len(pipeline_definition)} steps")
            logger.info(f"🔥 SUBPROCESS: Compiled contexts for {len(compiled_contexts)} wells")
            logger.info("🔥 SUBPROCESS: Calling execute_compiled_plate NOW...")

            log_thread_count("before execute_compiled_plate")

            # PRE-EXECUTION STATE VALIDATION
            logger.info("🔥 SUBPROCESS: PRE-EXECUTION VALIDATION...")
            print("🔥 SUBPROCESS STDOUT: PRE-EXECUTION VALIDATION...")

            if pipeline_definition is None:
                error_msg = "🔥 CRITICAL: pipeline_definition is None!"
                logger.error(error_msg)
                print(f"🔥 SUBPROCESS STDOUT CRITICAL: {error_msg}")
                raise RuntimeError(error_msg)

            if compiled_contexts is None:
                error_msg = "🔥 CRITICAL: compiled_contexts is None!"
                logger.error(error_msg)
                print(f"🔥 SUBPROCESS STDOUT CRITICAL: {error_msg}")
                raise RuntimeError(error_msg)

            logger.info(f"🔥 SUBPROCESS: PRE-EXECUTION OK - pipeline:{len(pipeline_definition)}, contexts:{len(compiled_contexts)}")
            print(f"🔥 SUBPROCESS STDOUT: PRE-EXECUTION OK - pipeline:{len(pipeline_definition)}, contexts:{len(compiled_contexts)}")

            # NUCLEAR EXECUTION WRAPPER: Force any error to surface
            death_marker("BEFORE_EXECUTION_CALL", f"pipeline_steps={len(pipeline_definition)}, contexts={len(compiled_contexts)}")
            logger.info("🔥 SUBPROCESS: CALLING NUCLEAR EXECUTION WRAPPER...")
            print("🔥 SUBPROCESS STDOUT: CALLING NUCLEAR EXECUTION WRAPPER...")

            death_marker("ENTERING_FORCE_ERROR_DETECTION")
            progress_context = {
                "execution_id": f"subprocess::{os.getpid()}::{plate_path}",
                "plate_id": plate_path,
                "axis_id": "",
            }
            results = force_error_detection("execute_compiled_plate", orchestrator.execute_compiled_plate,
                execution_bundle=execution_bundle,
                visualizer=None,    # Let orchestrator auto-create visualizers based on compiled contexts
                log_file_base=log_file_base,  # Pass log base for worker process logging
                progress_queue=execution_bundle.runtime_environment.worker_start.multiprocessing_context().Queue(),
                progress_context=progress_context,
            )
            death_marker("AFTER_FORCE_ERROR_DETECTION", f"results_type={type(results)}")

            logger.info("🔥 SUBPROCESS: NUCLEAR EXECUTION WRAPPER RETURNED!")
            print("🔥 SUBPROCESS STDOUT: NUCLEAR EXECUTION WRAPPER RETURNED!")
            death_marker("EXECUTION_WRAPPER_RETURNED")

            log_thread_count("after execute_compiled_plate")

            logger.info("🔥 SUBPROCESS: execute_compiled_plate RETURNED successfully!")
            logger.info(f"🔥 SUBPROCESS: Results: {type(results)}, length: {len(results) if results else 'None'}")

            # FORCE ERROR DETECTION: Check for None results immediately
            if results is None:
                error_msg = "🔥 CRITICAL: execute_compiled_plate returned None - this should never happen!"
                logger.error(error_msg)
                print(f"🔥 SUBPROCESS STDOUT CRITICAL: {error_msg}")
                raise RuntimeError(error_msg)

        except Exception as execution_error:
            # FORCE ERROR PROPAGATION: Re-raise with enhanced context
            error_msg = f"🔥 EXECUTION ERROR in execute_compiled_plate: {execution_error}"
            logger.error(error_msg, exc_info=True)
            print(f"🔥 SUBPROCESS STDOUT EXECUTION ERROR: {error_msg}")
            print(f"🔥 SUBPROCESS STDOUT EXECUTION TRACEBACK: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from execution_error
        finally:
            monitoring_active.clear()  # Stop monitoring

        logger.info("🔥 SUBPROCESS: Execution completed!")

        # AGGRESSIVE RESULT VALIDATION: Force errors to surface
        logger.info("🔥 SUBPROCESS: Starting aggressive result validation...")

        # Check 1: Results exist
        if not results:
            error_msg = "🔥 EXECUTION FAILED: No results returned from execute_compiled_plate!"
            logger.error(error_msg)
            print(f"🔥 SUBPROCESS STDOUT ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        # Check 2: Results is a dictionary
        if not isinstance(results, dict):
            error_msg = f"🔥 EXECUTION FAILED: Results is not a dict, got {type(results)}: {results}"
            logger.error(error_msg)
            print(f"🔥 SUBPROCESS STDOUT ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        # Check 3: Expected number of results
        if len(results) != len(wells):
            error_msg = f"🔥 EXECUTION FAILED: Expected {len(wells)} results, got {len(results)}. Wells: {wells}, Result keys: {list(results.keys())}"
            logger.error(error_msg)
            print(f"🔥 SUBPROCESS STDOUT ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        # Check 4: All wells have results
        missing_wells = set(wells) - set(results.keys())
        if missing_wells:
            error_msg = f"🔥 EXECUTION FAILED: Missing results for wells: {missing_wells}"
            logger.error(error_msg)
            print(f"🔥 SUBPROCESS STDOUT ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        # Check 5: All results have proper structure and check for errors
        failed_wells = []
        for well_id, result in results.items():
            logger.info(f"🔥 SUBPROCESS: Validating result for well {well_id}: {result}")

            if not isinstance(result, dict):
                error_msg = f"🔥 EXECUTION FAILED: Result for well {well_id} is not a dict: {type(result)} = {result}"
                logger.error(error_msg)
                print(f"🔥 SUBPROCESS STDOUT ERROR: {error_msg}")
                raise RuntimeError(error_msg)

            if 'status' not in result:
                error_msg = f"🔥 EXECUTION FAILED: Result for well {well_id} missing 'status' field: {result}"
                logger.error(error_msg)
                print(f"🔥 SUBPROCESS STDOUT ERROR: {error_msg}")
                raise RuntimeError(error_msg)

            if result.get('status') != 'success':
                error_msg = result.get('error_message', 'Unknown error')
                details = result.get('details', 'No details')
                full_error = f"🔥 EXECUTION FAILED for well {well_id}: {error_msg} | Details: {details}"
                logger.error(full_error)
                print(f"🔥 SUBPROCESS STDOUT ERROR: {full_error}")
                failed_wells.append((well_id, error_msg, details))

        # Check 6: Raise if any wells failed
        if failed_wells:
            error_summary = f"🔥 EXECUTION FAILED: {len(failed_wells)} wells failed out of {len(wells)}"
            for well_id, error_msg, details in failed_wells:
                error_summary += f"\n  - Well {well_id}: {error_msg}"
            logger.error(error_summary)
            print(f"🔥 SUBPROCESS STDOUT ERROR: {error_summary}")
            raise RuntimeError(error_summary)
        
        logger.info(f"🔥 SUBPROCESS: EXECUTION SUCCESS: {len(results)} wells executed successfully")
        
        # Success logged to log file (single source of truth)
        logger.info(f"🔥 SUBPROCESS: COMPLETED plate {plate_path} with {len(results)} results")
        
        logger.info(f"🔥 SUBPROCESS: Plate {plate_path} completed successfully")
        
    except Exception as e:
        error_msg = f"Execution failed for plate {plate_path}: {e}"
        logger.error(f"🔥 SUBPROCESS: {error_msg}", exc_info=True)
        print(f"🔥 SUBPROCESS STDOUT ERROR: {error_msg}")
        print(f"🔥 SUBPROCESS STDOUT TRACEBACK: {traceback.format_exc()}")
        # Error logged to log file (single source of truth)
    except BaseException as e:
        # Catch EVERYTHING including SystemExit, KeyboardInterrupt, etc.
        error_msg = f"CRITICAL failure for plate {plate_path}: {e}"
        logger.error(f"🔥 SUBPROCESS: {error_msg}", exc_info=True)
        print(f"🔥 SUBPROCESS STDOUT CRITICAL: {error_msg}")
        print(f"🔥 SUBPROCESS STDOUT CRITICAL TRACEBACK: {traceback.format_exc()}")
        # Error logged to log file (single source of truth)

def main():
    """Main entry point for subprocess runner."""
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python subprocess_runner.py <data_file.pkl> <log_file_base> [unique_id]")
        sys.exit(1)

    data_file = sys.argv[1]
    log_file_base = sys.argv[2]
    unique_id = sys.argv[3] if len(sys.argv) == 4 else None

    # Build log file name from provided base and unique ID
    if unique_id:
        log_file = f"{log_file_base}_{unique_id}.log"
    else:
        log_file = f"{log_file_base}.log"

    # PROCESS GROUP CLEANUP: Create new process group to manage all child processes
    try:
        import os
        import signal
        
        # Create new process group with this process as leader
        os.setpgrp()  # Create new process group
        process_group_id = os.getpgrp()
        
        print(f"🔥 SUBPROCESS: Created process group {process_group_id}")
        
        # Track all child processes for cleanup
        child_processes = set()
        
        def kill_all_children():
            """Kill all child processes and the entire process group."""
            try:
                print(f"🔥 SUBPROCESS: Killing process group {process_group_id}")
                # Kill entire process group (negative PID kills process group)
                os.killpg(process_group_id, signal.SIGTERM)
                
                # Give processes time to exit gracefully
                import time
                time.sleep(2)
                
                # Force kill if still alive
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Already dead
                    
                print(f"🔥 SUBPROCESS: Process group {process_group_id} terminated")
            except Exception as e:
                print(f"🔥 SUBPROCESS: Error killing process group: {e}")
        
        # Register cleanup function
        atexit.register(kill_all_children)
        
    except Exception as e:
        print(f"🔥 SUBPROCESS: Warning - Could not set up process group cleanup: {e}")

    # Set up logging first
    logger = setup_subprocess_logging(log_file)
    logger.info("🔥 SUBPROCESS: Starting OpenHCS subprocess runner")
    logger.info(f"🔥 SUBPROCESS: Args - data: {data_file}, log: {log_file}")
    logger.info(f"🔥 SUBPROCESS: Log file: {log_file}")

    # DEATH DETECTION: Set up heartbeat monitoring
    import threading
    import time

    def heartbeat_monitor():
        """Monitor that writes heartbeats to detect where process dies."""
        heartbeat_count = 0
        while True:
            try:
                heartbeat_count += 1
                heartbeat_msg = f"🔥 SUBPROCESS HEARTBEAT #{heartbeat_count}: Process alive at {time.time()}"
                logger.info(heartbeat_msg)
                print(heartbeat_msg)

                # Heartbeat logged to log file (single source of truth)
                # No separate heartbeat file needed

                time.sleep(2)  # Heartbeat every 2 seconds
            except Exception as monitor_error:
                logger.error(f"🔥 SUBPROCESS: Heartbeat monitor error: {monitor_error}")
                print(f"🔥 SUBPROCESS STDOUT: Heartbeat monitor error: {monitor_error}")
                break

    # Start heartbeat monitor in daemon thread
    heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    heartbeat_thread.start()
    logger.info("🔥 SUBPROCESS: Heartbeat monitor started")

    # NUCLEAR CRASH DETECTION - catch EVERYTHING
    def crash_handler(signum, frame):
        crash_msg = f"🔥 SUBPROCESS: CRASH DETECTED - Signal {signum} received!"
        logger.error(crash_msg)
        print(f"🔥 SUBPROCESS STDOUT CRASH: {crash_msg}")

        # Crash info logged to log file (single source of truth)

        # Dump stack trace
        try:
            import traceback
            logger.error("🔥 SUBPROCESS: CRASH - Dumping all thread stacks...")
            for thread_id, frame in sys._current_frames().items():
                logger.error(f"🔥 SUBPROCESS: CRASH Thread {thread_id}:")
                traceback.print_stack(frame)
        except:
            pass

        # Force exit
        os._exit(1)

    # Set up signal handlers for all possible crashes
    signal.signal(signal.SIGSEGV, crash_handler)  # Segmentation fault
    signal.signal(signal.SIGABRT, crash_handler)  # Abort
    signal.signal(signal.SIGFPE, crash_handler)   # Floating point exception
    signal.signal(signal.SIGILL, crash_handler)   # Illegal instruction
    signal.signal(signal.SIGTERM, crash_handler)  # Termination
    signal.signal(signal.SIGINT, crash_handler)   # Interrupt (Ctrl+C)

    # Set up atexit handler to catch silent deaths
    def exit_handler():
        logger.error("🔥 SUBPROCESS: ATEXIT - Process is exiting!")
        print("🔥 SUBPROCESS STDOUT: ATEXIT - Process is exiting!")
        # Exit info logged to log file (single source of truth)

    atexit.register(exit_handler)

    # Set up debug signal handler
    def debug_handler(signum, frame):
        logger.error("🔥 SUBPROCESS: SIGUSR1 received - dumping stack trace")
        import traceback
        import threading

        # Dump all thread stacks
        for thread_id, frame in sys._current_frames().items():
            logger.error(f"🔥 SUBPROCESS: Thread {thread_id} stack:")
            traceback.print_stack(frame)

        # Log thread info
        for thread in threading.enumerate():
            logger.error(f"🔥 SUBPROCESS: Thread: {thread.name}, alive: {thread.is_alive()}")

    signal.signal(signal.SIGUSR1, debug_handler)
    logger.info("🔥 SUBPROCESS: NUCLEAR CRASH DETECTION ENABLED - All signals monitored")
    
    try:
        # Load pickled data
        logger.info(f"🔥 SUBPROCESS: Loading data from {data_file}")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        plate_paths = data['plate_paths']
        pipeline_data = data['pipeline_data']  # Dict[plate_path, List[FunctionStep]]
        global_config = data['global_config']

        logger.info(f"🔥 SUBPROCESS: Loaded data for {len(plate_paths)} plates")
        logger.info(f"🔥 SUBPROCESS: Plates: {plate_paths}")

        # Process each plate (like test_main.py but for multiple plates)
        for plate_path in plate_paths:
            plate_data = pipeline_data[plate_path]
            execution_bundle = plate_data['execution_bundle']
            logger.info(f"🔥 SUBPROCESS: Processing plate {plate_path} with {len(execution_bundle.pipeline_definition)} steps")

            run_single_plate(
                plate_path=plate_path,
                execution_bundle=execution_bundle,
                logger=logger,
                log_file_base=log_file_base,
            )

        logger.info("🔥 SUBPROCESS: All plates completed successfully")

        # Run global multi-plate consolidation if enabled and multiple plates were processed
        if len(plate_paths) > 1 and global_config.analysis_consolidation_config.enabled:
            try:
                logger.info("🔥 SUBPROCESS: Starting global multi-plate consolidation")

                from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_multi_plate_summaries
                from pathlib import Path

                # Collect individual MetaXpress summaries from each plate
                summary_paths = []
                plate_names = []

                for plate_path in plate_paths:
                    # Build results directory path using same logic as orchestrator
                    plate_path_obj = Path(plate_path)
                    path_config = global_config.path_planning_config

                    # Determine output plate root
                    if path_config.global_output_folder:
                        base = Path(path_config.global_output_folder)
                    else:
                        base = plate_path_obj.parent

                    output_plate_root = base / f"{plate_path_obj.name}{path_config.output_dir_suffix}"

                    # Get results directory
                    materialization_path = global_config.materialization_results_path
                    if Path(materialization_path).is_absolute():
                        results_dir = Path(materialization_path)
                    else:
                        results_dir = output_plate_root / materialization_path

                    # Look for MetaXpress summary in results directory
                    summary_filename = global_config.analysis_consolidation_config.output_filename
                    summary_path = results_dir / summary_filename

                    if summary_path.exists():
                        summary_paths.append(str(summary_path))
                        plate_names.append(output_plate_root.name)
                        logger.info(f"Found summary for plate {output_plate_root.name}: {summary_path}")
                    else:
                        logger.warning(f"No summary found for plate {plate_path} at {summary_path}")

                if len(summary_paths) > 1:
                    # Determine global output location
                    if path_config.global_output_folder:
                        global_output_dir = Path(path_config.global_output_folder)
                    else:
                        # Use parent of first plate's output directory
                        global_output_dir = Path(summary_paths[0]).parent.parent.parent

                    global_summary_filename = global_config.analysis_consolidation_config.global_summary_filename
                    global_summary_path = global_output_dir / global_summary_filename

                    # Consolidate all summaries
                    logger.info(f"Consolidating {len(summary_paths)} summaries to {global_summary_path}")
                    consolidate_multi_plate_summaries(
                        summary_paths=summary_paths,
                        output_path=str(global_summary_path),
                        plate_names=plate_names
                    )
                    logger.info("✅ GLOBAL CONSOLIDATION: Completed successfully")
                else:
                    logger.info(f"⏭️ GLOBAL CONSOLIDATION: Only {len(summary_paths)} summary found, skipping")

            except Exception as e:
                logger.error(f"❌ GLOBAL CONSOLIDATION: Failed: {e}", exc_info=True)
        
    except Exception as e:
        logger.error(f"🔥 SUBPROCESS: Fatal error: {e}", exc_info=True)
        print(f"🔥 SUBPROCESS STDOUT FATAL: {e}")
        print(f"🔥 SUBPROCESS STDOUT FATAL TRACEBACK: {traceback.format_exc()}")
        # Error logged to log file (single source of truth)
        logger.error(f"🔥 SUBPROCESS: Fatal error for all plates: {e}")
        sys.exit(1)
    except BaseException as e:
        # Catch EVERYTHING including SystemExit, KeyboardInterrupt, etc.
        logger.error(f"🔥 SUBPROCESS: CRITICAL SYSTEM ERROR: {e}", exc_info=True)
        print(f"🔥 SUBPROCESS STDOUT CRITICAL SYSTEM: {e}")
        print(f"🔥 SUBPROCESS STDOUT CRITICAL SYSTEM TRACEBACK: {traceback.format_exc()}")
        # Critical error logged to log file (single source of truth)
        logger.error(f"🔥 SUBPROCESS: Critical system error for all plates: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
