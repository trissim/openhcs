"""
Orchestrator module for OpenHCS.

This module provides the core orchestration functionality for the OpenHCS pipeline,
including the PipelineOrchestrator class and GPU scheduling utilities.

The orchestrator is responsible for:
1. Initializing the processing environment
2. Managing workspace paths and plate directories
3. Initializing microscope handlers
4. Creating processing contexts
5. Executing pipelines
6. Managing GPU resources

Classes and functions in this module follow strict architectural principles:
- Immutability after construction (Clause 66)
- No inferred capabilities (Clause 88)
- Explicit GPU pre-declaration (Clause 293)
- GPU scheduling affinity (Clause 295)
"""

# Import main orchestrator class
from openhcs.core.orchestrator.orchestrator import \
    PipelineOrchestrator
# Import GPU scheduler functions
from openhcs.core.orchestrator.gpu_scheduler import (
    get_gpu_registry_status, initialize_gpu_registry,
    is_gpu_registry_initialized)

# Define public API
__all__ = [
    # Main orchestrator class
    'PipelineOrchestrator',

    # For backward compatibility
    'DispatchCore',

    # GPU scheduler functions
    'initialize_gpu_registry',
    'is_gpu_registry_initialized',
    'get_gpu_registry_status',
]
