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
"""

# Import main orchestrator class
from openhcs.core.orchestrator.orchestrator import \
    PipelineOrchestrator

# Define public API
__all__ = [
    # Main orchestrator class
    'PipelineOrchestrator',

    # For backward compatibility
    'DispatchCore',

]
