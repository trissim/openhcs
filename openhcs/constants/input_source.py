"""
Input Source Strategy Enum for OpenHCS.

This module defines the InputSource enum for explicit input source declaration
in pipeline steps, replacing the @chain_breaker decorator system with a cleaner,
more declarative approach.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 88 — No Inferred Capabilities
- Clause 245 — Declarative Enforcement
"""

from enum import Enum


class InputSource(Enum):
    """
    Enum defining input source strategies for pipeline steps.
    
    This enum replaces the @chain_breaker decorator system with explicit
    input source declaration, providing cleaner and more predictable
    pipeline behavior.
    
    The InputSource enum supports two strategies:
    
    1. **PREVIOUS_STEP** (Default): Standard pipeline chaining where each step
       reads from the output directory of the previous step. This is the normal
       pipeline flow behavior.
       
    2. **PIPELINE_START**: Step reads from the original pipeline input directory,
       effectively "breaking the chain" and accessing the initial input data.
       This replaces the @chain_breaker decorator functionality.
    
    Usage Examples:
    
    Standard chaining (default behavior):
    ```python
    step = FunctionStep(
        func=my_processing_function,
        name="process_images"
        # input_source defaults to InputSource.PREVIOUS_STEP
    )
    ```
    
    Chain breaking for position generation:
    ```python
    step = FunctionStep(
        func=ashlar_compute_tile_positions_gpu,
        name="compute_positions",
        input_source=InputSource.PIPELINE_START  # Access original input images
    )
    ```
    
    Quality control accessing original data:
    ```python
    step = FunctionStep(
        func=quality_control_function,
        name="qc_check",
        input_source=InputSource.PIPELINE_START  # Compare against original
    )
    ```
    """
    
    PREVIOUS_STEP = "previous"
    """
    Standard pipeline chaining strategy.
    
    The step reads input from the output directory of the previous step
    in the pipeline. This is the default behavior and maintains normal
    pipeline flow where each step processes the output of the previous step.
    
    This strategy:
    - Maintains sequential data flow
    - Enables progressive image processing
    - Uses VFS backend from previous step
    - Is the default for all steps
    """
    
    PIPELINE_START = "start"
    """
    Pipeline start input strategy (replaces @chain_breaker).
    
    The step reads input from the original pipeline input directory,
    bypassing all previous step outputs. This is equivalent to the
    @chain_breaker decorator behavior but declared explicitly.
    
    This strategy:
    - Accesses original input data
    - Bypasses all previous processing steps
    - Uses disk backend for VFS consistency
    - Is required for position generation and quality control
    
    Common use cases:
    - Position generation functions (MIST, Ashlar)
    - Quality control and validation steps
    - Analysis requiring original image data
    - Debugging and comparison operations
    """
