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
    Main-flow source strategies for pipeline steps.
    
    This enum replaces the @chain_breaker decorator system with explicit
    input source declaration, providing cleaner and more predictable
    pipeline behavior.
    
    The InputSource enum supports two strategies:
    
    1. **PREVIOUS_STEP** (Default): Standard pipeline chaining where each step
       receives the ordinary main-flow result of the previous step.
       
    2. **PIPELINE_START**: The step receives the pipeline-start main-flow input,
       bypassing previous step results. This replaces the @chain_breaker
       decorator functionality.

    The enum does not select separately named images or other typed values. A
    callable declares those as artifact inputs, and compilation satisfies them
    through step source bindings or prior artifact producers alongside the main
    flow.
    
    Usage Examples:
    
    Standard chaining (default behavior):
    ```python
    step = FunctionStep(
        func=my_processing_function,
        name="process_images",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PREVIOUS_STEP,
        ),
    )
    ```
    
    Chain breaking for position generation:
    ```python
    step = FunctionStep(
        func=ashlar_compute_tile_positions_gpu,
        name="compute_positions",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
    )
    ```
    
    Quality control accessing original data:
    ```python
    step = FunctionStep(
        func=quality_control_function,
        name="qc_check",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
    )
    ```
    """
    
    PREVIOUS_STEP = "previous"
    """
    Standard pipeline chaining strategy.
    
    The step consumes the ordinary main-flow result of the previous step. This
    is the default behavior and maintains normal pipeline chaining.
    
    This strategy:
    - Maintains sequential data flow
    - Enables progressive image processing
    - Is the default for all steps
    """
    
    PIPELINE_START = "start"
    """
    Pipeline start input strategy (replaces @chain_breaker).
    
    The step consumes the pipeline-start main-flow input, bypassing all previous
    step results. This is equivalent to the @chain_breaker behavior but declared
    explicitly.
    
    This strategy:
    - Accesses original input data
    - Bypasses all previous processing steps
    - Is required for position generation and quality control
    
    Common use cases:
    - Position generation functions (MIST, Ashlar)
    - Quality control and validation steps
    - Analysis requiring original image data
    - Debugging and comparison operations
    """
