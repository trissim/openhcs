Stitching Functions
==================

.. module:: openhcs.processing.backends

OpenHCS provides **GPU-accelerated stitching functions** for high-performance microscopy image assembly. The stitching workflow consists of two main phases: **position generation** and **image assembly**.

Stitching Workflow
------------------

OpenHCS implements a **two-phase stitching approach** optimized for large-scale microscopy datasets:

**Phase 1: Position Generation**
    Calculate optimal tile positions using advanced algorithms

    - **Ashlar Algorithm**: GPU-accelerated edge alignment with phase correlation
    - **MIST Algorithm**: Feature-based registration with robust outlier handling
    - **Special I/O**: Positions are automatically passed between steps

**Phase 2: Image Assembly**
    Assemble final images using calculated positions with GPU acceleration

    - **Subpixel Positioning**: Precise tile placement with interpolation
    - **Advanced Blending**: Multiple blending methods (fixed, dynamic, none)
    - **GPU Optimization**: CuPy-accelerated assembly for large datasets

Position Generation Functions
-----------------------------

**Ashlar GPU Algorithm** (Gold Standard Usage)

.. code-block:: python

    from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents

    # GPU-accelerated position calculation from gold standard script
    step = FunctionStep(
        func=[(ashlar_compute_tile_positions_gpu, {
            'overlap_ratio': 0.1,
            'max_shift': 15.0,
            'stitch_alpha': 0.2
        })],
        name="positions",
        variable_components=[VariableComponents.CHANNEL],
        force_disk_output=True  # Positions saved for assembly step
    )

**Ashlar CPU Algorithm**

.. code-block:: python

    from openhcs.processing.backends.pos_gen.ashlar_main_cpu import ashlar_compute_tile_positions_cpu

    # CPU fallback for compatibility
    step = FunctionStep(
        func=[(ashlar_compute_tile_positions_cpu, {
            'overlap_ratio': 0.1,
            'max_shift': 15.0,
            'stitch_alpha': 0.2
        })],
        name="positions_cpu",
        variable_components=[VariableComponents.CHANNEL]
    )

**MIST Algorithm**

.. code-block:: python

    from openhcs.processing.backends.pos_gen.mist_processor_cupy import mist_compute_tile_positions

    # Alternative stitching algorithm
    step = FunctionStep(
        func=mist_compute_tile_positions,
        name="mist_positions",
        variable_components=[VariableComponents.CHANNEL]
    )

Image Assembly Functions
------------------------

**GPU Assembly** (Gold Standard Usage)

.. code-block:: python

    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

    # GPU-accelerated assembly from gold standard script
    step = FunctionStep(
        func=[(assemble_stack_cupy, {
            'blend_method': "fixed",
            'fixed_margin_ratio': 0.1
        })],
        name="assemble",
        variable_components=[VariableComponents.CHANNEL],
        force_disk_output=True  # Final stitched images
    )

**CPU Assembly**

.. code-block:: python

    from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu

    # CPU fallback for compatibility
    step = FunctionStep(
        func=[(assemble_stack_cpu, {
            'blend_method': "fixed",
            'fixed_margin_ratio': 0.1
        })],
        name="assemble_cpu",
        variable_components=[VariableComponents.CHANNEL]
    )

Special I/O Integration
-----------------------

Stitching functions use **special I/O** for seamless data flow between position generation and assembly:

.. code-block:: python

    from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs

    # Position generation functions output positions
    @special_outputs("positions")
    def ashlar_compute_tile_positions_gpu(image_stack, grid_dimensions, ...):
        # Calculate positions and return them as special output
        return processed_images, positions

    # Assembly functions automatically receive positions
    @special_inputs("positions")
    def assemble_stack_cupy(image_tiles, positions, ...):
        # Positions automatically provided from previous step
        return assembled_image

Complete Stitching Pipeline
---------------------------

Gold standard example from production script:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.constants.constants import VariableComponents

    # Import stitching functions
    from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
    from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

    # Complete stitching workflow
    stitching_pipeline = [
        # Phase 1: Calculate tile positions
        FunctionStep(
            func=[(ashlar_compute_tile_positions_gpu, {
                'overlap_ratio': 0.1,
                'max_shift': 15.0,
                'stitch_alpha': 0.2
            })],
            name="positions",
            variable_components=[VariableComponents.CHANNEL],
            force_disk_output=True
        ),

        # Phase 2: Assemble final images
        FunctionStep(
            func=[(assemble_stack_cupy, {
                'blend_method': "fixed",
                'fixed_margin_ratio': 0.1
            })],
            name="assemble",
            variable_components=[VariableComponents.CHANNEL],
            force_disk_output=True
        )
    ]

Algorithm Comparison
--------------------

**Ashlar Algorithm**:
- **Best for**: High-precision stitching with subpixel accuracy
- **Performance**: GPU-accelerated, handles large datasets
- **Method**: Edge-based alignment with phase correlation
- **Use case**: Production workflows requiring maximum quality

**MIST Algorithm**:
- **Best for**: Feature-rich images with distinct landmarks
- **Performance**: Robust to outliers and noise
- **Method**: Feature-based registration
- **Use case**: Challenging datasets with poor overlap

Blending Methods
----------------

**Fixed Blending** (``blend_method="fixed"``):
- Uses fixed margin ratio for consistent blending
- Best for uniform illumination
- Parameter: ``fixed_margin_ratio`` (default: 0.1)

**Dynamic Blending** (``blend_method="dynamic"``):
- Adapts blending based on overlap regions
- Best for variable illumination
- Parameter: ``overlap_blend_fraction`` (default: 1.0)

**No Blending** (``blend_method="none"``):
- Simple tile placement without blending
- Fastest option for preview or testing

Performance Optimization
------------------------

**GPU Acceleration**:
- Use ``ashlar_compute_tile_positions_gpu`` and ``assemble_stack_cupy`` for maximum performance
- Automatic memory management prevents CUDA out-of-memory errors

**Memory Management**:
- Large datasets automatically use chunked processing
- VFS backend selection optimizes memory usage

**Parallel Processing**:
- Position generation and assembly can run in parallel across channels
- Use ``variable_components=[VariableComponents.CHANNEL]`` for channel-wise processing

See Also
--------

- :doc:`processing_backends` - Complete processing backends overview
- :doc:`../architecture/special_io_system` - Special I/O system details
- :doc:`function_step` - Using functions in pipelines
- :doc:`../concepts/basic_microscopy` - Stitching workflow concepts
