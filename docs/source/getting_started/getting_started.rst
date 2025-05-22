Getting Started with EZStitcher
===========================

Installation
-----------

.. code-block:: bash

    pip install ezstitcher  # Requires Python 3.11

For `pyenv <https://github.com/pyenv/pyenv>`_ users:

.. code-block:: bash

    pyenv install 3.11
    pyenv local 3.11
    python -m venv .venv
    source .venv/bin/activate
    pip install ezstitcher

Quick Start
----------

The fastest way to get started is with the EZ module:

.. code-block:: python

    from ezstitcher import stitch_plate
    from pathlib import Path

    # Stitch a plate with default settings
    stitch_plate("path/to/microscopy/data")

This will:
- Detect plate format automatically
- Process channels and Z-stacks
- Generate and stitch images
- Save output to "*_stitched" directory

Common Options
------------

.. code-block:: python

    stitch_plate(
        "path/to/plate",
        output_path="path/to/output",    # Custom output location
        normalize=True,                  # Enhance contrast
        flatten_z=True,                  # Convert Z-stacks to 2D
        z_method="max",                  # Z projection method
        well_filter=["A01", "B02"]       # Process specific wells
    )

Next Steps
---------

- Read :doc:`../user_guide/introduction` for an overview of EZStitcher concepts
- See :doc:`../user_guide/basic_usage` for detailed EZ module usage
- Explore :doc:`../user_guide/intermediate_usage` for custom pipelines
- Check :doc:`../concepts/architecture_overview` for technical details
