Installation and first launch
=============================

OpenHCS requires Python 3.11 or newer. The supported interactive application is
the desktop GUI.

Install
-------

Create a Python virtual environment if possible, then install the GUI extra:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install "openhcs[gui]"

Launch the application with:

.. code-block:: bash

   openhcs

``openhcs-gui`` launches the same application. The old terminal interface is
deprecated and is not part of the published package.

Optional viewers
----------------

Napari and Fiji/ImageJ are optional and can be installed separately:

.. code-block:: bash

   python -m pip install "openhcs[gui,napari]"
   python -m pip install "openhcs[gui,fiji]"
   python -m pip install "openhcs[gui,viz]"  # both viewers

GPU support
-----------

Only install the GPU extra on a system with a compatible CUDA 12 environment:

.. code-block:: bash

   python -m pip install "openhcs[gui,gpu]"

A CPU-only installation is the normal ``openhcs[gui]`` installation without
the ``gpu`` extra. Do not use ``--no-deps``: OpenHCS requires its declared core
dependencies even when CUDA libraries are absent.

First launch
------------

The main window provides:

1. the Plate Manager for microscopy datasets;
2. the Pipeline Editor for processing steps;
3. compilation and execution controls;
4. progress, logs, and optional viewer integration.

Start with a representative one- or two-well subset. Add the plate, construct
or import a pipeline, compile it, and inspect any reported source or artifact
errors before executing the full screen.

Updating
--------

Update the installed capabilities you use rather than installing every optional
backend:

.. code-block:: bash

   python -m pip install --upgrade "openhcs[gui]"

See :doc:`../getting_started/getting_started` for CellProfiler import and the
current programmatic declaration boundary.
