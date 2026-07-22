Installation and first launch
=============================

OpenHCS requires Python 3.11 or newer. The supported interactive application is
the desktop GUI.

Desktop installers
------------------

Windows and macOS release pages provide a small ``OpenHCS`` desktop installer
for users who do not already have Python. The installer works entirely in your
user account: it uses uv to install a managed Python, creates a dedicated
OpenHCS environment, installs the CPU-safe GUI package, and creates an OpenHCS
desktop launcher. It does not replace or modify a system Python.

The desktop bundle includes the Qt application, CellProfiler compatibility,
the local MCP server, Napari, and Fiji/Bio-Formats support. GPU libraries are
not included. PyImageJ resolves and caches the Fiji/Bio-Formats Java
distribution on first use rather than embedding a standalone ``Fiji.app``.

Download the installer archive for your operating system from the matching
`GitHub release <https://github.com/OpenHCSDev/openhcs/releases>`_, extract it,
and open the included installer. Re-running the same installer updates the
isolated environment. Installation details and failures are retained in the
OpenHCS user log directory shown by the installer.

The initial bootstrap installers are not code-signed or notarized. Windows
SmartScreen or macOS Gatekeeper may therefore ask you to confirm that you trust
the downloaded release. The release archive is only the bootstrap interface;
Python and OpenHCS remain managed by uv and PyPI in the dedicated environment.

Manual installation
-------------------

Create a Python virtual environment if possible, then install the same CPU-safe
desktop capabilities selected by the native installers:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"

Launch the application with:

.. code-block:: bash

   openhcs

``openhcs-gui`` launches the same application. The old terminal interface is
deprecated and is not part of the published package.

To verify the installed MCP, runtime, and Napari path with a bounded portable
neurite workflow, run:

.. code-block:: bash

   openhcs-mcp-demo --json

The command generates a small local plate, executes the packaged neurite preset,
requires MCP to observe nonzero payloads in the live Napari window, and shuts
down only the runtime and viewer endpoints it allocated. Its output directory is
reported in the JSON result.

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

A CPU-only installation is any installation without the ``gpu`` extra. Do not
use ``--no-deps``: OpenHCS requires its declared core dependencies even when
CUDA libraries are absent.

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

   python -m pip install --upgrade "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"

See :doc:`../getting_started/getting_started` for CellProfiler import and the
current programmatic declaration boundary.
