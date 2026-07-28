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

Download the installer for your operating system from the matching
`GitHub release <https://github.com/OpenHCSDev/openhcs/releases>`_. On Windows,
download and run ``OpenHCS-Windows-Installer.exe``. On macOS, open
``OpenHCS-macOS-Installer.dmg`` and then open ``OpenHCS Installer``. Neither
platform requires ZIP extraction. Re-running the same installer updates the
isolated environment. Installation details and failures are retained in the
OpenHCS user log directory shown by the installer.

The current release workflow is prepared to fail closed unless the Windows
installer is Authenticode-signed and timestamped and the macOS installer is
Developer-ID-signed, notarized, and stapled. No credentialed tag has completed
that workflow yet. Releases through ``0.6.5`` remain unsigned and may still
trigger Windows SmartScreen or macOS Gatekeeper confirmation. This guide will
identify the first signed release after its public assets pass live signature
and notarization verification.

The downloaded installer is only the bootstrap interface; Python and OpenHCS
remain managed by uv and PyPI in the dedicated environment.

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

Smaller manual viewer installs
------------------------------

The desktop installers already include both Napari and Fiji/ImageJ. For a
smaller manual Python environment, select either viewer separately or use
``viz`` for both:

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
