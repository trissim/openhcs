Install and start OpenHCS
=========================

This how-to guide gets the supported desktop application running. To learn the
workflow through a bounded example after installation, follow
:doc:`../guide_for_biologists/intro_stitching`.

Requirements
------------

- Windows or macOS for the desktop installer; Python is included
- Python 3.11 through 3.13 for a manual installation on Windows, macOS, or Linux
- A CUDA 12 compatible environment only if you choose the optional ``gpu``
  dependencies

Desktop installers
------------------

For Windows or macOS, download the installer from the
`latest OpenHCS release <https://github.com/OpenHCSDev/OpenHCS/releases/latest>`_.
The installer creates an isolated, user-scoped environment and includes the
desktop GUI, local MCP server, CellProfiler compatibility, Bio-Formats, Napari,
and Fiji/ImageJ. It does not require an existing Python installation.

You do not need to remove package indexes configured for another Python
project. The desktop installers and in-app updater ignore workstation pip
configuration files and inherited primary or extra package-index overrides.

Open the downloaded installer and follow the prompts. When installation
finishes, launch **OpenHCS** from the created shortcut. The installer shows its
live output while it works and keeps the complete installation log available
from the finished page. The first Fiji launch materializes and caches the
checksummed Fiji, JDK, and Python bridge runtime, so it takes longer than later
launches.

.. important::

   To upgrade OpenHCS 0.7.23 or earlier, save your work, close OpenHCS, and run
   the latest official installer once. Later in-app updates use a verified
   replacement environment and preserve the previous environment for recovery.

If macOS blocks the official bootstrap because it is unsigned and not notarised,
first try to open **OpenHCS Installer.app**. Then open **System Settings >
Privacy & Security**, scroll to **Security**, click **Open Anyway**,
authenticate, and confirm **Open**. Only approve the disk image downloaded from
the official OpenHCS GitHub release. Apple normally makes **Open Anyway**
available for about an hour after the blocked attempt; see `Apple's current
Gatekeeper override instructions
<https://support.apple.com/guide/mac-help/open-an-app-by-overriding-security-settings-mh40617/mac>`_.

Manual Python installation
--------------------------

Create a virtual environment, then install the same CPU-safe capability set as
the desktop installer:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"
   openhcs

For a smaller environment, install only the capabilities you need:

.. code-block:: bash

   python -m pip install "openhcs[gui]"
   python -m pip install "openhcs[gui,napari]"
   python -m pip install "openhcs[gui,fiji]"
   python -m pip install "openhcs[gui,viz]"       # both viewers
   python -m pip install "openhcs[gui,gpu]"       # CUDA libraries

The base package can be installed for headless imports and services with
``python -m pip install openhcs``. Do not install the ``gpu`` extra on a
CPU-only system. See :doc:`../user_guide/cpu_only_mode` for runtime controls.

Confirm the launch
------------------

The main window should show Plate Manager on the left and Pipeline Editor on
the right. No plate is added automatically during a normal launch. If the
application does not open, run ``openhcs --log-level DEBUG`` from a terminal
and use :doc:`../guide_for_biologists/troubleshooting_FAQ` to locate the log.

Verify the local integration
----------------------------

For a bounded check of the installed MCP, runtime, and Napari path, run:

.. code-block:: bash

   openhcs-mcp-demo --json

The command generates a small local plate, executes the packaged neurite
preset, requires MCP to observe nonzero payloads in the live Napari window, and
reports its output directory in the JSON result.

Update an installation
----------------------

Re-running a desktop installer updates its isolated OpenHCS environment. For a
manual environment, upgrade the capability set you use:

.. code-block:: bash

   python -m pip install --upgrade "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"

Choose your next task
---------------------

- Learn the desktop workflow: :doc:`../guide_for_biologists/intro_stitching`
- Check whether your microscopy data fits: :doc:`../guide_for_biologists/domain_expert_onboarding`
- Import or write pipelines: :doc:`../user_guide/index`
- Configure a local agent client: :doc:`../user_guide/mcp_clients`
- Look up the Python boundary: :doc:`../api/index`
- Understand pipelines and dimensions: :doc:`../concepts/index`
