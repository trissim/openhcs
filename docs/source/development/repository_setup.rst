Repository setup
================

OpenHCS uses eight first-party packages as both published dependencies and Git
submodules. ``setup.py`` does not replace published requirements with local
paths automatically.

Requirements
------------

- Python 3.11 through 3.13
- Git with submodule support
- a virtual environment

Clone and install
-----------------

.. code-block:: console

   git clone --recurse-submodules https://github.com/OpenHCSDev/OpenHCS.git
   cd OpenHCS
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

   python -m pip install -e external/ObjectState
   python -m pip install -e external/python-introspect
   python -m pip install -e external/metaclass-registry
   python -m pip install -e external/arraybridge
   python -m pip install -e external/pycodify
   python -m pip install -e external/PolyStore
   python -m pip install -e external/pyqt-reactive
   python -m pip install -e external/zmqruntime
   python -m pip install -e ".[dev,gui]"

Install optional extras such as ``dev-gui``, ``cellprofiler-compat``, ``viz``,
or ``gpu`` only when the work needs them. GPU dependencies require a
compatible CUDA 12 environment; CPU-only tests do not require the ``gpu``
extra.

For an existing clone, initialize missing dependencies with:

.. code-block:: console

   git submodule update --init --recursive

Published-dependency testing
----------------------------

To test the dependency versions in ``pyproject.toml``, use a clean environment
and install only ``-e ".[dev,gui]"``. Do not install the submodules there.
``OPENHCS_DEV_MODE`` is not a supported dependency-selection switch.

Verification
------------

.. code-block:: console

   OPENHCS_CPU_ONLY=1 python -m pytest tests/unit -q
   python -m pip install -e ".[docs]"
   sphinx-build -W --keep-going -b html docs/source docs/_build/html

GPU tests require the corresponding optional dependencies and a compatible
CUDA environment. ``git submodule update --remote`` changes recorded package
revisions and should be reviewed as a dependency update, not used as an
installation step.

Packaging
---------

``python -m build`` resolves the published first-party dependency requirements
from ``pyproject.toml``. Test the resulting wheel in a clean environment before
release instead of relying on editable submodule installs.
