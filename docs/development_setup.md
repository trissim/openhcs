# Development setup

OpenHCS uses eight first-party packages as both published dependencies and Git
submodules. The current `setup.py` does **not** replace the dependencies from
`pyproject.toml` with local paths automatically. Install the submodules
explicitly when working on the integrated source tree.

## Requirements

- Python 3.11–3.13
- Git with submodule support
- A virtual environment is strongly recommended

## Clone and initialize

```bash
git clone --recurse-submodules https://github.com/OpenHCSDev/OpenHCS.git
cd OpenHCS

# For an existing clone:
git submodule update --init --recursive
```

## Install the integrated development tree

Create and activate a virtual environment, then install each first-party package
in editable mode before OpenHCS. This is the same dependency-source model used by
the integration workflow.

```bash
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
```

Add `dev-gui`, `cellprofiler-compat`, `viz`, or `gpu` only when the work needs
those dependencies:

```bash
python -m pip install -e ".[dev,dev-gui,gui,cellprofiler-compat]"
```

GPU dependencies require a compatible CUDA 12 environment. CPU-only tests do
not require the `gpu` extra.

## Test published dependencies instead

To test OpenHCS against the dependency versions declared in `pyproject.toml`, do
not install the submodules into the environment:

```bash
python -m venv .venv-pypi
source .venv-pypi/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gui]"
```

`OPENHCS_DEV_MODE` is not a supported dependency-selection switch. Although
helper functions with that name remain in `setup.py`, `setup()` currently uses
the dependencies declared by `pyproject.toml`.

## Run tests

```bash
OPENHCS_CPU_ONLY=1 python -m pytest tests/unit -q
```

Use the integration workflow and `docs/source/guides/testing_guide.rst` for the
backend, microscope, execution-mode, and viewer matrices.

## Build documentation

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

## Update submodules deliberately

`git submodule update --remote` changes the recorded commits for every selected
submodule. Review those changes and package compatibility together; do not use
it as a routine installation step.

## Packaging

`python -m build` uses the published first-party dependency requirements from
`pyproject.toml`. Before releasing, test the wheel in a clean environment rather
than relying on editable submodule installs.
