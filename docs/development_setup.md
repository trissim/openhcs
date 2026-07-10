# Development Setup

OpenHCS uses two deliberately separate dependency modes:

- **Published installs** use the exact dependency versions declared in
  [`pyproject.toml`](../pyproject.toml). On `main`, those versions correspond to
  the submodule commits pinned by `main`.
- **Development installs** use editable installations of those pinned
  submodules from [`external/`](../external/).

The separation matters because feature branches can advance and publish newer
dependency versions that are not compatible with the current `main` release.

## Development Installation

Use a dedicated virtual environment for each OpenHCS worktree. Editable
installations belong to the Python environment, not to a Git branch, so sharing
one environment between `main` and `benchmark-platform` can make one worktree
import code from the other.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
git submodule update --init --recursive
python scripts/dev_install.py --extras dev,gui
```

The installer validates that every root dependency pin matches the version
declared by its checked-out submodule. It then submits OpenHCS and all eight
submodules to pip in one resolver transaction as editable projects and verifies
their installed source paths.

To register only the editable projects without resolving third-party packages:

```bash
python scripts/dev_install.py --extras "" --no-deps
```

A bare `pip install -e .` does **not** automatically install the submodules.
It uses the production dependency metadata. If matching editable dependencies
are already installed in the environment, pip may retain them, which can hide
this distinction; use the development installer for deterministic behavior.

## Main Dependency Set

| Distribution | Main version | Editable source |
|---|---:|---|
| `objectstate` | `1.0.17` | `external/ObjectState` |
| `polystore` | `0.1.9` | `external/PolyStore` |
| `arraybridge` | `0.2.10` | `external/arraybridge` |
| `metaclass-registry` | `0.1.4` | `external/metaclass-registry` |
| `pycodify` | `0.1.2` | `external/pycodify` |
| `pyqt-reactive` | `0.1.21` | `external/pyqt-reactive` |
| `python-introspect` | `0.1.4` | `external/python-introspect` |
| `zmqruntime` | `0.1.8` | `external/zmqruntime` |

Do not run `git submodule update --remote` on `main` as routine setup. That
command advances checkouts to branch tips instead of restoring the commits
pinned by the OpenHCS commit. Use `git submodule update --init --recursive`.

## Production-Metadata Testing

Use a clean environment where the submodules have not been installed editable:

```bash
python -m venv /tmp/openhcs-production-test
source /tmp/openhcs-production-test/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gui]"
```

To build the same package metadata used by the PyPI workflow:

```bash
python -m build
```

Before a release, run:

```bash
python scripts/verify_release_ready.py
```

The release checks reject mismatches between root pins and submodule versions,
and reject exact dependency versions that are unavailable from PyPI.
