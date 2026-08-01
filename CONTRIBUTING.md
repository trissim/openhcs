# Contributing to OpenHCS

OpenHCS welcomes reproducible bug reports, microscope fixtures, processing
functions, documentation, and focused code changes.

## Report an issue

Search the issue tracker first, then include the OpenHCS version, installation
route, operating system, Python version, minimal steps, expected behavior, and
relevant logs. For microscopy failures, describe the instrument, axes, naming
pattern, and expected output without uploading sensitive image data.

Security vulnerabilities should be reported privately as described in
[SECURITY.md](SECURITY.md).

## Development checkout

OpenHCS uses source submodules for its independently published libraries:

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/OpenHCSDev/OpenHCS.git
cd OpenHCS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "external/python-introspect"
python -m pip install -e "external/metaclass-registry"
python -m pip install -e "external/ObjectState"
python -m pip install -e "external/arraybridge"
python -m pip install -e "external/pycodify"
python -m pip install -e "external/PolyStore"
python -m pip install -e "external/pyqt-reactive"
python -m pip install -e "external/zmqruntime"
python -m pip install -e ".[dev,dev-gui,gui,mcp]"
```

See `docs/development_setup.md` for optional GPU, viewer, Bio-Formats, OMERO,
and CellProfiler compatibility dependencies.

## Architecture expectations

Read `docs/source/architecture/quick_start.rst`, then
`system_overview.rst`, `nominal_ownership.rst`, and
`abstraction_lattices.rst` before a structural change.

- Change the declaration or typed owner of a behavior.
- Derive schemas, UI projections, registries, and runtime plans from that owner.
- Do not add mirrored metadata tables, parallel kind registries, sidecars, or
  compatibility shims when the owning contract can express the rule.
- Keep compilation deterministic. Workers consume frozen compiled plans rather
  than reinterpreting mutable authoring objects.
- Put reusable behavior in the library that owns it; update the OpenHCS
  submodule pointer only after that library change is independently tested.

## Tests and pull requests

Run the narrowest relevant tests while developing, then the CPU unit gate:

```bash
OPENHCS_CPU_ONLY=1 python -m pytest tests/unit
python -m ruff check openhcs tests
```

GUI tests can use `QT_QPA_PLATFORM=offscreen`. Add a regression that fails for
the reported behavior and exercises its public or nominal boundary. Keep pull
requests focused, explain the evidence boundary, and note any platform or GPU
journey you could not run locally. All contributions must follow
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
