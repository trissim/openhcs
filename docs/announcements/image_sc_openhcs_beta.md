# image.sc announcement: OpenHCS beta

Post only after every required item in the readiness checklist is verified
against the same public release. This draft intentionally does not freeze a
commit or name a not-yet-published patch release.

## Proposed title

OpenHCS beta: build and run typed HCS pipelines from the GUI, Python, or a local agent

## Ready-to-paste post

Hello image.sc community,

I am preparing an OpenHCS beta announcement and would value feedback from
people who build or maintain high-content microscopy workflows.

OpenHCS is an open-source, Python-native platform for turning a microscopy
plate into a compiled, reproducible analysis. You can author the same typed
pipeline in the desktop GUI, generated Python, or a local MCP-capable agent;
inspect it in Napari or Fiji; and import supported CellProfiler `.cppipe`
workflows into ordinary OpenHCS declarations.

The key design choice is compile before execute. Source matching,
configuration, artifact routing, memory contracts, and worker requirements are
validated across the selected wells, sites, channels, Z planes, and timepoints
before expensive work starts. The GUI, Python API, executor, and MCP tools all
consume the same `PipelineConfig` and ordered `FunctionStep` declarations.

The beta includes:

- ImageXpress and Opera Phenix plate parsing, plus Bio-Formats and OMERO-backed
  source/storage integration;
- visual pipeline authoring with bidirectional GUI to Python editing;
- CPU multiprocessing and opt-in CUDA/OpenCL processing backends;
- typed images, labels, measurements, relationships, tables, ROIs, and files;
- live, process-isolated streaming to Napari and Fiji/ImageJ;
- supported CellProfiler import with explicit compatibility reports; and
- a local stdio MCP server that projects the typed registry, authoring schema,
  compiler, runtime progress, results, and viewer state to compatible clients.

For CellProfiler interoperability, the automated suite runs a source-backed
corpus of 30 example pipelines under documented equivalence policies. Passing
that Official30 corpus is a concrete regression boundary, not a claim that
every historical CellProfiler plugin, version, or setting is supported. An
unfamiliar pipeline should be imported and compiled to expose its exact
compatibility boundary.

### A bounded agent workflow

After configuring explicit read and write roots, a useful first request is:

> Inspect this Opera Phenix plate, infer its axes, and draft a nuclei
> segmentation plus per-cell intensity pipeline. Compile it and explain every
> validation result, but do not execute it.

Review the compiled plan, then authorize a deliberately small run:

> Run A01, site 1; stream the labels to Napari, save the measurements, and
> summarize the cell counts.

The supported MCP server runs locally; OpenHCS does not provide a public hosted
endpoint. Tool inputs and outputs are also subject to the privacy and retention
policy of the agent client and model provider you choose. Treat data roots,
execution, and output writes as capabilities and grant only what the workflow
needs.

### Try it

Windows and macOS installers, verified workflow media, and the PyPI command are
on the landing page:

https://openhcsdev.github.io/openhcs/

For an existing Python 3.11 to 3.13 environment:

```bash
python -m pip install "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"
openhcs
```

The desktop installers are CPU-first. Fiji's Java components are resolved on
first use and GPU packages are opt-in. Current beta installers are unsigned, so
Windows SmartScreen or macOS Gatekeeper may ask you to confirm that you trust
them.

MCP client setup and trust boundaries:

https://openhcs.readthedocs.io/en/latest/user_guide/mcp_clients.html

Documentation and five-minute interface demo:

https://openhcs.readthedocs.io/

https://openhcs.readthedocs.io/en/latest/_static/openhcs.mp4

Source and issue tracker:

https://github.com/OpenHCSDev/OpenHCS

### Feedback I am looking for

I would especially like to hear from people with:

- an ImageXpress or Opera Phenix plate with a nontrivial naming pattern;
- a CellProfiler pipeline they would like to import;
- multi-site, multi-channel, Z-stack, or time-series assays;
- workflows producing measurements, ROIs, relationships, or Analyst outputs;
- a need to move between visual editing, Python, viewers, and bounded agent
  automation.

If you try it, please share the microscope format, image axes, biological goal,
and expected outputs. A synthetic or redacted representative is enough. I am
also happy to help the first few labs migrate one representative pipeline and
use that experience to improve onboarding.

OpenHCS is MIT licensed. Thank you for testing, criticism, and workflow
examples.

## Release readiness checklist

The announcement is a **go** only when every required item is verified against
the same tagged commit and public version. Checks below are deliberately left
open until the announcement release exists.

### 1. Identity and automated acceptance

- [ ] `main`, the annotated tag, PyPI, GitHub Release, plugin bundle, and MCP
      Registry resolve to the same version and commit.
- [ ] The worktree is clean and the package classifier remains Beta.
- [ ] Integration Tests, Official30, documentation, and website workflows pass
      for the tagged commit.
- [ ] Python 3.11 and 3.13 boundaries, wheel installation, native installers,
      and backend/microscope jobs pass.

### 2. Public artifacts and clean journeys

- [ ] Wheel, sdist, direct installer assets, landing-page links, and registry
      metadata are public and resolve to the announced release.
- [ ] A clean Windows installer launch, MCP demo, Napari nonzero-layer check,
      and desktop shortcut check pass.
- [ ] A clean macOS installer launch, MCP demo, Napari smoke, and desktop app
      check pass.
- [ ] A fresh PyPI environment imports and launches with the documented extras.
- [ ] Unsigned/notarized installer warnings remain explicit on the landing page
      and release notes.

### 3. Scientific and agent journeys

- [ ] Add a representative plate, inspect bindings, author or import a
      pipeline, compile, run, and inspect materialized results.
- [ ] Reusing an output directory replaces current-run outputs and cannot
      contaminate measurement consolidation with stale execution data.
- [ ] Editing function code updates the live step declaration and saved
      pipeline; moved cached paths fall back to a valid directory.
- [ ] Napari and Fiji receive correctly indexed payloads over platform-safe
      transport.
- [ ] The two prompts above work through a fresh stdio client: the first cannot
      execute, and the second is bounded to A01/site 1.

### 4. Support boundary

- [ ] Every claim above links to documentation, a focused test, or a public CI
      result.
- [ ] Landing-page gallery media is captured from real UI state; no fabricated
      or unverified session is presented as a demo.
- [ ] Issue, security, contribution, conduct, and citation paths are public.
- [ ] The maintainer can answer installation and first-workflow replies during
      the first several days after posting.
