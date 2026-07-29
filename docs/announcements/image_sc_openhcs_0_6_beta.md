# image.sc announcement: OpenHCS 0.6 beta

Post only after every required item in the readiness checklist is checked
against the public release being announced.

## Proposed title

OpenHCS 0.6 beta: open high-content screening workflows in a GUI, Python, Napari/Fiji, and MCP

## Ready-to-paste post

Hello image.sc community,

I am releasing the OpenHCS 0.6 beta and would value feedback from people who
build or maintain high-content microscopy workflows.

OpenHCS is an open-source, Python-native platform for high-content screening.
Its desktop GUI, generated Python, headless runtime, CellProfiler importer, and
optional MCP server all use the same public `PipelineConfig` plus ordered
`FunctionStep` declarations. Pipelines are compiled before execution so source
matching, configuration, artifact routing, memory contracts, and worker
requirements can fail early rather than after a large screen has started.

The beta currently includes:

- visual pipeline authoring with bidirectional GUI ↔ Python editing;
- source bindings for plate fields such as well, site, channel, Z, and
  timepoint;
- ImageXpress and Opera Phenix plate parsing, plus Bio-Formats and OMERO-backed
  source/storage integration; the separate OMERO web panel remains
  experimental;
- supported CellProfiler `.cppipe` import into ordinary OpenHCS declarations;
- live image and result streaming to Napari and Fiji/ImageJ;
- CPU multiprocessing, optional CUDA backends, and persisted custom Python
  functions;
- typed images, object labels, measurements, relationships, tables, grids,
  graphs, and metadata, with ROI and file materialization;
- an optional local MCP server so compatible agents can inspect the same
  schemas, documentation, pipeline state, runtime progress, and viewer state.

For CellProfiler interoperability, OpenHCS uses a source-backed corpus of 30
example pipelines as end-to-end regression evidence. The current release
passes that Official30 suite under the documented equivalence policies. This
is not a claim that every historical CellProfiler module, plugin, version, and
setting is supported: unfamiliar pipelines should still be imported and
compiled to expose the exact compatibility boundary.

### Try it

For Windows or macOS, the landing page links directly to small graphical
installers; no ZIP extraction or existing Python installation is required:

https://openhcsdev.github.io/openhcs/

For an existing Python 3.11–3.13 environment:

```bash
python -m pip install "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"
openhcs
```

The desktop installers are CPU-first and install CellProfiler compatibility,
Napari, Fiji/PyImageJ, Bio-Formats, and local MCP support. Fiji's Java
components are resolved on first use, and GPU packages are opt-in. Current
installer assets are unsigned, so Windows SmartScreen or macOS Gatekeeper may
ask you to confirm that you trust them.

A five-minute interface demo is available here:

https://openhcs.readthedocs.io/en/latest/_static/openhcs.mp4

Documentation:

https://openhcs.readthedocs.io/

Source and issue tracker:

https://github.com/OpenHCSDev/openhcs

### Feedback I am looking for

I would especially like to hear from people who have:

- an ImageXpress or Opera Phenix plate with a nontrivial naming/layout pattern;
- an existing CellProfiler pipeline they would like to import;
- multi-site, multi-channel, Z-stack, or time-series assays;
- workflows that produce measurements, ROIs, or CellProfiler Analyst outputs;
- a need to move between visual editing, Python, interactive viewers, and
  automated execution.

If you are willing to try it, please reply with the microscope format, image
axes, biological goal, and expected outputs. I am also happy to help the first
few labs migrate one representative pipeline and use that experience to
improve onboarding.

OpenHCS is MIT licensed. Thank you for any testing, criticism, or workflow
examples you can share.

## Release readiness checklist

The release is a **go** only when every required item is verified against the
same commit and public version.

### 1. Source and release identity

- [ ] `origin/main`, the annotated version tag, and the GitHub Release resolve
      to the same commit.
- [ ] Package, Codex plugin, MCP bundle, and `server.json` metadata all project
      the announced version from `openhcs.__version__`.
- [ ] The worktree contains no uncommitted production changes.
- [ ] The PyPI project classifier is `Development Status :: 4 - Beta`.

### 2. Automated acceptance

- [ ] The complete GitHub Integration Tests workflow passes for the tagged
      commit.
- [ ] Official30 headless CellProfiler parity passes in that workflow.
- [ ] Python 3.11 and 3.13 boundary jobs pass on Linux, Windows, and macOS.
- [ ] Wheel installation, desktop installer, backend/microscope, and OMERO jobs
      pass.
- [ ] Documentation and Website workflows pass.
- [ ] The persisted-custom-function spawn regression passes with two genuine
      worker processes.
- [ ] The typed `run_plate` action is enabled during execution and dispatches
      Stop; cancellation returns the manager to an actionable state.

### 3. Public artifacts

- [ ] PyPI exposes the exact announced wheel and sdist, neither yanked.
- [ ] The GitHub Release is public, non-draft, and marked latest.
- [ ] `OpenHCS-Windows-Installer.exe` and
      `OpenHCS-macOS-Installer.dmg` are attached without ZIP wrapping.
- [ ] Both landing-page “latest/download” installer URLs return HTTP 200 and
      resolve to the announced release.
- [ ] The landing page displays the announced version and does not describe
      patch-level fixes as newly introduced platform features.
- [ ] The MCP Registry lists `io.github.OpenHCSDev/openhcs` at the announced
      version, marks it latest, and pins
      `openhcs[gui,mcp,viz]==<announced-version>`.

### 4. Installation and first-use journeys

- [ ] A clean Windows installer run completes, creates a desktop launcher,
      launches OpenHCS, and registers detected supported MCP clients without
      overwriting unrelated entries.
- [ ] A clean macOS DMG run completes the equivalent user-scoped installation
      and launch journey.
- [ ] Installer progress, failure logs, startup splash output, updater, and
      unsigned-installer warnings are accurate.
- [ ] A fresh PyPI environment can import the base package and launch the GUI
      with the documented desktop extras.
- [ ] `openhcs-mcp-demo --json` completes its portable workflow, observes
      nonzero data in Napari, and cleans up its allocated endpoints.

### 5. Scientific workflow journeys

- [ ] Add a representative plate, inspect source inventory/bindings, author or
      import a pipeline, compile it, run it, and inspect materialized results.
- [ ] A supported `.cppipe` imports through the GUI and exposes ordinary
      OpenHCS steps without a hidden CellProfiler runtime.
- [ ] Measurement rows retain biological coordinates and aggregate output
      names do not imply site/channel coordinates that were aggregated away.
- [ ] Reusing an output directory cannot contaminate the current summary with
      stale measurements from an older execution.
- [ ] Napari and Fiji receive correctly indexed payloads through their tested
      viewer routes.
- [ ] A persisted custom function runs with at least two spawned workers,
      materializes its declared outputs, cancels cleanly, and can be replaced
      and rerun without restarting the GUI.

### 6. Announcement and support

- [ ] Every feature claim above has a documentation, test, or live-release
      evidence boundary.
- [ ] The screenshot preview and five-minute demo URLs are public.
- [ ] Documentation, GitHub issues, PyPI, release, installer, and MCP Registry
      links are public and current.
- [ ] The post states CPU-first/GPU opt-in and unsigned-installer boundaries.
- [ ] The requested feedback is specific enough for a biologist to respond
      with microscope format, axes, assay goal, and expected outputs.
- [ ] The maintainer is prepared to answer installation and first-workflow
      replies during the first several days after posting.
