# Figure Plan

**Target:** 8 main figures for a Nature Methods-style platform manuscript.

The revised figure story follows the manuscript's accessibility rule: show the practical workflow problem first, then show how OpenHCS keeps the analysis as one workflow record. Compiler/runtime terms should appear only after the reader understands the user-facing value.

Editable diagram drafts are in `paper/figures/diagrams/*.dot`; rendered SVG/PNG drafts are in `paper/figures/rendered/`. Rebuild all diagrams and the contact sheet with `python paper/figures/render_diagrams.py`.

## Figure 1. Fragmented Bioimage Tools To One Workflow Record

Bioimage workflows often span CellProfiler, Fiji/ImageJ, napari, OMERO, Zarr-backed storage, Python notebooks, exported files, and batch execution. The figure should show the same analysis split across tools, then show OpenHCS keeping images, metadata, parameters, intermediate results, viewer outputs, generated Python, and workers attached to one workflow record.

**Main message:** Users can keep the tools they already use without splitting the workflow into disconnected records.

## Figure 2. Checks Before Execution

Show that OpenHCS resolves sources, parameters, functions, intermediate results, output destinations, and worker execution before a run starts. Use user-facing failure examples: wrong channel, stale mask, copied threshold, incompatible backend, output table from an old run, and ambiguous worker/viewer state.

**Main message:** OpenHCS catches workflow mistakes before long runs instead of treating them as hidden script behavior.

## Figure 3. Drop-In Python And Backend-Specific Functions

Show an ordinary Python function entering the workflow through signature-derived parameters and declared memory/backend requirements. Include NumPy/Numba, CuPy/CuCIM, JAX, PyTorch, TensorFlow, and pyclesperanto as possible backend paths when intentionally chosen.

**Main message:** OpenHCS is not a fixed module catalog; ordinary Python and backend-specific algorithms become workflow steps.

## Figure 4. Step-Level Viewer Output And Managed Sources Stay Attached

Show two distinct but connected paths. For imported CellProfiler workflows, image-loading semantics come from the `.cppipe` pipeline. For native or managed workflows, explicit source bindings connect local files, OMERO, and Zarr-backed stores. Separately, napari and Fiji are enabled on individual steps; OpenHCS launches or reuses the viewer on the configured port and streams that step's images during execution.

**Main message:** Imported workflows do not need extra manual source binding when loading is already encoded, and viewer output is a simple step-level configuration rather than a separate workflow.

## Figure 5. CellProfiler Import As A Preservation Test

Show `.cppipe` parsing into image sources, source mappings, CellProfiler-compatible workflow steps, named outputs, and parity comparison against native CellProfiler outputs.

**Main message:** CellProfiler compatibility is the strict validation case proving trusted legacy workflows can enter OpenHCS without losing meaning.

## Figure 6. Benchmark Validation Structure

Show the benchmark manifest feeding native CellProfiler and OpenHCS runs. Separate output parity, execution timing, total wall time, throughput, RAM, and category summaries.

**Main message:** Correctness, execution speed, cold-run overhead, and HCS throughput are reported separately.

## Figure 7. Single-Thread Speed And Many-Sample Throughput

Panel A should show the constrained one-sample, one-thread/core, CPU-only speedup distribution with the at-least-4x minimum target. Panel B should show persistent-worker throughput, samples per hour, worker count, sample count, and RAM.

**Main message:** The 4x minimum single-thread result is the quantitative floor; persistent workers extend the same workflow to HCS-scale throughput.

## Figure 8. GUI, Python, And Provenance Share One State

Show GUI editing, generated Python, inherited/default/local parameter values, dirty state, re-import, and runtime execution as views over the same workflow state.

**Main message:** OpenHCS makes workflows teachable and reviewable: visual edits, code, and execution refer to the same analysis.

## Supplementary Tables

## Supplementary Table 1. Benchmark Corpus

Each row should list one `.cppipe` workflow, source category, source URL or citation, dataset source, assay family, semantic pressure, output pressure, CellProfiler modules used, native CellProfiler runtime, OpenHCS execution runtime, total OpenHCS runtime, speedup, parity status, and notes.

## Supplementary Table 2. CellProfiler Module Coverage

Each row should list one CellProfiler module class, import status, parity-test status, accelerated path if relevant, backend used, unsupported settings or features, and notes.

## Supplementary Table 3. Worker/RAM Scaling

Each row should list a throughput condition, pipeline group, sample count, worker count, core count, peak RAM, RAM per worker, total wall time, samples per hour, speedup versus native CellProfiler, and speedup versus OpenHCS single-worker execution.

## Next Iteration

- Update DOT diagram names or labels to match this revised figure order.
- Keep technical labels secondary to the user-facing story.
- Add panel letters once diagrams become multi-panel composites.
- Re-render SVGs after DOT edits.
- Convert final SVGs to publication layout only after benchmark numbers and figure order are stable.
