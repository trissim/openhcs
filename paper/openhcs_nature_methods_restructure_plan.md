# OpenHCS Nature Methods Restructure Plan

## Purpose

This plan restructures the OpenHCS Nature Methods draft so that the paper is accessible to non-technical readers while preserving the technical depth needed for reviewers. The model is the MFD platform draft: start from a concrete practical bottleneck, explain the platform intervention in plain terms, validate it with demanding evidence, and move implementation details into supporting roles.

The current OpenHCS draft contains the right ingredients, but it over-centers CellProfiler in the title and early narrative and asks readers to understand too much architecture before they understand the value. The revised manuscript should frame OpenHCS as a composable bioimage workflow platform: useful for high-content screening, but not limited to high-volume screening. CellProfiler compatibility, parity, and at least 4x speedup are the strongest proof that the platform can preserve trusted existing workflows, not the whole product identity. The other central claim is that preserved workflows do not become dead ends: they enter the same editable state system as custom Python functions, viewer artifacts, storage bindings, generated code, workers, and backend-specific algorithms.

## Editorial Diagnosis

The current draft is strongest when it describes an ordinary lab workflow: images in one place, trusted analysis in another, inspection in napari or Fiji, custom Python in a notebook, measurements exported elsewhere, and batch execution as a separate concern. That is the accessible paper.

The draft is weakest when it explains OpenHCS by naming the internal machinery first. Terms such as semantic integration, FunctionStep, ObjectState, PolyStore, ArrayBridge, runtime artifacts, materialization plans, and ZMQRuntime are technically accurate, but they should not be the reader's first encounter with the value proposition. They belong in Methods, figure labels, or short supporting tables after the practical problem is clear.

The main edit is therefore not only a reorder. It is a narrative hierarchy change:

1. Practical workflow problem.
2. OpenHCS keeps the whole analysis as one workflow record.
3. Users keep the tools they already use and combine them safely.
4. CellProfiler parity and 4x minimum single-thread speedup prove this is serious.
5. Architecture explains why it works.

The next manuscript edit should make clear that the integrated workflow path is already implemented and tested, not a proposed future direction. The current draft still uses too much "a user can..." language. For implemented features, Results should use evidence language: "OpenHCS imports", "OpenHCS binds", "OpenHCS streams", "OpenHCS executes", "the benchmark includes", and "the tests exercise".

## Main Claim

OpenHCS lets labs compose bioimage workflows across existing tools, storage systems, viewers, custom Python functions, and compute backends without losing the connection between images, parameters, outputs, and execution.

CellProfiler import is the hardest preservation test: if OpenHCS can preserve trusted `.cppipe` workflows, reproduce native outputs, and run every tested workflow at least 4x faster under constrained CPU-only conditions, then the platform is not just a wrapper or convenience GUI. It is a workflow platform that can absorb existing analyses and make them inspectable, editable, extensible, and faster.

The main user-facing promise is that labs do not have to give up tools they already use. If a lab already uses CellProfiler, OMERO, Fiji, napari, local Python, GPU libraries, or conventional file outputs, OpenHCS should let those tools become synergistic parts of one workflow rather than mutually exclusive choices.

The second user-facing promise is that OpenHCS is not a dead-end compatibility layer. Imported workflows can be inspected, parameterized, round-tripped between GUI and Python, extended with ordinary Python functions, routed to implemented viewer integrations, and combined with backend-specific algorithms when those algorithms are appropriate for the assay. CellProfiler parity validates legacy preservation; GPU/backend extensibility validates a different use case: enabling new or non-CellProfiler-equivalent analyses inside the same workflow model.

## Implemented Demonstration Path

The manuscript should present one integrated OpenHCS path as completed evidence:

1. Import real CellProfiler `.cppipe` workflows.
2. Preserve the image-loading semantics encoded in the `.cppipe` workflow. Do not imply that imported CellProfiler workflows require a separate manual source-binding step when the pipeline already defines loading, metadata, and names/types behavior.
3. Inspect named intermediate masks or outputs by enabling the napari or Fiji step configuration. OpenHCS auto-opens the viewer on the selected port and streams images from that step while the pipeline runs.
4. Add ordinary Python functions through the same workflow-step mechanism used by absorbed CellProfiler modules. This is technically covered because CellProfiler-compatible module absorption already enters the OpenHCS function pathway, and native/custom Python functions use the same mechanism.
5. Run output parity against native CellProfiler.
6. Run constrained single-thread/core speed benchmarks.
7. Scale execution across wells/workers and report throughput/RAM.

This path should not read as a hypothetical tutorial. It should read as the evidence spine of the paper. Where final benchmark numbers are still marked `TODO`, the exact values remain placeholders, but the existence of the implemented path is not tentative.

Suggested Results 1 replacement title:

> A complete OpenHCS workflow combines imported CellProfiler pipelines, managed image sources, viewer inspection, Python extension, and worker execution

Suggested opening paragraph:

> We exercised the full OpenHCS workflow path by importing real CellProfiler `.cppipe` workflows with their image-loading semantics intact, enabling napari or Fiji output on selected steps so OpenHCS auto-opens the viewer and streams images while the pipeline runs, adding Python functions through the same workflow-step mechanism used by absorbed CellProfiler modules, and running parity, single-thread speed, and worker-scaling benchmarks. These operations use one workflow record rather than separate import, viewer, scripting, and benchmark pathways.

Use this section to turn Results 1-4 from capability descriptions into evidence-backed manuscript sections.

## Accessibility Goal

Non-technical readers should understand the value before they encounter internal names such as ObjectState, ArrayBridge, PolyStore, ZMQRuntime, FunctionStep, source bindings, materialization plans, artifact contracts, or "semantic integration platform." The manuscript should avoid leading with phrases that sound like computer-science jargon to biologists. "Composable bioimage workflow platform" is the preferred plain-language frame.

Use "one workflow record" or "one workflow" in the main text. Use "workflow object", "typed state", "runtime artifact", and related terms only when explaining implementation. Internal systems should appear after their user-facing consequences: traceable state, GUI/code round-trip, custom function insertion, viewer-backed inspection, managed sources, backend-specific execution, and persistent workers.

The manuscript should use this explanation order:

1. Labs already use many good tools, but the workflow breaks at the boundaries.
2. OpenHCS keeps images, metadata, parameters, intermediate outputs, viewers, Python functions, storage, and compute backends connected as one workflow record.
3. Users can keep the tools they already trust while adding custom Python, viewer inspection, managed storage, reactive GUI/code editing, parallel workers, and backend-specific algorithms.
4. CellProfiler parity plus speedup proves the abstraction preserves meaning and improves execution.
5. The architecture explains how this is possible, but it is not the first thing readers must learn.

## Language And Wording Rules

The rewrite should follow the MFD draft's accessibility pattern: technical details stay present, but they are not required to understand the value. A biologist who has used CellProfiler, Fiji, napari, OMERO, or Python notebooks should understand the main claim before encountering compiler/runtime terminology.

Use reader-facing terms before implementation terms:

| Prefer in main text | Technical term to introduce later |
|---|---|
| workflow record | typed state, workflow object |
| intermediate result | runtime artifact |
| where outputs go | materialization plan |
| image source | source binding |
| Python function | FunctionStep |
| viewer output | PolyStore backend |
| worker process | ZMQRuntime |
| memory/backend requirement | memory contract |
| checked before the run | compiler validation |
| same workflow in GUI and code | pycodify/ObjectState round-trip |

Prefer concrete lab actions:

- inspect a mask in napari
- send an ROI-style output to Fiji
- change a threshold in the GUI
- export the same workflow as Python
- load images from OMERO or Zarr-backed storage
- add a custom Python quality-control function
- rerun the same workflow across wells
- compare imported CellProfiler outputs to native CellProfiler outputs

Use implemented-evidence verbs when the feature exists:

| Avoid if implemented | Prefer |
|---|---|
| a user can import | OpenHCS imports |
| can bind images | imported `.cppipe` workflows preserve their image-loading semantics; native/managed workflows use OpenHCS source bindings |
| can stream to napari | enabling napari on a step auto-opens the viewer and streams that step's images |
| can export to Fiji | enabling Fiji on a step auto-opens Fiji and streams that step's images |
| can add Python functions | Python functions enter the workflow through the same step mechanism |
| can run benchmarks | parity and speed benchmarks exercise the workflow |
| can scale across workers | worker-scaling benchmarks report throughput and RAM |

Avoid abstract platform language unless it is immediately explained:

- Do not lead with "semantic integration".
- Do not lead with "execution substrate".
- Do not lead with "artifact contract".
- Do not lead with "materialization".
- Do not lead with package names.
- Do not use long ecosystem lists unless each item is tied to a user action.

Use this sentence pattern for main-text explanations:

1. Start with what the user can do.
2. State what OpenHCS keeps connected.
3. Name the technical mechanism only if it clarifies the claim.

Example:

Bad:

> PolyStore materializes runtime artifacts through backend-specific destination policies.

Good:

> The same mask can be saved, compared, or streamed to napari because OpenHCS treats it as a named workflow output.

The main text should sound like a method for working scientists, not a systems paper abstract. The Methods and supplementary material can use the precise internal names once the reader already understands why those mechanisms matter.

## Revised Title Options

Preferred:

> OpenHCS: a composable bioimage workflow platform

Alternatives:

> OpenHCS unifies bioimage workflows across legacy pipelines, viewers, storage, Python, and compute backends

> OpenHCS preserves trusted bioimage workflows while making them composable, inspectable, and faster

Avoid titles that make CellProfiler sound like the main product:

> OpenHCS: preserving CellProfiler workflows while unifying the high-content screening ecosystem

That title is directionally useful but too narrow. CellProfiler should appear in the abstract and validation results, not in the title.

## Abstract Restructure

### Current Issue

The current abstract starts with the ecosystem problem, which is good, but it quickly becomes a list of system roles and internal capabilities. CellProfiler import appears as the central proof, but the abstract does not sharply separate the platform claim from the validation claim.

### Proposed Abstract Shape

1. One sentence problem: bioimage workflows now span images, metadata, parameters, viewers, storage, Python, legacy pipelines, and heterogeneous compute.
2. One sentence failure mode: current tools are strong individually but force manual handoffs that break provenance, reproducibility, and extensibility.
3. One sentence OpenHCS solution: OpenHCS keeps the analysis as one workflow record with defined sources, parameters, functions, intermediate results, outputs, and execution plans.
4. One sentence user-facing outcome: imported CellProfiler workflows preserve their encoded loading semantics, native/managed workflows can use explicit source bindings, napari/Fiji output is enabled per step, and GUI/code editing, custom Python functions, backend-specific execution, and workers remain parts of one workflow instead of separate workflow versions.
5. One sentence validation: imported CellProfiler workflows reproduce native outputs across the benchmark corpus and run at least 4x faster under single-thread/core CPU-only conditions.
6. One sentence significance: the result is a practical path from trusted bioimage workflows to modern composable execution.

### Draft Replacement Abstract

Bioimage workflows increasingly span more than image analysis: they combine image metadata, multidimensional arrays, segmentation masks, measurement tables, managed image stores, interactive viewers, legacy pipeline formats, custom Python functions, and heterogeneous CPU/GPU backends. Existing tools remain essential, but their boundaries often turn one scientific workflow into disconnected GUI settings, scripts, exported files, viewer state, and batch jobs. We present OpenHCS, a composable bioimage workflow platform that keeps these pieces connected as one inspectable and executable workflow record. OpenHCS imports trusted CellProfiler pipelines with their loading semantics intact, streams selected step outputs to napari or Fiji when those step configurations are enabled, edits parameters through GUI and generated Python views of the same state, adds ordinary Python functions with declared memory backends, and scales execution across workers without changing what the workflow means. As a stringent validation, OpenHCS imports CellProfiler `.cppipe` workflows, reproduces native CellProfiler outputs under declared parity checks, and achieves at least `TODO: 4x` execution speedup across the benchmark corpus under one-sample, one-thread/core, CPU-only conditions. OpenHCS therefore provides a practical route for carrying trusted bioimage analyses into a composable, inspectable, and extensible execution platform rather than preserving them as static legacy files.

## Introduction Restructure

### Current Issue

The current introduction is accurate but reads like an ecosystem catalog. It lists many tools and then explains OpenHCS architecture. This can make non-technical readers feel they need to understand every tool boundary before understanding the value.

### Proposed Flow

#### Paragraph 1: The practical lab problem

Focus on what happens in a lab: images in one place, metadata in another, segmentation in CellProfiler or Python, inspection in napari/Fiji, outputs in CSVs, custom analysis in notebooks, and batch execution elsewhere. Keep the scope bioimage-wide: OpenHCS is capable of HCS, but it should not read as useful only for high-volume screening.

Draft:

> A bioimage analysis is rarely contained in one program. A lab may store images in OMERO, Zarr-backed storage, or acquisition folders, run a trusted CellProfiler pipeline, inspect masks in napari or Fiji, add a custom Python quality-control step, export measurement tables, and later repeat the workflow across many wells or a small focused dataset. Each tool is useful, but the scientific workflow is the thing that has to remain coherent: the same images, channels, parameters, masks, measurements, and outputs must refer to the same analysis state.

#### Paragraph 2: The boundary failure

Avoid tool-bashing. State that existing tools are strong but the handoffs are weak.

Draft:

> The problem is not that these tools are inadequate. CellProfiler, Fiji/ImageJ, napari, OMERO, Zarr-backed image storage, scientific Python, and GPU libraries each solve important parts of the problem. The failure mode appears between them. Parameters are copied between GUIs and scripts, intermediate images are saved as files whose provenance is no longer clear, viewers inspect masks that may not correspond to the current run, and custom code loses the dimensional and metadata assumptions carried by the original workflow.

#### Paragraph 3: OpenHCS solution in user language

Explain the shared workflow record before architecture.

Draft:

> OpenHCS addresses this boundary problem by keeping the workflow itself as the shared record. Images, source metadata, parameters, functions, intermediate masks, measurement tables, exported files, viewer streams, and worker execution are named parts of one workflow. A GUI edit, generated Python file, viewer request, imported CellProfiler module, or custom Python function all target the same analysis instead of creating parallel versions. This is what prevents imported workflows from becoming dead ends: they can be inspected, inherited, overridden, extended, serialized, and rerun through the same state model.

#### Paragraph 4: Technical mechanism, still accessible

Introduce architecture terms only after value is clear.

Draft:

> Internally, OpenHCS uses typed source schemas, function contracts, runtime artifacts, storage backends, memory-backend conversion, and process-isolated workers. These terms are implementation details of a simple user-facing guarantee: before a run starts, OpenHCS checks what data exist, what each step consumes and produces, where outputs will go, which memory backend is required, and which worker will execute the work.

#### Paragraph 5: Validation through CellProfiler

Position CellProfiler as proof, not product identity.

Draft:

> CellProfiler compatibility is the strictest test of this model because `.cppipe` files encode years of trusted biological image-analysis practice. OpenHCS does not treat them as opaque files or run them in a separate sidecar. It compiles them into normal OpenHCS workflows, compares outputs against native CellProfiler, and reports speed only after parity is established.

Add one clarifying sentence after this paragraph:

> CellProfiler parity is not the standard for every OpenHCS analysis; it is the preservation test for imported CellProfiler workflows. Native OpenHCS functions and GPU/backend-specific algorithms can be used when a lab intentionally chooses a different analysis method.

#### Paragraph 6: Paper overview

Draft:

> We show that OpenHCS keeps bioimage workflows coherent across legacy pipelines, Python functions, viewers, storage backends, and worker execution. We validate this by importing a broad CellProfiler workflow corpus with loading semantics intact, reproducing native outputs under output-parity checks, measuring constrained CPU-only speedups, and demonstrating how the same workflow model supports step-level viewer output, generated Python, custom functions, and scalable execution.

## Results Restructure

The Results section should be shorter and more evidence-driven. The current draft has many useful subsections, but several repeat the same claim. Proposed main Results sequence:

### Result 1: One workflow record across tools

Goal: Establish the platform claim with an implemented integrated workflow path, not a hypothetical lab-facing story.

Use content from current sections:

- "A familiar CellProfiler workflow becomes an editable OpenHCS workflow"
- "OpenHCS keeps image analysis, measurements, viewers, and code in one workflow"
- "Imported workflows support several lab-facing use cases"

Edit direction:

- Start from the implemented workflow path: real CellProfiler imports with their image-loading semantics intact, Python workflow steps, napari/Fiji step output enabled by config, editable GUI/Python state, output parity, single-thread speed, and worker throughput.
- Show the same workflow record moving through all surfaces.
- Avoid naming every internal library.
- Emphasize that users keep the tools they already use and gain synergy instead of being forced to choose one ecosystem.
- Make the "not a dead end" point explicit: importing a workflow should make it more inspectable and extensible, not freeze it as a compatibility artifact.
- Use evidence verbs. Avoid making already-tested features sound aspirational.

### Result 2: OpenHCS checks the workflow before execution

Goal: Explain why this is not a script wrapper without making readers learn the compiler first.

Use content from:

- "OpenHCS checks workflow meaning before the run starts"
- "Intermediate images, masks, and tables become named workflow products"
- "Parameters become traceable instead of fragile"

Edit direction:

- Keep the table, but rewrite headers in user-facing language. Example headers: "What OpenHCS checks", "Why the user cares", and "Failure avoided".
- Reduce architecture vocabulary.
- Make examples concrete: wrong channel, stale mask, copied threshold, output table not matching current run.
- Preserve the reactive state / git-like provenance point. It is not cosmetic; it is the mechanism that lets users understand inherited values, local overrides, GUI edits, generated Python, and what changed between workflow versions.

### Result 3: Any Python function can become a workflow step

Goal: Elevate this as a central feature, not a side note.

Needed content:

- Drop-in Python functions.
- Declared memory backends: NumPy, Numba, CuPy, CuCIM, JAX, PyTorch, TensorFlow, pyclesperanto.
- Automatic UI/signature handling.
- Backend conversion through ArrayBridge.

Edit direction:

- Make this a standalone result before CellProfiler.
- Explain that OpenHCS is not a fixed module catalog.
- CellProfiler modules are one dialect; custom Python functions and backend variants are equally native.
- State that custom functions are the escape hatch that prevents imported or GUI-built workflows from becoming closed systems.

### Result 4: Managed stores and viewers stay attached to the workflow

Goal: Cover OMERO, Zarr-backed storage, Fiji, and napari as first-class integration features.

Use content from:

- "OpenHCS unifies the tools biologists already use"
- "OMERO and institutional image stores become workflow sources, not side archives"
- Viewer integration material from current draft.

Edit direction:

- Separate source-side integration from viewer/output-side integration.
- Emphasize the actual simple viewer behavior: enabling napari or Fiji config on a step auto-opens the viewer and sends that step's images while the pipeline runs.
- Keep source binding separate from CellProfiler import. For imported `.cppipe` workflows, loading semantics are derived from the pipeline. For native or managed-store workflows, OpenHCS source bindings cover local, OMERO, Zarr-backed, and other source paths. Explicit OME-Zarr support should be a follow-up implementation/test item unless verified separately.
- Current evidence framing: OMERO integration runs in CI across multiple versions on each push; Fiji and napari are both implemented viewer integrations. The distinction is test environment, not support level: Fiji requires a GUI-dependent test environment and is therefore tested outside CI, while napari can be exercised through the available automated/viewer path. The paper should make implementation/test status explicit where space allows without implying Fiji is less supported than napari.
- Do not say "could be streamed" or "could be bound" for tested paths. Say which integration paths are implemented/tested and reserve tentative language only for future or unverified deployment modes.

### Result 5: CellProfiler workflows compile into normal OpenHCS workflows

Goal: Introduce the hardest compatibility proof after the platform value is established.

Use current "CellProfiler pipelines compile into normal OpenHCS pipelines" section.

Edit direction:

- Keep the mapping table.
- Shorten surrounding prose.
- Say explicitly: CellProfiler import validates the generality of the model because it preserves a mature external workflow format.

### Result 6: Parity across a broad benchmark corpus

Goal: Correctness before speed.

Use current benchmark corpus section.

Edit direction:

- Put corpus size and categories up front once final values are stable.
- Keep source category / assay family / semantic pressure / output pressure if they are backed by manifest fields.
- Make clear that a pipeline only counts after output parity passes under the declared comparison policy.

### Result 7: At least 4x faster under constrained single-thread CPU-only conditions

Goal: Performance proof without overclaiming.

Use current speed section.

Edit direction:

- State one-thread or one-core, one-sample, CPU-only as conservative.
- Report min/median/mean/max from final artifacts.
- Make "at least 4x on every tested pipeline" the quantitative headline and lead-in to the performance figures if final data support it.
- Do not bury the 4x claim inside platform prose.

### Result 7b: Backend-specific algorithms extend workflows beyond CellProfiler parity

Goal: Clarify the role of GPU/backend support without requiring CellProfiler-equivalent behavior.

Use content from current GPU/backend section, but change the framing:

- CellProfiler parity validates preservation of imported `.cppipe` workflows.
- GPU, CuCIM, pyclesperanto, JAX, PyTorch, TensorFlow, and other backend paths support native or custom OpenHCS functions when a lab intentionally chooses those algorithms.
- These algorithms do not need to be parity-equivalent to CellProfiler unless they are presented as replacements for a CellProfiler module.
- The claim should be: OpenHCS keeps backend-specific algorithms inside the same workflow/state/artifact model, so users can extend an analysis without leaving the platform.

### Result 8: Many-well throughput and persistent workers

Goal: Show HCS-scale operating mode.

Use current throughput section.

Edit direction:

- Tie to practical HCS: fixed costs amortize across wells.
- Report samples per hour and RAM per worker.
- Keep formula if useful, but make it optional or figure-supported.

## What To Cut Or Move Out Of Main Text

Move these elements to Methods, supplement, or figure notes unless they are needed for a specific claim:

- Reusable-library inventory paragraphs that list ObjectState, ArrayBridge, PolyStore, ZMQRuntime, pyqt-reactive, pycodify, python-introspect, and metaclass-registry together.
- Detailed CellProfiler module mapping prose beyond one concise table and one explanation of why parity is strict.
- Long benchmark grouping explanations if the figure and supplement can carry the exact category definitions.
- Multiple restatements that OpenHCS is not only a faster CellProfiler runner. Say it once in the Results transition and once in Discussion.
- Phrases that are correct but inaccessible as entry points: semantic integration, semantic execution substrate, materialization plan, runtime artifact, backend contract, source-binding configuration.

Keep these elements in main text because they are user-facing:

- One workflow record across tools.
- GUI/code round-trip and traceable parameter edits.
- Drop-in Python functions and backend-specific algorithms.
- CellProfiler-imported loading semantics, native/managed source bindings, and step-level napari/Fiji output.
- CellProfiler output parity before performance claims.
- At least 4x minimum single-thread/core speedup as the quantitative lead.

## Discussion Restructure

### Current Issue

Discussion repeats several Results claims and spends too much space explaining that OpenHCS is not only CellProfiler.

### Proposed Discussion Topics

1. OpenHCS shifts integration from import/export to one workflow record.
2. CellProfiler results show serious compatibility because preserved outputs and speed gains are measured together.
3. The platform does not stop at compatibility: reactive state, GUI/code round-trip, custom functions, viewer artifacts, storage bindings, backend variants, and workers make preserved workflows extensible.
4. Limitations: supported CellProfiler module/settings subset, backend parity requirements, managed-store deployment complexity, GPU variant validation, worker lifecycle/RAM tradeoffs.
5. Field implication: labs can keep trusted workflows while adopting modern execution and inspection.

### Draft Discussion Opening

> OpenHCS is built around a simple claim: bioimage workflows should remain coherent as they move between tools. The system is not defined by any single integration. CellProfiler import, OMERO and Zarr-backed source handling, Fiji and napari inspection, custom Python functions, backend memory conversion, generated Python, reactive parameter state, and persistent workers all depend on the same underlying idea: the workflow remains one record with named sources, parameters, functions, intermediate results, outputs, and execution plans. This is why compatibility does not become a dead end: a preserved workflow can still be inspected, edited, extended, and executed through the rest of the platform.

## Methods Restructure

Methods should preserve technical depth but be moved out of the main narrative where possible.

Proposed Methods order:

1. Pipeline object model and compilation.
2. Source schema and source binding.
3. Function registration and memory-backend contracts.
4. Runtime artifacts and materialization.
5. Viewer and storage backends.
6. Worker execution and ZMQRuntime.
7. CellProfiler import.
8. Parity comparison.
9. Benchmark timing and throughput.
10. Reproducibility package.

## Figure Plan Changes

The existing figure plan is close but should shift emphasis.

### Figure 1: The broken bioimage workflow boundary

Show the same experiment fragmented across CellProfiler, Python, napari/Fiji, OMERO, Zarr-backed storage, exported CSVs, and batch workers. Then show OpenHCS keeping them as one workflow.

### Figure 2: OpenHCS as one workflow record

Show sources, parameters, functions, intermediate results, outputs, viewers, and workers as parts of one workflow record. Keep compiler/runtime details lower-level.

### Figure 3: Drop-in Python and backend contracts

New or elevated figure. Show ordinary Python functions entering through signature/metadata, memory decorators, ArrayBridge conversion, and backend variants. Make clear that backend-specific algorithms are first-class workflow steps even when they are not parity-equivalent to a CellProfiler module.

### Figure 4: Step-level viewer output and managed sources

Show that imported CellProfiler workflows preserve encoded loading semantics, native/managed workflows can use explicit source bindings, and napari/Fiji output is enabled on individual steps. The figure should show viewers auto-launched or reused on configured ports and receiving images while the pipeline runs.

### Figure 5: CellProfiler compiler dialect

Move current Figure 3 here. This is validation, not the opening identity.

### Figure 6: Parity benchmark structure

Keep current Figure 4.

### Figure 7: CPU-only speed and throughput

Combine single-core speedup and many-well worker throughput if possible.

If the figure includes backend acceleration, separate it visually from the CellProfiler parity benchmark. CellProfiler parity is a preservation benchmark; backend-specific algorithms are an extensibility path.

### Figure 8: Typed editing and provenance

Keep current typed state figure, but make it user-facing: GUI, Python, assistant, runtime all edit or inspect the same state.

## Concrete Edit Queue

### Pass 1: Reframe top matter

- Replace title with the preferred composable bioimage workflow platform title.
- Replace abstract with the platform-first abstract above.
- Add a short editorial note that CellProfiler numbers remain TODO until final artifacts are frozen.
- Remove "preserving CellProfiler workflows" as the primary title frame.

### Pass 2: Rewrite introduction

- Replace current ecosystem-catalog introduction with the six-paragraph flow above.
- Keep citations to CellProfiler, Fiji/ImageJ, napari, OMERO, Zarr-backed image storage, scientific Python, GPU libraries, and workflow systems.
- Reduce Table 1 or move it to supplement if it slows the opening.

### Pass 3: Reorder Results

- Move custom Python/backend integration before CellProfiler.
- Move OMERO/Fiji/napari integration before CellProfiler validation.
- Consolidate repeated "one workflow" and architecture-heavy prose into the one-workflow-record frame.
- Preserve CellProfiler parity/speed sections but make them validation results.
- Preserve and elevate reactive UI/git-like state management as a user-facing result, either in Result 2 or as a short dedicated subsection, because it is central to workflow editability and reviewability.
- Rewrite Results 1-4 as completed evidence using the implemented demonstration path, not as hypothetical capability descriptions.

### Pass 4: Tighten technical language

- Replace internal-library-first sentences with user-facing consequence-first sentences.
- Keep internal library names in a table or Methods.
- Avoid requiring readers to know FunctionStep, ObjectState, PolyStore, ArrayBridge, runtime artifact, materialization, source binding, or ZMQRuntime before the benefit is clear.
- Do not demote custom functions, reactive state, GUI/code round-trip, or viewer integrations to mere implementation details; demote the package names, not the user-facing capabilities.

### Pass 5: Update figures

- Revise `paper/06_figures.md` to match the platform-first figure order.
- Keep existing DOT diagrams where reusable, but likely rename/reorder Figure 3 and Figure 4 concepts.
- Add a dedicated drop-in Python/backend figure if not already represented clearly.

### Pass 6: Create shorter main-text draft

- Target 3,500-4,500 main-text words before Methods.
- Move detailed compatibility boundaries, reusable-library descriptions, benchmark corpus details, and module coverage tables to supplementary sections.
- Keep main-text claims tied to final benchmark artifacts.

## Claims That Need Final Evidence Before Submission

- Exact number of benchmark workflows.
- Exact official/tutorial/third-party breakdown.
- At least 4x minimum single-thread/core speedup across all tested workflows.
- Median, mean, and maximum speedup.
- Full parity status and tolerated differences.
- Module coverage table.
- Source integration status and evidence: imported `.cppipe` workflows preserve their own loading semantics; OMERO is implemented and CI-tested across multiple versions on each push; current manuscript claims should say Zarr-backed storage unless explicit OME-Zarr support is added and tested.
- Fiji and napari integration status and evidence: both are implemented viewer integrations. Fiji requires GUI-dependent testing outside CI; napari can use the available automated/viewer test path. Do not imply Fiji is less supported than napari solely because CI cannot exercise it.
- GPU/backend claims: distinguish preservation from extension. GPU/backend-specific algorithms do not need CellProfiler parity unless claimed as replacements for CellProfiler modules. They should be described as implemented/native extensibility paths when evidence supports them.
- Reactive UI / git-like state management / GUI-code round-trip evidence: identify the strongest tests, screenshots, or examples showing inherited values, local overrides, dirty state, generated Python, re-import, and cross-surface consistency.
- Custom function evidence: identify a concise example showing an ordinary Python function entering the workflow with signature-derived parameters and declared memory/backend behavior.

## Resolved Framing Decisions

1. The paper should lead with "composable bioimage workflow platform," not "semantic integration platform" and not HCS-only framing. HCS is a major capability and validation setting, but OpenHCS should also read as useful for lower-volume bioimage workflows.
2. CellProfiler should not be in the title. It belongs in the abstract and Results as the stringent compatibility and performance proof.
3. Integrations should be shown as one workflow record, not as separate examples competing for the main story. If readers already use CellProfiler, OMERO, Fiji, napari, Python, or backend-specific libraries, the message is that OpenHCS lets them keep using those tools together.
4. OMERO, Fiji, and napari claims can be framed as implemented integrations, with evidence notes: OMERO runs in CI across multiple versions on each push; Fiji and napari are both implemented viewer integrations, with Fiji tested outside CI because of GUI requirements.
5. The quantitative headline should be at least 4x minimum speedup on a single thread/core, leading directly into the performance figures. Throughput and worker scaling should follow as the HCS-scale extension.
6. GPU/backend support should not be framed as needing CellProfiler parity by default. Parity is required for claims about preserving CellProfiler workflows; backend-specific algorithms are valid when presented as intentional workflow extensions.
7. Reactive UI, git-like state management, GUI/code round-trip, and custom functions should be treated as core platform features. They are what make OpenHCS an extensible workflow substrate rather than a dead-end importer.
