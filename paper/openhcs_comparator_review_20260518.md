# OpenHCS Draft Comparator Review

## Scope

Reviewed `paper/openhcs_nature_methods_draft.md` against the existing Nature Methods comparator set plus MCMICRO and BIOMERO on 2026-05-18.

## Overall Judgment

The current draft is now structurally close to the successful platform-paper pattern:

1. It starts with image analysis as the bottleneck after acquisition.
2. It treats existing tools as valuable rather than obsolete.
3. It states OpenHCS as one workflow record instead of a tool catalog.
4. It separates CellProfiler preservation from the broader platform claim.
5. It keeps Bio-Formats claims gated until implementation evidence exists.

The remaining weakness is not the high-level framing. The remaining weakness is proof concreteness. MCMICRO and BIOMERO make their platform stories feel concrete by naming the exact user workflow shape: raw images to single-cell data for MCMICRO; OMERO datasets to HPC-executed workflows for BIOMERO. OpenHCS should make its equivalent path just as tangible.

## Revised North Star: Universal Bioimage Execution Engine

The comparison against MCMICRO, BIOMERO, CellProfiler, Fiji/ImageJ, napari, OMERO, Bio-Formats, Python libraries, and workflow managers suggests a stronger OpenHCS frame:

> OpenHCS is the common execution layer for bioimage analysis. Existing tools keep their identities, but their workflows, functions, sources, viewers, outputs, and worker execution can run through one non-ad-hoc OpenHCS model.

This should guide the next draft pass, but should be translated into biologist-facing language. "Universal execution engine" is a useful internal framing phrase; the manuscript should usually say "common execution layer", "one workflow record", or "one execution model" unless the surrounding text defines what "execution engine" means.

Core implications:

- CellProfiler users should not have to abandon `.cppipe` workflows. They should be able to run trusted pipelines through OpenHCS, preserve semantics under parity checks, inspect intermediates, add Python functions, and scale execution.
- BIOMERO-style deployments should not be treated as competitors. BIOMERO or OMERO-centered systems could launch OpenHCS workflows on SLURM/HPC because OpenHCS is the analysis execution layer, not the image server.
- Python users should not need to build a plugin ecosystem around every function. Any registered callable can become a workflow step with UI-visible parameters, declared array/backend requirements, source semantics, side-channel outputs, and worker execution.
- Fiji, napari, OMERO, Zarr, microscope folders, and future Bio-Formats-backed discovery should be treated as sources, destinations, viewers, or launch surfaces around the same execution model.
- The central claim is non-ad-hoc composability: OpenHCS does not glue tools together through one-off scripts; it maps callables, arrays, side channels, source identities, intermediate results, and execution workers into one inspectable workflow record.

Manuscript caution:

- Avoid saying "irreplaceable" directly. It sounds like marketing and invites reviewer pushback.
- Instead, show why the architecture is hard to replace with ad-hoc scripts: ordinary callables, backend-aware arrays, source identities, named intermediates, viewer destinations, generated Python, parity reports, and worker execution all share one model.
- Avoid making OpenHCS sound like a generic workflow manager. The domain-specific parts are source dimensionality, image/object/measurement side channels, CellProfiler-compatible semantics, viewer outputs, and memory/backend-aware array execution.

## Comparison To MCMICRO

MCMICRO's strength is the very concrete pipeline spine:

- raw multiplexed tissue images,
- preprocessing/stitching/registration,
- segmentation,
- quantification,
- single-cell/spatial analysis outputs.

OpenHCS currently has the ingredients, but they are expressed more abstractly:

- import `.cppipe`,
- preserve loading semantics,
- stream step outputs,
- add Python functions,
- compare parity,
- benchmark speed and worker throughput.

Actionable improvement:

- Add one short "representative workflow" paragraph or figure panel that follows one imported CellProfiler pipeline from `.cppipe` plus acquisition folder through named masks, measurements, viewer inspection, optional Python QC, parity, and worker execution.
- Keep it generic enough to represent the benchmark corpus, but concrete enough that a biologist can picture what happens.

Suggested draft insertion near the start of Results:

> In a representative OpenHCS run, a `.cppipe` file and its encoded image-loading rules enter OpenHCS as a workflow; source images are resolved into well, site, channel, z-plane, and timepoint identities; processing modules produce named images, objects, measurements, and files; a selected mask can be streamed to napari or Fiji during execution; an assay-specific Python quality-control function can be inserted as another step; and the final outputs can be compared to native CellProfiler or executed across many wells with persistent workers.

## Comparison To BIOMERO

BIOMERO's strength is a clear OMERO-centered deployment story:

- OMERO stores image data,
- users launch analysis workflows from OMERO,
- workflows execute on HPC,
- outputs return to the managed image environment,
- FAIR workflow sharing and reduced user infrastructure burden are central.

OpenHCS overlaps in OMERO and scalable execution, but its center is different:

- OpenHCS is not an OMERO-HPC bridge.
- OMERO is one source/store integration.
- The OpenHCS unit is the workflow record that survives across GUI, generated Python, imported CellProfiler, viewer output, local/microscope/managed sources, and workers.

Actionable improvement:

- Keep the OMERO row in the positioning table, but add BIOMERO-style wording in Discussion that OpenHCS can use managed stores without making managed stores mandatory.
- Avoid letting the manuscript imply that OMERO is the main source model; microscope handlers, local files, imported `.cppipe` semantics, and TODO Bio-Formats-backed discovery all matter.
- Add the positive integration angle: BIOMERO-like systems can be framed as potential launch surfaces for OpenHCS workflows on SLURM/HPC rather than as mutually exclusive alternatives.

Suggested Discussion sentence:

> Unlike OMERO-centered execution bridges, OpenHCS does not require a managed image server as the organizing unit: OMERO can be a source, local and microscope-handler folders can be sources, imported CellProfiler loading rules can define sources, and all of these feed the same workflow record.

Optional stronger version:

> OMERO-centered systems can launch or organize analysis, but OpenHCS supplies the common bioimage execution model: the same workflow can be launched from a managed image environment, run from local acquisition folders, imported from CellProfiler, extended with Python callables, inspected in napari or Fiji, and executed on local workers or SLURM/HPC infrastructure.

## Comparison To Arkitekt And BioImageIT

Arkitekt and BioImageIT validate the broad integration/platform category. They also show a risk: platform papers can become abstract unless the user path is obvious.

OpenHCS does well here:

- "one workflow record" is plain enough,
- the draft no longer opens with tool lists,
- Results now has a proof chain.

Actionable improvement:

- Add a small "what remains connected" diagram or table in Figure 2: sources, parameters, functions, intermediates, viewers, generated Python, workers, outputs.
- Keep implementation names such as ObjectState, PolyStore, ArrayBridge, and ZMQRuntime out of the first two figures.

## Comparison To ilastik And BiaPy

ilastik and BiaPy are stronger than the current draft on explicit non-expert user benefit.

OpenHCS has relevant text, but it can be made more concrete:

- wet-lab user remains in GUI,
- computational collaborator reviews generated Python,
- viewer inspection is step-level,
- source mapping removes manual folder reorganization,
- custom Python is an extension path rather than plugin-development overhead.

Actionable improvement:

- Add one short sentence in Results or Discussion that says OpenHCS does not remove expert validation, but it moves routine interaction into GUI/viewer/workflow surfaces while preserving Python review.

Suggested sentence:

> OpenHCS does not remove the need to validate scientific methods, but it keeps routine interactions accessible: a wet-lab user can edit parameters and inspect masks in viewers, while a computational collaborator can review or modify the generated Python representation of the same workflow.

## Comparison To NanoPyx And Performance Papers

NanoPyx makes performance a direct contribution. OpenHCS has a strong speed claim, but it must remain carefully separated:

- CellProfiler parity first,
- single-thread CPU speed floor second,
- total wall time third,
- persistent-worker throughput fourth.

The current draft now does this well. Remaining action:

- Once final numbers are available, the abstract should include exact workflow count, minimum speedup, and maybe median speedup if strong.
- Figure 8 should visually separate minimum, distribution, and throughput; avoid one blended "speedup" graphic.

## Best Next Edits

1. Add the representative workflow paragraph near the start of Results.
2. Add a Discussion sentence distinguishing OpenHCS from OMERO-centered bridges like BIOMERO.
3. Add the accessibility/validation sentence connecting wet-lab GUI use and computational Python review.
4. Keep MCMICRO and BIOMERO in the comparator note and reference queue.
5. Consider adding MCMICRO/BIOMERO rows or footnotes to the positioning table only if the manuscript needs explicit direct comparison; otherwise avoid overloading the main text.
6. Add a short "common execution layer" paragraph to the Discussion that states the non-ad-hoc architecture in user-facing terms.
7. Update the positioning table to make integrations synergistic rather than competitive: CellProfiler pipelines run through OpenHCS; BIOMERO/OMERO can launch or provide sources; Python callables become UI-accessible workflow steps; napari/Fiji are viewer destinations.

## Do Not Change

- Do not move CellProfiler into the title.
- Do not make OMERO the center of the paper.
- Do not claim Bio-Formats automatic plate semantics before implementation and tests.
- Do not present OpenHCS as a fixed domain pipeline like MCMICRO; its contribution is workflow-state composability across tools and execution modes.
- Do not describe OpenHCS as merely glue, a wrapper, or a convenience UI. The important distinction is a single typed execution model for callables, arrays, source identities, side-channel records, viewers, outputs, and workers.
