# Figure Plan

**Target:** three main figures plus focused supplementary figures and tables.

The earlier eight-figure plan was too granular for this manuscript. The main paper should not spend separate figures on checks, source bindings, Python functions, viewers, validation structure, and throughput when those are pieces of one platform story. The revised plan uses three high-density figures: one conceptual problem/solution figure, one quantitative validation figure, and one platform/UI/integration figure.

Editable diagram drafts are in `paper/figures/diagrams/*.dot`; rendered SVG/PNG drafts are in `paper/figures/rendered/`. Rebuild all diagrams and the contact sheet with `python paper/figures/render_diagrams.py`.

## Figure 1. Scaled Acquisition Should Not Make Analysis The Bottleneck

This should be one large conceptual figure, not two separate figures. The left side shows automated microscopy producing many wells, sites, channels, z-planes, and timepoints. The middle shows the analysis burden: segmentation masks, measurements, quality-control decisions, review, reruns, output tables, and collaborator handoff. The failure mode is fragmentation: copied parameters, exported files, detached viewers, custom scripts, batch jobs, and managed image stores that no longer clearly refer to the same analysis.

The right side shows OpenHCS as the way those pieces stay connected. CellProfiler, Fiji/ImageJ, napari, OMERO or BIOMERO-style launch surfaces, Zarr/local/microscope-handler sources, Python functions, GPU/deep-learning methods, generated Python, output tables, and workers should appear around one OpenHCS workflow object.

**Main message:** OpenHCS lets labs keep familiar bioimage tools while preventing analysis and review from becoming disconnected bottlenecks after acquisition.

Suggested panels:

- A. Acquisition scale creates many images and review tasks.
- B. Typical analysis fragments across tools and files.
- C. OpenHCS keeps sources, workflows, viewers, functions, outputs, and workers attached to one analysis.

## Figure 2. CellProfiler Biological-Workflow Preservation And Speed

This should combine biological examples, validation, and performance in one main quantitative figure. The reader should see representative official CellProfiler images and outputs before the aggregate metrics, and should see that performance is reported only after preservation is established.

The figure starts with `.cppipe` import: loading and metadata modules become source mappings, processing modules become CellProfiler-compatible workflow steps, and images, objects, measurements, relationships, grids, and saved files become named outputs. Native CellProfiler provides the reference run. OpenHCS produces the comparison run.

The quantitative panels should use the current `official30_well_throughput` benchmark figures as provisional data sources.

The current presentation CSVs include a wound-healing native duration equal to the configured 900-s timeout ceiling without an explicit completion flag. Regenerate aggregate timing and projected-throughput panels from the 29 completed native timings until that case is rerun with explicit status metadata. The 30-workflow output-equivalence panel remains unchanged.

Suggested panels:

- A. Representative official CellProfiler biological source images spanning at least one cellular assay and one morphologically distinct assay such as wound healing, yeast, or worm phenotyping.
- B. Native CellProfiler outputs, OpenHCS outputs, and difference views for the representative images. Rerun these selected workflows with image-output comparison enabled; do not infer image equivalence from the value-output corpus result.
- C. `.cppipe` import and native CellProfiler equivalence comparison pipeline.
- D. Output equivalence across the 30-workflow corpus.
- E. Single-sample, one-thread/core, CPU-only speedup distribution showing every tested workflow at least 4x faster; timeout-censored cases must be marked rather than plotted as completed native runtimes.
- F. Module coverage: corpus-exercised processing modules, source/infrastructure modules, additional registered processing modules outside the corpus, and any missing or known-invalid absorbed modules.
- G. Persistent-worker throughput, labeled as measured OpenHCS throughput with projected comparison to serial native CellProfiler execution.
- H. RAM or worker-scaling tradeoff, likely supplementary if space is tight.

Every biological image and workflow panel must credit the official CellProfiler source collection and the original dataset citation where available. The corpus comprises 22 official CellProfiler 3 examples, seven official CellProfiler tutorial workflows, and one CellProfiler 4 benchmark-supplement workflow.

**Main message:** OpenHCS preserves established CellProfiler analyses on biological image data and reduces their execution time under declared benchmark conditions.

## Figure 3. One Editable Workflow Across GUI, Code, Viewers, Functions, And Workers

This is the platform mega-figure. It should use cleaned-up screenshots or schematic screenshots based on the current GUI, not raw development screenshots. The figure should make the system tangible for non-technical readers: this is what it means for one analysis to remain editable, extensible, inspectable, and executable.

Suggested panels:

- A. Pipeline editor showing named function steps, grouping/source hints, and viewer/materialization badges.
- B. Step editor beside generated Python showing that GUI edits and code describe the same `FunctionStep`.
- C. Function-pattern editor showing per-invocation behavior and parameters for a callable.
- D. Source/microscope-handler binding panel showing local, managed, vendor folder, and Bio-Formats-backed source paths.
- E. Viewer/output panel showing a selected intermediate mask streamed to napari or Fiji and/or exported as CellProfiler/CellProfiler Analyst-compatible results.
- F. Function/backend panel showing ordinary Python, NumPy/Numba, CuPy/CuCIM, JAX, PyTorch, TensorFlow, and pyclesperanto methods entering as configured workflow steps.
- G. Worker panel showing the same workflow running through persistent workers without changing the analysis.

**Main message:** OpenHCS is not a closed plugin catalog or a loose collection of scripts. GUI editing, generated Python, source binding, viewer output, GPU/deep-learning callables, CellProfiler-compatible outputs, and worker execution are all surfaces of the same workflow.

## Supplementary Figures

Use supplement for details that are real but too narrow for main figures.

- Supplementary Figure 1. Pre-run checks: sources, dimensions, source bindings, function contracts, memory backends, outputs, and worker execution.
- Supplementary Figure 2. Microscope handlers: ImageXpress, Opera Phenix, local/Zarr/OMERO source identities, and Bio-Formats-backed source discovery.
- Supplementary Figure 3. GUI/code/provenance details: inherited/default/local values, dirty state, generated Python, re-import, and execution.
- Supplementary Figure 4. Extended benchmark diagnostics: cold-run overhead, execution-only timing, RAM per worker, and per-pipeline outliers.

## Supplementary Tables

## Supplementary Table 1. Benchmark Corpus

Each row should list one `.cppipe` workflow, official CellProfiler source collection, immutable source revision, source URL, original dataset citation where available, assay family, semantic pressure, output pressure, CellProfiler modules used, native CellProfiler runtime or timeout status, OpenHCS execution runtime, total OpenHCS runtime, speedup, equivalence status, and notes. Equivalence, timing, and timeout status should remain separate.

## Supplementary Table 2. CellProfiler Module Coverage

Each row should list one registered CellProfiler module class, whether the benchmark corpus exercises it, its source/infrastructure or processing role, importability, declared contract and backend, unsupported settings or features, and notes. Processing modules outside the corpus should remain explicitly untested rather than being inferred covered from family similarity.

## Supplementary Table 3. Worker/RAM Scaling

Each row should list a throughput condition, pipeline group, sample count, worker count, core count, peak RAM, RAM per worker, total wall time, samples per hour, speedup versus native CellProfiler, and speedup versus OpenHCS single-worker execution.

## Next Iteration

- Collapse existing DOT drafts into three composite figure specs instead of maintaining eight independent main figures.
- Keep Figure 1 accessible and non-technical.
- Keep Figure 2 quantitative and reviewer-defensible.
- Build Figure 3 from cleaned screenshots/schematics rather than raw GUI captures.
- Move narrow mechanism details into supplement unless they are needed to understand the main claim.
