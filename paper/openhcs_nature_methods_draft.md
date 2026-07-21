# OpenHCS: a composable bioimage workflow platform

**Working draft.** Public adoption, funding, citation, company-use statements, and final integration validation require source/code verification before submission.

## Abstract

High-throughput microscopy can produce biological images faster than they can be analyzed, inspected, and converted into reproducible evidence. OpenHCS is an open-source bioimage workflow platform that keeps source identity, parameters, intermediate results, viewer inspection, custom functions, and worker execution in one executable analysis. Imported CellProfiler pipelines provide reference analyses on biological image datasets. Across 30 workflows from official CellProfiler examples, tutorials, and benchmark supplements, OpenHCS reproduced the declared native output values with no unresolved differences. Among 29 workflows with completed native timing, every workflow executed at least 4.03-fold faster (median, 7.39-fold) under one-sample, one-thread/core, CPU-only conditions. GUI edits, generated Python, viewer outputs, CPU/GPU functions, and persistent workers operate on the same workflow record. OpenHCS preserves established biological analyses while reducing execution time for larger screens, repeated quality-control cycles, and method development.

## Introduction

Image analysis is where microscopy becomes biological evidence. A screen or time-course may begin with automated acquisition, but the experiment becomes interpretable only after cells, organelles, tissues, tracks, intensities, textures, morphologies, and quality-control failures have been identified and reviewed. As acquisition throughput increases, analysis and review often become the slowest and most fragile part of the workflow.

The bottleneck includes runtime, inspection, extension, provenance, and repeated execution. Researchers need to adapt an analysis when an assay changes, inspect intermediate masks when results look wrong, add assay-specific logic, rerun the same workflow across many samples, and explain which parameters produced a table or figure. These needs often span high-throughput batch tools, plugin systems, commercial workflows, open viewers, managed image stores, Python functions, and reproducible execution systems.

Fiji/ImageJ, CellProfiler, napari, OMERO, Zarr-backed image storage, Bio-Formats, scientific Python, GPU libraries, and workflow systems each solve distinct parts of biological image analysis and computational reproducibility [Schneider2012; Schindelin2012; Carpenter2006; McQuin2018; Sofroniew2019; Allan2012; Moore2021; BioFormats; vanDerWalt2014; Haase2020; Koster2012; DiTommaso2017; Galaxy2020; Galaxy2024]. The boundary problem appears when an analysis crosses those tools. Parameters are copied between GUIs and scripts, intermediate images are saved without clear provenance, viewers inspect masks that may not correspond to the current run, and custom code loses dimensional context such as channel, timepoint, z-plane, field, or well.

OpenHCS treats the workflow itself as the shared record. Images, source metadata, parameters, functions, intermediate masks, measurement tables, exported files, viewer streams, and worker execution are named parts of one workflow graph. Imported workflows remain editable and executable: they can be inspected, inherited, overridden, extended with Python, serialized as generated code, streamed to viewers, compared against references, and rerun through the same state model.

Before a run starts, OpenHCS checks which data exist, what each step consumes and produces, where outputs will go, which memory backend is required, and which worker will execute the work. Source schemas, function contracts, named runtime artifacts, storage backends, memory conversion, and process-isolated workers implement these checks.

CellProfiler equivalence provides the strictest test of this model. A `.cppipe` file contains image-loading rules, module settings, image names, object names, measurement expectations, display choices, and output behavior. OpenHCS compiles `.cppipe` files into editable workflows, compares their outputs against native CellProfiler, and reports speed only after equivalence is established. Native OpenHCS functions and backend-specific algorithms remain distinct methods unless they are evaluated against their own reference.

Validation spans the complete execution path from source discovery or import through inspection, extension, equivalence testing, and scaled execution. The validation corpus applies official CellProfiler workflows to biological image datasets spanning DNA damage, cell and tissue morphology, Cell Painting, translocation, wound healing, tracking, imaging flow cytometry, yeast screening, and worm phenotyping. The same workflow model supports step-level viewer output, generated Python, custom functions, managed sources, and persistent workers.

## Results

### One executable workflow retains analysis state across tools

Official CellProfiler `.cppipe` workflows enter OpenHCS with their image-loading semantics intact. Source images resolve to well, site, channel, z-plane, and timepoint identities. Processing steps produce named images, object labels, measurements, relationships, grids, and files. Selected outputs can be streamed to napari or Fiji during execution, extended with Python functions, compared against native CellProfiler references, and executed across wells with persistent workers (Figure 1).

OpenHCS records these operations as one workflow graph and one workflow state. A source binding, GUI edit, generated Python file, viewer request, imported CellProfiler module, custom Python function, or worker submission updates the same analysis. The workflow therefore retains provenance for image identity, parameter state, intermediate results, output destinations, viewer streams, and execution conditions.

Compile-time checks expose common setup failures before long runs. Image inputs must resolve to the intended well, site, channel, z-plane, and timepoint. Each imported module or Python function must receive the inputs it expects. CPU and backend requirements must be declared. Output policies must specify whether masks, measurements, files, viewer outputs, and comparison artifacts are kept, written, streamed, compared, or discarded.

Parameter state remains traceable. A threshold, channel name, output setting, or source mapping can come from an imported setting, a parent/default configuration, or a local override. Entering a value overrides the inherited value; clearing a value returns the field to inherited state. The GUI and generated Python represent the same workflow, so a domain expert can remain in the GUI while a computational collaborator reviews generated code or adds a quality-control function.

### CellProfiler output equivalence validates biological workflow preservation

Reference equivalence is the validation target for an imported analysis: execution may change, but the declared outputs used for biological interpretation must remain equivalent. A CellProfiler `.cppipe` file contains module settings, image names, object names, measurement expectations, display choices, and output behavior. OpenHCS parses the file, maps image-loading and metadata modules to image sources, maps processing modules to CellProfiler-compatible functions, and stores images, objects, measurements, relationships, grids, and files as named workflow results.

| CellProfiler concept | OpenHCS representation | Result |
|---|---|---|
| `.cppipe` file | Imported pipeline file | Official CellProfiler workflows enter OpenHCS without manual rewriting |
| Images, Metadata, NamesAndTypes | Image sources and image-name-to-file matching | Image identity is checked before execution |
| Processing module | OpenHCS workflow step with a CellProfiler-compatible callable | Module behavior runs through the normal analysis workflow |
| Images and objects | Named image and label outputs | Intermediates can be stored, streamed, compared, or reused |
| Measurements | Named measurement outputs | Tables remain tied to the workflow that produced them |
| Relationships and grids | Named non-image outputs | Object relationships and geometry are preserved explicitly |
| SaveImages and ExportToSpreadsheet | Output destinations and format writers | Files are produced from workflow results, not hidden side effects |
| Module settings | Traceable parameter state | Settings can be edited, inherited, and reviewed in the workflow record |

Preservation is defined by output equivalence under a declared comparison profile. The official-30 manifest uses value-output comparison: exported tables and database values are compared for every workflow, and image artifacts are compared when the native reference contains only images. A workflow passes only when the enabled comparisons report no unresolved differences. Speed is reported only for passing workflows (Figure 2).

The validation corpus contains 22 workflows and associated images from the official CellProfiler 3 example collection, seven workflows from the official CellProfiler tutorial collection, and one workflow from the CellProfiler 4 benchmark supplement [CellProfilerExamples; CellProfilerTutorials; Stirling2021]. These materials were created and distributed by the CellProfiler project and its contributors; OpenHCS uses them as external biological reference analyses. The assays include DNA-damage measurement, human and Drosophila cell morphology, tumor morphology, Cell Painting morphology and quality control, protein translocation, wound healing, time-lapse tracking, imaging flow cytometry, colocalization, positive-cell classification, yeast screening, and *C. elegans* phenotyping. The corresponding operations include segmentation, filtering, intensity and texture measurement, illumination correction, spatial relationships, tracking, image and table export, and specialized worm morphology. Each corpus row retains its official CellProfiler source, workflow, images, assay category, and original citation where available.

Native CellProfiler produced the reference outputs for each workflow. OpenHCS imported and ran the same `.cppipe` file on the same biological images, then compared corresponding output values under the declared equivalence policy. All 30 workflows reached an accuracy fraction of 1.0, with no unresolved differences under the enabled checks. The benchmark artifacts retain the comparison profile, pass table, module coverage, unsupported-feature status, speedup, and tolerated numerical differences.

The benchmark coverage table contained 30 supported `.cppipe` cases, 471 CellProfiler module instances, 58 unique CellProfiler module names, and 7,158 setting rows. All setting rows were mapped, and the benchmark reported no missing processing modules and no known-invalid absorbed processing modules. Across the current absorbed CellProfiler module catalogue, 53 modules are explicitly covered, 28 additional modules are covered by a shared semantic abstraction, and 8 modules remain outside the covered set. Version-specific parity artifacts record the CellProfiler and OpenHCS versions used for each comparison; new CellProfiler versions receive their own declared compatibility checks before output preservation or speed is reported for those versions.

### Reference-equivalent workflows execute at least fourfold faster

The primary performance benchmark constrains execution to one sample, one thread/core, CPU-only execution, and no batching. This condition separates the CellProfiler-compatible execution path from additional cores, GPU kernels, larger batch size, and amortized throughput over many wells.

The benchmark reports execution-only timing separately from total phase timing. Execution-only timing measures the part of the run most directly comparable to CellProfiler module execution. Total phase timing includes startup, imports, compilation, preparation, and execution, reflecting a one-off user run.

Twenty-nine workflows had completed native execution timings. Their median native execution time was 5.39 s, compared with 0.653 s for OpenHCS. Execution-only speedup was at least 4.03-fold for every one of these workflows, with a median of 7.39-fold. Total phase speedup was lower because startup and compilation are included; the median total phase speedup was 3.42-fold. The remaining wound-healing row equaled the configured 900-s native timeout ceiling without an explicit completion flag and is excluded from timing summaries pending a final benchmark rerun.

The single-thread CPU-only result establishes a performance floor for imported CellProfiler workflows. The measured reduction applies directly to repeated biological-analysis tasks such as screening additional wells, rerunning quality control, and iterating segmentation or measurement settings when execution dominates the analysis cycle. GPU-backed functions and backend-specific algorithms are supported as intentional OpenHCS methods, but they are not part of the CellProfiler-preserving speed result unless separately benchmarked against an appropriate reference.

### Sources, viewers, code, and Python methods remain attached to the run

Many microscopy groups organize data in managed stores such as OMERO. Others work from local disks, Zarr-backed stores, or acquisition folders exported by vendor acquisition software. OpenHCS treats those inputs as workflow sources. The workflow records which images are analyzed, how images map to biological and acquisition dimensions, and which outputs were produced from those images.

Microscope handlers convert acquisition-specific file layouts into analysis-ready image identities. ImageXpress-style exports can place timepoints and z-planes in nested folders rather than encoding every dimension in each filename. Opera Phenix exports can use acquisition-order field indices rather than spatial field order and can omit images when autofocus fails. OpenHCS builds an analysis-ready view of the plate, normalizes field labels, and fills missing Opera Phenix images with black placeholders where the handler can infer the expected grid. Bio-Formats-backed source discovery extends the same handler system to broadly readable microscopy datasets by labeling discovered images by series, channel, z-plane, and timepoint and requiring explicit user mapping when well or site labels cannot be inferred safely.

Viewer integration follows the named-output model. napari and Fiji outputs are enabled through step configuration. When napari or Fiji streaming is enabled for a step, OpenHCS launches or reuses the viewer on the configured port and streams that step's images while the pipeline runs. The same mask can be saved, compared against a reference, streamed to napari, or sent to Fiji because the mask remains a named workflow result.

OpenHCS exposes ordinary Python functions as workflow steps with the same parameter editing, validation, viewer output, generated code, and worker execution used by imported modules. Function signatures expose editable parameters. Memory-backend declarations record whether a function expects arrays from a NumPy-like CPU library, a GPU array library, or a deep-learning framework. An imported CellProfiler workflow can therefore be extended with an assay-specific quality-control function, and a native OpenHCS workflow can mix CPU image processing, GPU-compatible array operations, and table-producing functions when those choices are appropriate for the analysis.

Each integration preserves a distinct role in the biological analysis and has a corresponding evidence boundary.

| Existing tool or platform | Preserved role | OpenHCS integration | Reported evidence |
|---|---|---|---|
| CellProfiler | Established modular bioimage analysis and `.cppipe` workflows | CellProfiler pipelines run through OpenHCS with loading semantics preserved and named results exposed | 30-workflow value-output equivalence corpus; speed reported only after equivalence |
| Fiji/ImageJ | Familiar inspection and image-processing environment | Fiji receives selected step outputs as named workflow results | Implemented viewer destination; GUI/runtime environment requirements reported separately |
| napari | Python-native multidimensional image inspection | napari receives selected step outputs while workflow identity and output names remain attached | Implemented viewer destination with available automated viewer path |
| OMERO and BIOMERO-style systems | Managed image storage, collaboration, and launch surfaces | OMERO can provide workflow sources and managed environments can launch OpenHCS workflows | OMERO support tested across multiple versions; launch-surface support should cite exact tested versions |
| Bio-Formats | Broad pixel and metadata readability | Bio-Formats-backed discovery labels images where metadata support safe inference and requires explicit mapping when metadata are ambiguous | Source discovery uses the same normalized image-label model as vendor handlers |
| Microscope vendor folders | Acquisition-specific plate layouts and metadata conventions | Vendor handlers normalize wells, fields, channels, z-planes, timepoints, and missing-image behavior before analysis starts | ImageXpress and Opera Phenix quirks are handled explicitly |
| Python, GPU, and deep-learning libraries | Assay-specific operations and backend-specific methods | Python callables become workflow steps with declared parameters, memory expectations, outputs, viewer policies, and worker execution | Integrated as native methods; CellProfiler equivalence requires separate validation |
| Generic workflow managers | Scalable and reproducible command execution | OpenHCS supplies bioimage-specific source identity, named outputs, viewer destinations, and parity artifacts inside an executable workflow | Worker benchmarks report throughput, queue depth, and RAM tradeoffs |

### Many-well throughput uses persistent workers

OpenHCS is also evaluated in the way high-content screening is often used: many samples, persistent worker processes, and enough RAM to keep workers alive across repeated work. In that setting, import cost, compile cost, and backend warmup are paid once and divided across many samples rather than paid again for every well.

OpenHCS throughput was measured directly in replicated-well runs over all 30 workflows. Comparisons to CellProfiler were projected by multiplying each workflow's measured native single-sample execution time by the number of wells; native CellProfiler was not rerun as a matched multi-process workload. The projected summaries exclude the unresolved wound-healing timing. Across the remaining 29 workflows, minimum and median execution speedups were 7.22-fold and 9.88-fold with two cores and four wells per core, 10.8-fold and 14.2-fold with three cores and four wells per core, and 12.3-fold and 16.7-fold with four cores and eight wells per core. At four cores, increasing queue depth from one to eight wells per core raised median projected execution speedup from 12.8-fold to 16.7-fold.

Scaling is reported per resource, not only per wall-clock speedup. Each worker adds CPU capacity but also memory pressure because it may hold imported libraries, compiled kernels, open stores, cached arrays, and intermediate outputs. In the four-core queue-depth sweep, median peak RAM increased from 3.98 GB at one well per core to 4.25 GB at eight wells per core, with maximum peak RAM reaching 14.6 GB.

## Discussion

OpenHCS preserves existing bioimage workflows by keeping sources, parameters, intermediate outputs, viewers, generated code, and worker execution in one provenance-tracked workflow. Thirty official CellProfiler workflows applied to their associated biological image data supply the strongest preservation test. Imported `.cppipe` workflows reproduce native CellProfiler outputs under declared equivalence checks and retain named results that can be inspected, extended, serialized, streamed to viewers, and executed through workers.

Compatibility also creates an extension path. A preserved CellProfiler workflow can be modified with an ordinary Python function, routed to napari or Fiji for step-level inspection, reviewed as generated Python, or run across persistent workers. Native OpenHCS workflows can use the same source, output, viewer, function, and worker mechanisms without requiring a CellProfiler origin. Backend-specific GPU or deep-learning methods remain distinct scientific methods unless a separate parity or validation target is defined for them.

The equivalence result validates OpenHCS as an execution environment for the 30 biological analyses. Native CellProfiler supplies the version-matched reference behavior, while the official workflows and images supply the biological assay context. Assay-specific biological validation remains attached to the originating workflow rather than being redefined by the execution platform.

The measured speedup makes preservation consequential for experiment scale. Screening, longitudinal imaging, parameter refinement, and quality-control reruns repeatedly apply the same analysis to additional images. Reducing execution time without changing the declared outputs increases the analysis that can be completed within a fixed computational window when execution is the limiting phase.

The CellProfiler importer covers the module and setting subset represented in the equivalence-tested corpus. Modules outside that set require explicit validation before preservation is reported. Bio-Formats improves readability, while ambiguous well, site, channel, z-plane, or timepoint identity requires explicit source mapping. Viewer integration depends on GUI/runtime environment availability. Persistent workers add process-lifecycle, memory, restart, and cache-warmth tradeoffs. Performance results are therefore separated into execution-only time, total wall time, single-thread CPU speed, measured OpenHCS throughput, and projected comparison to serial native CellProfiler execution.

Output equivalence and constrained execution timing establish a preservation-plus-performance baseline. Under one-sample, one-thread/core, CPU-only conditions, all 29 workflows with completed native timing executed at least 4.03-fold faster after equivalence was established. Persistent workers amortized startup and compilation costs in many-sample runs while exposing throughput and RAM tradeoffs directly. OpenHCS provides a migration path for retaining established CellProfiler analyses while adding provenance, viewer inspection, custom functions, generated code, and scalable execution.

## Online Methods

### Pipeline object model and compilation

OpenHCS pipelines are directed workflow graphs built from source definitions, parameter state, workflow steps, and output policies. Before execution, the pipeline is compiled into runtime contexts. Compilation resolves input sources, validates function requirements, determines intermediate and final outputs, assigns runtime paths, and prepares callable execution hooks. The compiled form is the unit submitted to workers.

### Source schema and source binding

Source schemas describe the experimental identity of input data, including dimensions such as well, site, channel, timepoint, and z-plane when present. Source bindings connect those named workflow sources to local files, managed stores, or virtual sources. This lets analysis code refer to the intended image role rather than a one-off path string.

### Microscope handlers and Bio-Formats source discovery

Microscope handlers convert acquisition-specific file layouts into analysis-ready image identities. Vendor-specific handlers encode known quirks such as nested timepoint and z-plane folders, field remapping, metadata sidecars, pixel-size lookup, channel names, and missing-image policy. The Bio-Formats handler uses Bio-Formats metadata to discover series, channel, z-plane, and timepoint dimensions for broadly supported datasets, then emits the same normalized image labels used by vendor-specific handlers. When Bio-Formats cannot infer well or site labels, OpenHCS requires an explicit source schema rather than silently constructing ambiguous image sets.

### Function registration and memory-backend requirements

OpenHCS functions declare their memory interface through backend annotations for NumPy-like CPU arrays, GPU array libraries, OpenCL-based image-processing libraries, or deep-learning frameworks. These declarations record the expected input and output memory types. During execution, OpenHCS converts image payloads between compatible backends when required. Function signatures expose parameters in the GUI and generated Python representation.

### Runtime outputs and materialization

OpenHCS treats intermediate images, labels, measurements, relationships, grids, and files as named workflow outputs. Output policies determine whether a value is kept in memory, written to disk, stored in Zarr, streamed to napari, streamed to Fiji, compared against a reference, or discarded after execution.

### Viewer and storage backends

Viewer and storage integrations are treated as destinations for workflow outputs or sources for workflow inputs. napari and Fiji outputs receive named workflow results. OMERO, local disk, memory, and Zarr-backed stores participate in source and output handling according to the configured runtime path.

### Worker execution

OpenHCS uses process-level worker execution for isolation, progress reporting, viewer separation, and throughput scaling. Persistent workers can amortize startup, imports, compilation, and warmup across repeated samples. Worker-level benchmarks report wall time, execution time, sample count, worker count, and memory use.

### CellProfiler import

CellProfiler `.cppipe` files are parsed into module blocks and settings. Infrastructure modules are mapped into source schemas and source bindings. Processing modules resolve to CellProfiler-compatible functions. Each resolved module becomes an OpenHCS workflow step with a runtime adapter that provides access to CellProfiler-style images, object labels, measurements, grids, and relationships while storing outputs as OpenHCS workflow results.

### Parity comparison

Native CellProfiler is run for each benchmark workflow to generate reference outputs. OpenHCS then imports and runs the same `.cppipe` file. The official-30 manifest enables value-output comparison. Exported tables and database values are compared for every workflow; image outputs are additionally compared when the native reference profile contains only images. Numeric comparisons use declared absolute and relative tolerances. Non-numeric identifiers and categorical values are compared exactly unless a specific CellProfiler-compatible normalization is documented. Each equivalence report records the enabled artifact classes, CellProfiler version or commit, and OpenHCS commit.

### Performance benchmarking

The primary benchmark condition uses one sample, one thread/core, CPU-only execution, and no batching. Timing is reported as execution-only time and total wall time. Execution-only time isolates module/runtime execution after startup and compile overhead. Total wall time includes startup, imports, compile, preparation, and execution. The wound-healing native duration equals the configured 900-s timeout ceiling, while the summary row lacks an explicit completion flag. This row is excluded from timing statistics pending a final rerun with explicit completion or censoring metadata. Throughput benchmarks are reported separately using multi-sample execution and worker-level parallelism. GPU-backed functions are reported only as OpenHCS method support unless a separate benchmark defines the GPU hardware, backend version, reference behavior, and comparison policy.

The benchmark runs use local or explicitly mounted paths recorded in the benchmark manifest. Cloud-bursting or network-filesystem-latency performance requires a benchmark that records the storage environment, path form, cache state, and worker placement.

### Benchmark corpus

The corpus contains 22 workflows and associated image sets from the official CellProfiler 3 examples repository, seven workflows and image sets from the official CellProfiler tutorials repository, and one workflow from the supplement to the CellProfiler 4 performance study [CellProfilerExamples; CellProfilerTutorials; Stirling2021]. The CellProfiler project and the cited dataset contributors retain authorship and provenance for these materials. The corpus table lists each workflow, official source URL, immutable source revision for the final benchmark release, original data citation where available, assay family, image-analysis behavior, output files, module coverage, equivalence status, single-thread/core speedup, and throughput status.

The official CellProfiler examples source is pinned in the benchmark manifest to commit `4972b59e670a4ae96c3d453803c92eeff378d054`. **Submission gate:** pin the exact CellProfiler tutorials and CellProfiler 4 supplement revisions used for the final benchmark run, then replace this sentence with the complete three-source revision record.

### Reproducibility package

The benchmark runner emits the OpenHCS commit, CellProfiler version or commit, Python version, operating system, CPU model, GPU model if used, RAM, pipeline manifest, dataset manifest, checksums where available, native CellProfiler timing, OpenHCS timing, parity reports, and raw CSVs used to generate figures. Benchmark figures are regenerated from saved CSV files without rerunning CellProfiler.

## Draft Figure Captions

### Figure 1. Sources, tools, viewers, code, and workers remain one workflow

Automated microscopy produces wells, sites, channels, z-planes, and timepoints that become masks, measurements, quality-control decisions, reruns, and output tables. OpenHCS keeps source identities, parameters, named outputs, viewer destinations, generated code, and worker execution attached to one runnable workflow. Diagram source: `paper/figures/rendered/fig01_field_integration_gap.svg`.

### Figure 2. Official CellProfiler biological workflows are preserved and accelerated

Representative official CellProfiler biological images, native outputs, OpenHCS outputs, and difference views make the preservation target visible. The examples include at least one cellular assay and one morphologically distinct assay such as wound healing, yeast, or worm phenotyping. These image panels require selected workflows to be rerun with image-output comparison enabled; they do not infer image equivalence from the value-output corpus result. The benchmark manifest feeds native CellProfiler and OpenHCS runs. OpenHCS parses `.cppipe` files into image sources, image-name-to-file matching, CellProfiler-compatible workflow steps, named outputs, and saved results. Quantitative panels separate output equivalence, module coverage, constrained one-sample CPU-only speedup, cold-run overhead, measured OpenHCS throughput, projected native comparison, and RAM scaling. Every image panel credits the official CellProfiler source and original dataset citation. Diagram sources: `paper/figures/rendered/fig03_cellprofiler_import_path.svg` and `paper/figures/rendered/fig04_benchmark_validation_structure.svg`.

### Figure 3. GUI edits, generated Python, and Python functions modify the same workflow

The pipeline editor, generated Python representation, source binding, step-level napari/Fiji output, ordinary Python functions, backend-specific functions, and worker execution all target the workflow being edited and run. Diagram sources: `paper/figures/rendered/fig07_typed_state_bidirectional_editing.svg` and `paper/figures/rendered/fig06_backend_extensibility.svg`.

## Supplementary Table Captions

### Supplementary Table 1. Benchmark corpus

Each row lists one `.cppipe` workflow, official CellProfiler source collection, immutable source revision, source URL, original dataset citation where available, assay family, image-analysis behavior, output files, CellProfiler modules, native CellProfiler runtime or timeout status, OpenHCS execution runtime, total OpenHCS runtime, speedup, equivalence status, and notes. Equivalence, timing, and timeout status remain separate columns.

### Supplementary Table 2. CellProfiler module coverage

Each row lists one CellProfiler module class, import status, explicitly parity-tested status, theoretically covered status for modules sharing implemented abstractions, not-covered status, accelerated path where relevant, backend used, unsupported settings or features, and notes.

### Supplementary Table 3. Worker/RAM scaling

Each row lists a throughput condition, pipeline group, sample count, worker count, core count, peak RAM, RAM per worker, total wall time, samples per hour, speedup versus native CellProfiler, and speedup versus OpenHCS single-worker execution.

## Code and Data Availability

OpenHCS source code: `https://github.com/OpenHCSDev/OpenHCS`.

OpenHCS documentation: `https://openhcs.readthedocs.io/`.

Benchmark scripts, manifests, raw timing CSVs, parity reports, figure-generation scripts, and generated figures: `https://github.com/OpenHCSDev/openhcs`.

The benchmark uses biological images and pipelines distributed by the CellProfiler project rather than OpenHCS-authored benchmark data. Original sources:

- official CellProfiler example pipelines and images: `https://github.com/CellProfiler/examples` [CellProfilerExamples]
- official CellProfiler tutorial pipelines and images: `https://github.com/CellProfiler/tutorials` [CellProfilerTutorials]
- CellProfiler 4 benchmark supplement: `https://github.com/carpenterlab/2021_Stirling_BMCBioInformatics` [Stirling2021]

The OpenHCS benchmark manifest records acquisition paths and maps every workflow to its original source. The final archived benchmark release will record immutable revisions, checksums, source licenses, and original dataset citations for all three source collections.

Reusable libraries:

- ObjectState: `https://github.com/OpenHCSDev/objectstate`
- ArrayBridge: `https://github.com/OpenHCSDev/arraybridge`
- PolyStore: `https://github.com/OpenHCSDev/PolyStore`
- ZMQRuntime: `https://github.com/OpenHCSDev/zmqruntime`
- pyqt-reactive: `https://github.com/OpenHCSDev/pyqt-reactive`
- pycodify: `https://github.com/OpenHCSDev/pycodify`
- python-introspect: `https://github.com/OpenHCSDev/python-introspect`
- metaclass-registry: `https://github.com/OpenHCSDev/metaclass-registry`

## Acknowledgements

We thank the CellProfiler project, its contributors, and the authors of the underlying biological datasets for making the example, tutorial, and benchmark materials available. The original CellProfiler sources and dataset-specific citations will accompany every benchmark manifest row and figure panel.

## References To Format

- `[Carpenter2006]` Carpenter et al. CellProfiler: image analysis software for identifying and quantifying cell phenotypes. Genome Biology 7, R100 (2006). DOI: 10.1186/gb-2006-7-10-r100.
- `[McQuin2018]` McQuin et al. CellProfiler 3.0: next-generation image processing for biology. PLoS Biology 16, e2005970 (2018). DOI: 10.1371/journal.pbio.2005970.
- `[Stirling2021]` Stirling et al. CellProfiler 4: improvements in speed, utility and usability. BMC Bioinformatics 22, 433 (2021). DOI: 10.1186/s12859-021-04344-9.
- `[CellProfilerExamples]` CellProfiler contributors. CellProfiler example pipelines and associated biological images. `https://github.com/CellProfiler/examples`. Cite the immutable revision used in the final benchmark release.
- `[CellProfilerTutorials]` CellProfiler contributors. CellProfiler tutorial pipelines and associated biological images. `https://github.com/CellProfiler/tutorials`. Cite the immutable revision used in the final benchmark release.
- `[Schneider2012]` Schneider, Rasband and Eliceiri. NIH Image to ImageJ: 25 years of image analysis. Nature Methods 9, 671-675 (2012). DOI: 10.1038/nmeth.2089.
- `[Schindelin2012]` Schindelin et al. Fiji: an open-source platform for biological-image analysis. Nature Methods 9, 676-682 (2012). DOI: 10.1038/nmeth.2019.
- `[Allan2012]` Allan et al. OMERO: flexible, model-driven data management for experimental biology. Nature Methods 9, 245-253 (2012). DOI: 10.1038/nmeth.1896.
- `[Sofroniew2019]` napari contributors. napari: a multi-dimensional image viewer for Python. Zenodo (2019). DOI: 10.5281/zenodo.3555620.
- `[vanDerWalt2014]` van der Walt et al. scikit-image: image processing in Python. PeerJ 2, e453 (2014). DOI: 10.7717/peerj.453.
- `[Moore2021]` Moore et al. OME-NGFF: a next-generation file format for expanding bioimaging data-access strategies. Nature Methods 18, 1496-1498 (2021). DOI: 10.1038/s41592-021-01326-w.
- `[BioFormats]` Linkert et al. Metadata matters: access to image data in the real world. Journal of Cell Biology 189, 777-782 (2010). DOI: 10.1083/jcb.201004104.
- `[MCMICRO]` Schapiro et al. MCMICRO: a scalable, modular image-processing pipeline for multiplexed tissue imaging. Nature Methods 19, 311-315 (2022). DOI: 10.1038/s41592-021-01308-y.
- `[BIOMERO]` Balaz et al. BIOMERO: a scalable and extensible image analysis framework. PLOS Computational Biology 19, e1011369 (2023). DOI: 10.1371/journal.pcbi.1011369.
- `[Koster2012]` Koester and Rahmann. Snakemake: a scalable bioinformatics workflow engine. Bioinformatics 28, 2520-2522 (2012). DOI: 10.1093/bioinformatics/bts480.
- `[DiTommaso2017]` Di Tommaso et al. Nextflow enables reproducible computational workflows. Nature Biotechnology 35, 316-319 (2017). DOI: 10.1038/nbt.3820.
- `[Galaxy2020]` The Galaxy Community. The Galaxy platform for accessible, reproducible and collaborative biomedical analyses: 2020 update. Nucleic Acids Research 48, 8205-8207 (2020). DOI: 10.1093/nar/gkaa554.
- `[Galaxy2024]` Abueg et al. The Galaxy platform for accessible, reproducible, and collaborative data analyses: 2024 update. Nucleic Acids Research 52, W83-W94 (2024). DOI: 10.1093/nar/gkae410.
- `[Haase2020]` Haase et al. CLIJ: GPU-accelerated image processing for everyone. Nature Methods 17, 5-6 (2020). DOI: 10.1038/s41592-019-0650-1.
- `[CuPy]` Okuta et al. CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations. Proceedings of Workshop on Machine Learning Systems at NeurIPS (2017). DOI: 10.25080/shinma-7f4c6e7-00e.
- `[CuCIM]` Lee et al. cuCIM: a GPU image I/O and processing library. Zenodo (2021). DOI: 10.25080/majora-1b6fd038-022.
- `[JAX]` Bradbury et al. JAX: composable transformations of Python+NumPy programs (2018).
- `[PyTorch]` Paszke et al. PyTorch: an imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems 32, 8024-8035 (2019).
- `[TensorFlow]` Abadi et al. TensorFlow: a system for large-scale machine learning. 12th USENIX Symposium on Operating Systems Design and Implementation, 265-283 (2016).
- `[ObjectState]` Simas. ObjectState: generic lazy dataclass configuration framework with dual-axis inheritance and contextvars-based resolution. Software. `https://github.com/OpenHCSDev/objectstate`.
- `[ArrayBridge]` Simas. ArrayBridge: unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto with automatic memory type conversion. Software. `https://github.com/OpenHCSDev/arraybridge`.
- `[PolyStore]` Simas. PolyStore: framework-agnostic multi-backend storage abstraction for ML and scientific computing. Software. `https://github.com/OpenHCSDev/PolyStore`.
- `[ZMQRuntime]` Simas. ZMQRuntime: generic ZMQ-based distributed execution framework. Software. `https://github.com/OpenHCSDev/zmqruntime`.
- `[pyqtReactive]` Simas. pyqt-reactive: reactive form generation framework for PyQt6. Software. `https://github.com/OpenHCSDev/pyqt-reactive`.
- `[pycodify]` Simas. pycodify: Python source code as a serialization format with automatic import resolution. Software. `https://github.com/OpenHCSDev/pycodify`.
- `[pythonIntrospect]` Simas. python-introspect: pure Python introspection toolkit for function signatures, dataclasses, and type hints. Software. `https://github.com/OpenHCSDev/python-introspect`.
- `[metaclassRegistry]` Simas. metaclass-registry: metaclass-driven plugin registry system with lazy discovery and caching. Software. `https://github.com/OpenHCSDev/metaclass-registry`.
