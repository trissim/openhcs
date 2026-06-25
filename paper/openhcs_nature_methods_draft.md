# OpenHCS: a composable bioimage workflow platform

**Working draft.** Public adoption, funding, citation, company-use statements, and final integration validation require source/code verification before submission.

## Abstract

Image-based biology increasingly depends on experiments whose acquisition can outpace their analysis, review, and provenance tracking. Existing bioimage tools are valuable, but workflows become fragile when image sources, parameters, intermediate masks, viewer inspection, custom logic, exported tables, and batch execution are split across disconnected programs and files.

OpenHCS is an open-source composable bioimage workflow platform for keeping those pieces in one runnable analysis. OpenHCS maps microscope plate folders and Bio-Formats-readable datasets into analysis-ready source identities, imports trusted CellProfiler `.cppipe` pipelines with their loading semantics intact, exposes ordinary CPU, GPU, and deep-learning Python functions as workflow steps, routes selected outputs to napari or Fiji, and executes the same workflow across persistent workers. A source binding, GUI edit, generated Python file, viewer output, imported CellProfiler module, custom Python function, or worker run updates the same workflow record rather than a separate copy of the experiment.

CellProfiler import provides a stringent preservation test because `.cppipe` files encode established biological image-analysis workflows. Across 30 benchmark workflows, OpenHCS reproduced native CellProfiler outputs under declared parity checks and achieved at least 4.03x execution-only speedup for every workflow under one-sample, one-thread/core, CPU-only conditions. OpenHCS therefore provides a backwards-compatible route from existing analyses to inspectable, extensible, provenance-tracked workflows that are faster under the tested benchmark conditions.

## Introduction

Image analysis is where microscopy becomes biological evidence. A screen or time-course may begin with automated acquisition, but the experiment becomes interpretable only after cells, organelles, tissues, tracks, intensities, textures, morphologies, and quality-control failures have been identified and reviewed. As acquisition throughput increases, analysis and review often become the slowest and most fragile part of the workflow.

The bottleneck includes runtime, inspection, extension, provenance, and repeated execution. Researchers need to adapt an analysis when an assay changes, inspect intermediate masks when results look wrong, add assay-specific logic, rerun the same workflow across many samples, and explain which parameters produced a table or figure. These needs often span high-throughput batch tools, plugin systems, commercial workflows, open viewers, managed image stores, Python functions, and reproducible execution systems.

Fiji/ImageJ, CellProfiler, napari, OMERO, Zarr-backed image storage, Bio-Formats, scientific Python, GPU libraries, and workflow systems each solve distinct parts of biological image analysis and computational reproducibility [Schneider2012; Schindelin2012; Carpenter2006; McQuin2018; Sofroniew2019; Allan2012; Moore2021; BioFormats; vanDerWalt2014; Haase2020; Koster2012; DiTommaso2017; Galaxy2020; Galaxy2024]. The boundary problem appears when an analysis crosses those tools. Parameters are copied between GUIs and scripts, intermediate images are saved without clear provenance, viewers inspect masks that may not correspond to the current run, and custom code loses dimensional context such as channel, timepoint, z-plane, field, or well.

OpenHCS treats the workflow itself as the shared record. Images, source metadata, parameters, functions, intermediate masks, measurement tables, exported files, viewer streams, and worker execution are named parts of one workflow graph. Imported workflows remain editable and executable: they can be inspected, inherited, overridden, extended with Python, serialized as generated code, streamed to viewers, compared against references, and rerun through the same state model.

Internally, OpenHCS uses source schemas, function contracts, runtime artifacts, storage backends, memory-backend conversion, and process-isolated workers. These mechanisms support a direct user-facing guarantee. Before a run starts, OpenHCS checks which data exist, what each step consumes, what each step produces, where outputs will go, which memory backend is required, and which worker will execute the work.

CellProfiler compatibility is the strictest test of this model. A `.cppipe` file contains image-loading rules, module settings, image names, object names, measurement expectations, display choices, and output behavior. OpenHCS compiles `.cppipe` files into normal OpenHCS workflows, compares outputs against native CellProfiler, and reports speed only after parity is established. CellProfiler parity is the preservation test for imported CellProfiler workflows. Native OpenHCS functions and backend-specific algorithms can be used when a lab intentionally chooses a different analysis method.

The evidence reported here follows one workflow path from import or source discovery through inspection, extension, parity testing, and scaled execution. OpenHCS imports a broad CellProfiler workflow corpus with loading semantics intact, reproduces native outputs under output-parity checks, measures constrained CPU-only speedups, and demonstrates how the same workflow model supports step-level viewer output, generated Python, custom functions, managed sources, and persistent workers.

## Results

### OpenHCS keeps bioimage analysis as one runnable workflow

OpenHCS was exercised as an integrated workflow path rather than as separate import, inspection, scripting, and benchmarking utilities. Real CellProfiler `.cppipe` workflows enter OpenHCS with their image-loading semantics intact. Source images resolve to well, site, channel, z-plane, and timepoint identities. Processing steps produce named images, object labels, measurements, relationships, grids, and files. Selected outputs can be streamed to napari or Fiji during execution, extended with Python functions, compared against native CellProfiler references, and executed across wells with persistent workers (Figure 1).

OpenHCS records these operations as one workflow graph and one workflow state. A source binding, GUI edit, generated Python file, viewer request, imported CellProfiler module, custom Python function, or worker submission updates the same analysis. The workflow therefore retains provenance for image identity, parameter state, intermediate results, output destinations, viewer streams, and execution conditions.

Compile-time checks expose common setup failures before long runs. Image inputs must resolve to the intended well, site, channel, z-plane, and timepoint. Each imported module or Python function must receive the inputs it expects. CPU and backend requirements must be declared. Output policies must specify whether masks, measurements, files, viewer outputs, and comparison artifacts are kept, written, streamed, compared, or discarded.

Parameter state remains traceable. A threshold, channel name, output setting, or source mapping can come from an imported setting, a parent/default configuration, or a local override. Entering a value overrides the inherited value; clearing a value returns the field to inherited state. The GUI and generated Python represent the same workflow, so a domain expert can remain in the GUI while a computational collaborator reviews generated code or adds a quality-control function.

### CellProfiler workflows are preserved under parity checks

CellProfiler import tests preservation of a widely used, trusted workflow format. A `.cppipe` file contains module settings, image names, object names, measurement expectations, display choices, and output behavior. OpenHCS parses the file, maps image-loading and metadata modules to image sources, maps processing modules to CellProfiler-compatible functions, and stores images, objects, measurements, relationships, grids, and saved outputs as named workflow results.

| CellProfiler concept | OpenHCS representation | Result |
|---|---|---|
| `.cppipe` file | Imported pipeline file | Trusted workflows enter OpenHCS without manual rewriting |
| Images, Metadata, NamesAndTypes | Image sources and image-name-to-file matching | Image identity is checked before execution |
| Processing module | OpenHCS workflow step with a CellProfiler-compatible callable | Module behavior runs through the normal analysis workflow |
| Images and objects | Named image and label outputs | Intermediates can be stored, streamed, compared, or reused |
| Measurements | Named measurement outputs | Tables remain tied to the workflow that produced them |
| Relationships and grids | Named non-image outputs | Object relationships and geometry are preserved explicitly |
| SaveImages and ExportToSpreadsheet | Output destinations and format writers | Files are produced from workflow results, not hidden side effects |
| Module settings | Traceable parameter state | Settings can be edited, inherited, and reviewed in the workflow record |

Compatibility is defined by strict output parity. A workflow counts only when native CellProfiler outputs and OpenHCS outputs match under the declared comparison policy. Numeric values, label identities, object relationships, measurement rows, and materialized files are compared separately where relevant. Speed is reported after output parity is established (Figure 2).

The benchmark target contains 30 `.cppipe` workflows drawn from official benchmark, example, tutorial, and public workflow sources. The corpus covers object identification, object filtering, size and shape measurement, texture measurement, colocalization, image math, illumination correction, grid-based object assignment, object tracking, object-to-image conversion, object overlays, image export, table export, and specialized morphology such as worm untangling. Manifest-backed categories record the source category, assay family, image-analysis behavior, and output-file behavior for each workflow.

For each workflow, native CellProfiler produced reference outputs. OpenHCS imported and ran the same `.cppipe` file, then compared corresponding outputs under the declared equivalence policy. All 30 workflows reached a parity accuracy fraction of 1.0 in the benchmark summary. A workflow was counted as passing only when the comparison reported no unresolved differences. The benchmark artifacts retain the pass table, module coverage, unsupported-feature status, speedup, and tolerated numerical differences.

The benchmark coverage table contained 30 supported `.cppipe` cases, 471 CellProfiler module instances, 58 unique CellProfiler module names, and 7,158 setting rows. All setting rows were mapped, and the benchmark reported no missing processing modules and no known-invalid absorbed processing modules. Across the current absorbed CellProfiler module catalogue, 53 modules are explicitly covered, 28 additional modules are covered by a shared semantic abstraction, and 8 modules remain outside the covered set. Version-specific parity artifacts record the CellProfiler and OpenHCS versions used for each comparison; new CellProfiler versions receive their own declared compatibility checks before output preservation or speed is reported for those versions.

### OpenHCS is at least 4x faster under constrained single-thread CPU-only conditions

The primary performance benchmark constrains execution to one sample, one thread/core, CPU-only execution, and no batching. This condition separates the CellProfiler-compatible execution path from additional cores, GPU kernels, larger batch size, and amortized throughput over many wells.

The benchmark reports execution-only timing separately from total phase timing. Execution-only timing measures the part of the run most directly comparable to CellProfiler module execution. Total phase timing includes startup, imports, compilation, preparation, and execution, reflecting a one-off user run.

Across the 30-workflow target, native CellProfiler median execution time was 5.40 s and OpenHCS median execution time was 0.691 s. Execution-only speedup was at least 4.03x for every workflow, with median 7.60x, mean 43.3x, and maximum 839x. All 30 workflows met the 4x execution-speed target. Total phase speedup was lower because startup and compilation are included; the median total phase speedup was 3.55x, and 10 of 30 workflows met the 4x target under that cold-run measurement.

The single-thread CPU-only result establishes the performance floor for imported CellProfiler workflows. GPU-backed functions and backend-specific algorithms are supported as intentional OpenHCS methods, but they are not part of the CellProfiler-preserving speed result unless separately benchmarked against an appropriate reference.

### Sources, viewers, code, and Python methods remain attached to the run

Many microscopy groups organize data in managed stores such as OMERO. Others work from local disks, Zarr-backed stores, or acquisition folders exported by vendor acquisition software. OpenHCS treats those inputs as workflow sources. The workflow records which images are analyzed, how images map to biological and acquisition dimensions, and which outputs were produced from those images.

Microscope handlers convert acquisition-specific file layouts into analysis-ready image identities. ImageXpress-style exports can place timepoints and z-planes in nested folders rather than encoding every dimension in each filename. Opera Phenix exports can use acquisition-order field indices rather than spatial field order and can omit images when autofocus fails. OpenHCS builds an analysis-ready view of the plate, normalizes field labels, and fills missing Opera Phenix images with black placeholders where the handler can infer the expected grid. Bio-Formats-backed source discovery extends the same handler system to broadly readable microscopy datasets by labeling discovered images by series, channel, z-plane, and timepoint and requiring explicit user mapping when well or site labels cannot be inferred safely.

Viewer integration follows the named-output model. napari and Fiji outputs are enabled through step configuration. When napari or Fiji streaming is enabled for a step, OpenHCS launches or reuses the viewer on the configured port and streams that step's images while the pipeline runs. The same mask can be saved, compared against a reference, streamed to napari, or sent to Fiji because the mask remains a named workflow result.

OpenHCS exposes ordinary Python functions as workflow steps with the same parameter editing, validation, viewer output, generated code, and worker execution used by imported modules. Function signatures expose editable parameters. Memory-backend declarations record whether a function expects arrays from a NumPy-like CPU library, a GPU array library, or a deep-learning framework. An imported CellProfiler workflow can therefore be extended with an assay-specific quality-control function, and a native OpenHCS workflow can mix CPU image processing, GPU-compatible array operations, and table-producing functions when those choices are appropriate for the analysis.

The main interoperability relationships are summarized in the following table. The table states the role OpenHCS preserves for each tool or platform and the part of the OpenHCS validation that supports the relationship.

| Existing tool or platform | Preserved role | OpenHCS integration | Validation status in this draft |
|---|---|---|---|
| CellProfiler | Trusted modular bioimage analysis and `.cppipe` workflows | CellProfiler pipelines run through OpenHCS with loading semantics preserved, named results exposed, and parity checked against native outputs | 30-workflow parity corpus; speed reported only after parity |
| Fiji/ImageJ | Familiar inspection and image-processing environment | Fiji receives selected step outputs as named workflow results | Implemented viewer destination; GUI/runtime environment requirements reported separately |
| napari | Python-native multidimensional image inspection | napari receives selected step outputs while workflow identity and output names remain attached | Implemented viewer destination with available automated viewer path |
| OMERO and BIOMERO-style systems | Managed image storage, collaboration, and launch surfaces | OMERO can provide workflow sources and managed environments can launch OpenHCS workflows | OMERO support tested across multiple versions; launch-surface support should cite exact tested versions |
| Bio-Formats | Broad pixel and metadata readability | Bio-Formats-backed discovery labels images where metadata support safe inference and requires explicit mapping when metadata are ambiguous | Source discovery uses the same normalized image-label model as vendor handlers |
| Microscope vendor folders | Acquisition-specific plate layouts and metadata conventions | Vendor handlers normalize wells, fields, channels, z-planes, timepoints, and missing-image behavior before analysis starts | ImageXpress and Opera Phenix quirks are handled explicitly |
| Python, GPU, and deep-learning libraries | Assay-specific operations and backend-specific methods | Python callables become workflow steps with declared parameters, memory expectations, outputs, viewer policies, and worker execution | Integrated as native methods; CellProfiler equivalence requires separate validation |
| Generic workflow managers | Scalable and reproducible command execution | OpenHCS supplies bioimage-specific source identity, named outputs, viewer destinations, and parity artifacts inside an executable workflow | Worker benchmarks report throughput, queue depth, and RAM tradeoffs |

### Many-well throughput uses persistent workers

OpenHCS is also evaluated in the way high-content screening is often used: many samples, persistent worker processes, and enough RAM to keep workers alive across repeated work. In that setting, import cost, compile cost, and backend warmup are paid once and divided across many samples rather than paid again for every well.

Throughput was measured by replicated-well runs over the same 30 workflows. With two cores and four wells per core, projected execution speedup had minimum 7.22x, median 9.90x, mean 53.2x, and maximum 1,196x. With three cores and four wells per core, the corresponding values were 10.8x, 14.3x, 74.6x, and 1,644x. With four cores and eight wells per core, they were 12.3x, 16.8x, 83.8x, and 1,823x. At four cores, increasing queue depth from one to eight wells per core raised median projected execution speedup from 12.9x to 16.8x.

Scaling is reported per resource, not only per wall-clock speedup. Each worker adds CPU capacity but also memory pressure because it may hold imported libraries, compiled kernels, open stores, cached arrays, and intermediate outputs. In the four-core queue-depth sweep, median peak RAM increased from 3.98 GB at one well per core to 4.25 GB at eight wells per core, with maximum peak RAM reaching 14.6 GB.

## Discussion

OpenHCS preserves existing bioimage workflows by keeping sources, parameters, intermediate outputs, viewers, generated code, and worker execution in one provenance-tracked workflow. CellProfiler import supplies the strongest preservation test in this draft. Imported `.cppipe` workflows run as OpenHCS workflows, reproduce native CellProfiler outputs under declared parity checks, and retain named results that can be inspected, extended, serialized, streamed to viewers, and executed through workers.

Compatibility also creates an extension path. A preserved CellProfiler workflow can be modified with an ordinary Python function, routed to napari or Fiji for step-level inspection, reviewed as generated Python, or run across persistent workers. Native OpenHCS workflows can use the same source, output, viewer, function, and worker mechanisms without requiring a CellProfiler origin. Backend-specific GPU or deep-learning methods remain distinct scientific methods unless a separate parity or validation target is defined for them.

The validation scope remains bounded. The CellProfiler importer covers the module and setting subset represented in the parity-tested corpus. Modules outside that set require explicit compatibility work before preservation is reported. Bio-Formats improves readability, and ambiguous well, site, channel, z-plane, or timepoint identity still requires explicit source mapping. Viewer integration depends on GUI/runtime environment availability. Persistent workers improve throughput while adding process-lifecycle, memory, restart, and cache-warmth tradeoffs. Performance results therefore remain separated into execution-only time, total wall time, single-thread CPU speed, and many-worker throughput.

The benchmark establishes a preservation-plus-performance baseline. Under one-sample, one-thread/core, CPU-only conditions, every tested imported CellProfiler workflow ran at least 4.03x faster than native CellProfiler execution after output parity was established. In many-sample settings, persistent workers amortized startup and compilation costs while exposing throughput and RAM tradeoffs directly. These results support OpenHCS as a backwards-compatible workflow platform for labs that want to keep trusted analyses while adding provenance, viewer inspection, custom functions, generated code, and scalable execution.

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

Native CellProfiler is run for each benchmark workflow to generate reference outputs. OpenHCS then imports and runs the same `.cppipe` file. Outputs are compared through equivalence checks for images, object labels, measurement rows, relationships, scalar values, and materialized files as appropriate for the workflow. Numeric comparisons use declared absolute and relative tolerances. Non-numeric identifiers and categorical values are compared exactly unless a specific CellProfiler-compatible normalization is documented. Each parity report records the CellProfiler version or commit and the OpenHCS commit used for the comparison.

### Performance benchmarking

The primary benchmark condition uses one sample, one thread/core, CPU-only execution, and no batching. Timing is reported as execution-only time and total wall time. Execution-only time isolates module/runtime execution after startup and compile overhead. Total wall time includes startup, imports, compile, preparation, and execution. Throughput benchmarks are reported separately using multi-sample execution and worker-level parallelism. GPU-backed functions are reported only as OpenHCS method support unless a separate benchmark defines the GPU hardware, backend version, reference behavior, and comparison policy.

The benchmark runs use local or explicitly mounted paths recorded in the benchmark manifest. Cloud-bursting or network-filesystem-latency performance requires a benchmark that records the storage environment, path form, cache state, and worker placement.

### Benchmark corpus

The target benchmark corpus contains 30 `.cppipe` workflows drawn from official benchmark, example, tutorial, and public third-party workflow sources. The corpus table lists each workflow, source category, source URL or citation, dataset source, assay family, the image-analysis behavior it tests, the output files it produces, module coverage, parity status, single-thread/core speedup, and throughput status where available.

### Reproducibility package

The benchmark runner emits the OpenHCS commit, CellProfiler version or commit, Python version, operating system, CPU model, GPU model if used, RAM, pipeline manifest, dataset manifest, checksums where available, native CellProfiler timing, OpenHCS timing, parity reports, and raw CSVs used to generate figures. Benchmark figures are regenerated from saved CSV files without rerunning CellProfiler.

## Draft Figure Captions

### Figure 1. Sources, tools, viewers, code, and workers remain one workflow

Automated microscopy produces wells, sites, channels, z-planes, and timepoints that become masks, measurements, quality-control decisions, reruns, and output tables. OpenHCS keeps source identities, parameters, named outputs, viewer destinations, generated code, and worker execution attached to one runnable workflow. Diagram source: `paper/figures/rendered/fig01_field_integration_gap.svg`.

### Figure 2. CellProfiler workflows are imported, parity checked, and benchmarked

The benchmark manifest feeds native CellProfiler and OpenHCS runs. OpenHCS parses `.cppipe` files into image sources, image-name-to-file matching, CellProfiler-compatible workflow steps, named outputs, and saved results. Native CellProfiler outputs provide the parity reference. The figure separates output parity, module coverage, constrained one-sample CPU-only speedup, cold-run overhead, many-sample persistent-worker throughput, and RAM scaling. Diagram sources: `paper/figures/rendered/fig03_cellprofiler_import_path.svg` and `paper/figures/rendered/fig04_benchmark_validation_structure.svg`.

### Figure 3. GUI edits, generated Python, and Python functions modify the same workflow

The pipeline editor, generated Python representation, source binding, step-level napari/Fiji output, ordinary Python functions, backend-specific functions, and worker execution all target the workflow being edited and run. Diagram sources: `paper/figures/rendered/fig07_typed_state_bidirectional_editing.svg` and `paper/figures/rendered/fig06_backend_extensibility.svg`.

### Figure 4. Persistent workers separate cold-run overhead from throughput

Startup, imports, compilation, backend warmup, execution, output writing, queue depth, worker count, and RAM are reported as separate quantities. This separation distinguishes one-off execution timing from many-sample throughput. Diagram source: `paper/figures/rendered/fig05_throughput_amortization.svg`.

## Supplementary Table Captions

### Supplementary Table 1. Benchmark corpus

Each row lists one `.cppipe` workflow, source category, source URL or citation, dataset source, assay family, what kinds of image-analysis behavior it tests, what output files it produces, CellProfiler modules used, native CellProfiler runtime, OpenHCS execution runtime, total OpenHCS runtime, speedup, parity status, and notes. Parity status and speedup are separate columns so performance is never reported without correctness context.

### Supplementary Table 2. CellProfiler module coverage

Each row lists one CellProfiler module class, import status, explicitly parity-tested status, theoretically covered status for modules sharing implemented abstractions, not-covered status, accelerated path where relevant, backend used, unsupported settings or features, and notes.

### Supplementary Table 3. Worker/RAM scaling

Each row lists a throughput condition, pipeline group, sample count, worker count, core count, peak RAM, RAM per worker, total wall time, samples per hour, speedup versus native CellProfiler, and speedup versus OpenHCS single-worker execution.

## Code and Data Availability

OpenHCS source code: `https://github.com/OpenHCSDev/OpenHCS`.

OpenHCS documentation: `https://openhcs.readthedocs.io/`.

Benchmark scripts, manifests, raw timing CSVs, parity reports, figure-generation scripts, and generated figures: `https://github.com/OpenHCSDev/openhcs`.

CellProfiler pipeline sources and dataset acquisition manifests: `https://github.com/OpenHCSDev/openhcs`.

Reusable libraries:

- ObjectState: `https://github.com/OpenHCSDev/objectstate`
- ArrayBridge: `https://github.com/OpenHCSDev/arraybridge`
- PolyStore: `https://github.com/OpenHCSDev/PolyStore`
- ZMQRuntime: `https://github.com/OpenHCSDev/zmqruntime`
- pyqt-reactive: `https://github.com/OpenHCSDev/pyqt-reactive`
- pycodify: `https://github.com/OpenHCSDev/pycodify`
- python-introspect: `https://github.com/OpenHCSDev/python-introspect`
- metaclass-registry: `https://github.com/OpenHCSDev/metaclass-registry`

## References To Format

- `[Carpenter2006]` Carpenter et al. CellProfiler: image analysis software for identifying and quantifying cell phenotypes. Genome Biology 7, R100 (2006). DOI: 10.1186/gb-2006-7-10-r100.
- `[McQuin2018]` McQuin et al. CellProfiler 3.0: next-generation image processing for biology. PLoS Biology 16, e2005970 (2018). DOI: 10.1371/journal.pbio.2005970.
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
