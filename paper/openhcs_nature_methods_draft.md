# OpenHCS: a composable bioimage workflow platform

**Working draft.** Public adoption, funding, citation, company-use statements, and final integration validation require source/code verification before submission.

## Abstract

Modern microscopes can acquire images faster than many labs can analyze them. After acquisition, researchers still need to segment cells, inspect masks, measure phenotypes, add assay-specific logic, and rerun analyses across many samples. Existing tools are valuable, but routine work becomes harder to trust when parameters are copied between programs, intermediate masks become disconnected files, and inspection is separated from batch execution. We present OpenHCS, a composable bioimage workflow platform that keeps trusted tools in one analysis path. OpenHCS starts from microscope folders, Bio-Formats-readable datasets, or CellProfiler pipelines; streams intermediates to napari or Fiji; adds ordinary Python, GPU, and deep-learning steps; exports CellProfiler/CellProfiler Analyst-compatible results; and executes across workers. OpenHCS reproduces native CellProfiler outputs across 30 benchmark workflows and achieves at least 4x execution speedup for every tested workflow under one-sample, one-thread/core, CPU-only conditions.

Image analysis is where microscopy becomes biological evidence. A screen or time-course may begin with automated acquisition, but the experiment is not interpretable until cells, organelles, tissues, tracks, intensities, textures, morphologies, and quality-control failures have been identified and reviewed. As acquisition scales, this analysis should scale with it. In practice, it often becomes the slowest and most fragile part of the workflow.

The bottleneck includes runtime, review, extensibility, and reproducibility. Users need to adapt an analysis when the assay changes, inspect intermediate masks when results look wrong, add small pieces of custom logic, repeat the same workflow across many samples, and explain which parameters produced a table or figure. These ordinary scientific needs often span multiple environments: high-throughput batch tools, plugin systems, commercial workflows, open viewers, managed image stores, Python functions, and reproducible execution systems.

ImageJ/Fiji, CellProfiler, napari, OMERO, Zarr-backed image storage, Bio-Formats, scientific Python, GPU libraries, and workflow systems each solve important parts of biological image analysis and computational reproducibility [Schneider2012; Schindelin2012; Carpenter2006; McQuin2018; Sofroniew2019; Allan2012; Moore2021; BioFormats; vanDerWalt2014; Haase2020; Koster2012; DiTommaso2017; Galaxy2020; Galaxy2024]. The main friction emerges at the boundaries between these tools. Moving between tools forces parameters to be copied manually, intermediate images to be saved without clear provenance, viewers to inspect masks that may not correspond to the current run, and custom code to lose critical dimensional context such as channel, timepoint, z-plane, field, or well. Bio-Formats makes many proprietary files readable; OpenHCS microscope handlers turn readable files and vendor plate folders into analysis-ready image sets with explicit well, site, channel, z-plane, and timepoint labels.

OpenHCS addresses this boundary problem by keeping images, parameters, intermediate masks, measurement tables, exported files, viewer output, and worker execution together as the analysis moves between tools. A GUI edit, generated Python file, viewer request, imported CellProfiler module, or custom Python function all update the analysis being run. Imported workflows can be inspected, inherited, overridden, extended, serialized, and rerun without creating a separate copy of the experiment.

Behind the scenes, OpenHCS records what images are being used, what each step expects, what each step produces, where outputs should go, which array or GPU backend is required, and how the work should run. These implementation details support a simple user-facing guarantee: before a run starts, OpenHCS checks whether the analysis is ready to execute.

CellProfiler compatibility is the strictest test of this model because `.cppipe` files encode years of trusted biological image-analysis practice. OpenHCS compiles `.cppipe` files into native, fully integrated workflows, compares outputs against native CellProfiler, and reports speed only after parity is established. We use CellProfiler parity as a stringent baseline to validate imported legacy workflows. Native OpenHCS functions and backend-specific algorithms operate beyond this baseline, allowing users to intentionally select alternative scientific methods.

We show that OpenHCS makes analysis and review scale with image acquisition while preserving the tools and workflows labs already trust. We validate this by importing a broad CellProfiler workflow corpus with loading semantics intact, reproducing native outputs under output-parity checks, measuring constrained CPU-only speedups, and demonstrating step-level viewer output, generated Python, custom functions, managed sources, and scalable execution.

## Results

### OpenHCS keeps images, masks, code, viewers, and batch runs together

OpenHCS starts from a practical observation: once images are acquired, analysis and review should scale with the experiment. The implemented proof chain follows a pipeline from import or source discovery, through inspection and extension, to parity testing and scaled execution. Steps that often become separate files or side scripts remain part of the runnable pipeline.

In a representative OpenHCS run, a `.cppipe` file and its encoded image-loading rules enter OpenHCS as a workflow; source images are resolved into well, site, channel, z-plane, and timepoint identities; processing modules produce named images, objects, measurements, and files; a selected mask can be streamed to napari or Fiji during execution; an assay-specific Python quality-control function can be inserted as another step; and the final outputs can be compared to native CellProfiler or executed across many wells with persistent workers.

OpenHCS keeps trusted tools usable together. If a workflow was built in CellProfiler, OpenHCS preserves how that pipeline loads images and what each image name means. If a collaborator needs Python, the pipeline can be exported as readable code. If a mask needs inspection, enabling viewer output on that step sends it to napari or Fiji during execution. Native OpenHCS workflows can additionally point to local folders, managed stores, or microscope-handler image sets directly.

Imported workflows remain dynamic: they can be inspected, modified, extended with Python, reviewed as code, streamed to viewers, and rerun at scale.

### OpenHCS catches setup mistakes before long runs

OpenHCS checks a workflow before running it. Image inputs must resolve to the intended well, site, channel, z-plane, and timepoint; each Python or imported function must have the inputs it expects; CPU or GPU requirements must be explicit; and masks, measurements, files, viewer outputs, and worker execution must be deliberately configured. Common failures such as wrong channels, stale acquisition folders, missing intermediate masks, incompatible array libraries, or outputs that no longer match the current run appear before long processing runs rather than after hours of computation.

Parameters remain traceable instead of fragile. A threshold, channel name, output setting, or source mapping can be inherited from a default, changed locally, or generated from an imported setting. The GUI can show where a value came from and what changed. Generated Python reconstructs the same workflow state, so a visual edit can become reviewable source code instead of an opaque GUI file.

Users do not need to understand the implementation to benefit from it. In the GUI, clearing a field means "inherit," entering a value means "override," and generated code explains the current workflow. A wet-lab user can remain in the GUI; a computational collaborator can review generated Python or add a short quality-control function; both are editing the pipeline that will run.

### Custom Python, GPU, and deep-learning methods become workflow steps

OpenHCS exposes ordinary Python functions as workflow steps with the same parameter editing, validation, viewer output, generated code, and worker execution used by imported modules. OpenHCS reads the function's parameters and records which library it expects, such as NumPy, Numba, CuPy, CuCIM, JAX, PyTorch, TensorFlow, pyclesperanto, or another supported backend. A lab can register an existing callable instead of rebuilding it as a separate plugin, script runner, GUI, and batch executor.

An imported CellProfiler workflow can therefore be extended with a short Python function that computes an assay-specific quality-control image. A native OpenHCS workflow can mix CPU image processing, GPU-compatible array operations, and table-producing functions when those choices are appropriate for the analysis. The function author writes the scientific operation rather than a separate GUI, serializer, viewer bridge, worker protocol, and benchmark adapter.

Backend-specific algorithms serve as intentional methodological extensions, integrated into the same parameter, output, and inspection system as legacy steps. CellProfiler parity is required when OpenHCS claims to preserve a CellProfiler module. A CuCIM, pyclesperanto, JAX, PyTorch, or TensorFlow function can also be used intentionally as a different analysis method. In both cases, parameters are visible, intermediates can be inspected, outputs can be saved or streamed deliberately, and execution can be scaled.

### Image folders, OMERO, napari, and Fiji remain part of the run

Many microscopy groups already organize data in managed stores such as OMERO. Others work from local disks, Zarr-backed stores, or acquisition folders exported by vendor acquisition software. OpenHCS treats these as workflow sources rather than side archives. The workflow records which images are being analyzed, how they map to biological and acquisition dimensions, and which outputs were produced from them.

Microscope handlers are the practical bridge between acquisition and analysis. ImageXpress-style exports can place timepoints and z-planes in nested folders rather than encoding every dimension in each filename; Opera Phenix exports can use acquisition-order field indices rather than spatial field order and can omit images when autofocus fails. OpenHCS handles these quirks before analysis starts by building an analysis-ready view of the plate, normalizing field labels, and filling missing Opera Phenix images with black placeholders where the handler can infer the expected grid. Bio-Formats-backed source discovery extends the same handler system to broadly readable microscopy datasets: Bio-Formats provides pixel and metadata access, while OpenHCS labels the discovered images by series, channel, z-plane, and timepoint and asks the user for help when well or site labels cannot be inferred safely.

Viewer integration follows the same principle. napari and Fiji outputs are not one-off side effects hidden inside processing code. They are enabled through step configuration. When napari or Fiji streaming is enabled for a step, OpenHCS launches or reuses the viewer on the configured port and streams that step's images while the pipeline runs. The same mask can be saved, compared against a reference, streamed to napari, or sent to Fiji because it remains a named result of the workflow.

An OpenHCS workflow can be reviewed as Python by a computational collaborator and inspected as images, channels, masks, measurements, and viewer outputs by a wet-lab user. CellProfiler-imported workflows preserve their encoded image-loading behavior; native OpenHCS workflows can use local folders, managed stores, microscope handlers, or Bio-Formats-readable datasets; napari, Fiji, generated Python, and benchmark outputs all refer to the pipeline being executed.

These integrations are implemented as part of the current system. OMERO support is tested across multiple versions, and napari and Fiji are implemented as viewer destinations for selected workflow outputs. Fiji requires a local graphical test environment, whereas napari can be exercised through the available automated viewer path.

### CellProfiler workflows become editable OpenHCS analyses

CellProfiler import tests preservation of an existing, mature workflow format. A `.cppipe` file contains module settings, image names, object names, measurement expectations, display choices, and output behavior. OpenHCS parses that file, maps image-loading and metadata modules to image sources, maps processing modules to CellProfiler-compatible functions, and stores images, objects, measurements, relationships, grids, and saved outputs as named workflow results.

| CellProfiler concept | OpenHCS representation | Result |
|---|---|---|
| `.cppipe` file | Imported pipeline file | Trusted workflows enter OpenHCS without manual rewriting |
| Images, Metadata, NamesAndTypes | Image sources and image-name-to-file matching | Image identity is checked before execution |
| Processing module | OpenHCS workflow step with a CellProfiler-compatible callable | Module behavior runs through the normal analysis workflow |
| Images and objects | Named image and label outputs | Intermediates can be stored, streamed, compared, or reused |
| Measurements | Named measurement outputs | Tables remain tied to the workflow that produced them |
| Relationships and grids | Named non-image outputs | Object relationships and geometry are preserved explicitly |
| SaveImages and ExportToSpreadsheet | Output destinations and format writers | Files are produced from workflow results, not hidden side effects |
| Module settings | Traceable parameter state | Settings can be edited, inherited, serialized, and audited |

We define compatibility by strict output parity: a pipeline counts only when native CellProfiler outputs and OpenHCS outputs match under the declared comparison policy. Numeric values, label identities, object relationships, measurement rows, and materialized files are compared separately where relevant. Speed is reported after output parity is established.

The official30 coverage analysis contained 30 supported `.cppipe` cases, 471 CellProfiler module instances, 58 unique CellProfiler module names, and 7,158 setting rows. All setting rows were mapped, and the benchmark reported no missing processing modules and no known-invalid absorbed processing modules. Across the current absorbed CellProfiler module catalogue, 53 modules are explicitly covered, 28 additional modules are covered by a shared semantic abstraction, and 8 modules remain outside the covered set.

### OpenHCS reproduces CellProfiler outputs across a broad benchmark corpus

The CellProfiler benchmark corpus is designed to test generality across more than a single curated demonstration. The target set contains 30 `.cppipe` workflows drawn from official benchmark, example, tutorial, and public workflow sources. Official examples test known CellProfiler semantics, tutorials test user-facing workflows, and public third-party pipelines test whether the importer generalizes beyond examples used during development.

The corpus covers common bioimage and HCS analysis patterns: object identification, object filtering, size and shape measurement, texture measurement, colocalization, image math, illumination correction, grid-based object assignment, object tracking, object-to-image conversion, object overlays, image export, table export, and specialized morphology such as worm untangling. Each workflow is assigned manifest-backed categories such as source category, assay family, what kinds of image-analysis behavior it tests, and what kinds of output files it produces. These categories make the benchmark interpretable without relying on filename heuristics.

For each workflow, native CellProfiler produced reference outputs. OpenHCS imported and ran the same `.cppipe` file, then compared the corresponding outputs under the declared equivalence policy. All 30 workflows reached a parity accuracy fraction of 1.0 in the benchmark summary. A workflow was counted as passing only when the comparison reported no unresolved differences; the benchmark artifacts retain the pass table, module coverage, unsupported-feature status, speedup, and tolerated numerical differences.

### OpenHCS is at least 4x faster under constrained single-thread CPU-only conditions

To isolate algorithmic speedup from hardware scaling, the primary performance benchmark constrains execution to one sample, one thread/core, CPU-only execution, and no batching. Under this constraint, execution speed is separated from additional cores, GPU kernels, larger batch size, and amortized throughput over many wells.

The benchmark reports execution-only timing separately from total phase timing. Execution-only timing measures the part of the run most directly comparable to CellProfiler module execution. Total phase timing includes startup, imports, compilation, preparation, and execution, reflecting a one-off user run.

Across the 30-workflow target, native CellProfiler median execution time was 5.40 s and OpenHCS median execution time was 0.691 s. Execution-only speedup was at least 4.03x for every workflow, with median 7.60x, mean 43.3x, and maximum 839x. All 30 workflows met the 4x execution-speed target. Total phase speedup was lower because startup and compilation are included; the median total phase speedup was 3.55x, and 10 of 30 workflows met the 4x target under that cold-run measurement.

The single-thread CPU-only result establishes the performance floor, demonstrating that trusted CellProfiler-compatible workflows are accelerated before applying GPU acceleration or parallel scaling. Throughput and backend-specific acceleration are additional layers on top of this matched baseline.

### Many-well throughput uses persistent workers

OpenHCS is also evaluated in the way high-content screening is often used: many samples, persistent worker processes, and enough RAM to keep workers alive across repeated work. In that setting, import cost, compile cost, and backend warmup are paid once and divided across many samples rather than paid again for every well.

Throughput was measured by replicated-well runs over the same 30 workflows. With two cores and four wells per core, projected execution speedup had minimum 7.22x, median 9.90x, mean 53.2x, and maximum 1,196x. With three cores and four wells per core, the corresponding values were 10.8x, 14.3x, 74.6x, and 1,644x. With four cores and eight wells per core, they were 12.3x, 16.8x, 83.8x, and 1,823x. At four cores, increasing queue depth from one to eight wells per core raised median projected execution speedup from 12.9x to 16.8x.

Scaling is reported per resource, not only per wall-clock speedup. Each worker adds CPU capacity but also memory pressure because it may hold imported libraries, compiled kernels, open stores, cached arrays, and intermediate outputs. In the four-core queue-depth sweep, median peak RAM increased from 3.98 GB at one well per core to 4.25 GB at eight wells per core, with maximum peak RAM reaching 14.6 GB.

## Discussion

OpenHCS is built around a simple claim: once images are acquired, the analysis should remain easy to inspect, edit, and rerun. CellProfiler import, OMERO and Zarr-backed source handling, Fiji and napari inspection, custom Python functions, backend memory conversion, generated Python, reactive parameter state, and persistent workers all serve that goal.

Compatibility becomes an entry point for extension. A preserved CellProfiler workflow can still be inspected, edited, extended, serialized, streamed to viewers, and executed through workers. A custom Python function, managed image source, or backend-specific algorithm can be added when a lab intentionally chooses that method.

OpenHCS provides a structured runtime where existing tools retain their identities while participating in a unified, runnable analysis. A CellProfiler pipeline can run through OpenHCS; an OMERO- or BIOMERO-style environment can launch or provide sources for OpenHCS workflows; a Python callable can become a UI-accessible workflow step; and napari or Fiji can receive outputs from the same run.

The CellProfiler benchmark gives the platform a demanding validation case. It asks whether OpenHCS can preserve a mature external workflow format, reproduce native outputs, and improve execution speed under deliberately constrained conditions. CellProfiler parity validates the import mechanism; the broader platform extends legacy workflows with viewer integration, custom Python, generated code, managed sources, and scalable execution.

OpenHCS augments existing tools by letting their outputs, parameters, viewers, and execution paths work together. OMERO can be a source, local and microscope-handler folders can be sources, imported CellProfiler loading rules can define sources, and all of these can feed a runnable analysis.

| Existing tool or platform | Primary strength | Boundary where users still hit friction | OpenHCS relationship |
|---|---|---|---|
| CellProfiler | Trusted modular bioimage analysis and `.cppipe` workflows | Preserved workflows can be difficult to extend, inspect as named intermediates, accelerate, or combine with Python/viewer/worker execution | CellProfiler pipelines can run through OpenHCS with loading semantics preserved under parity checks, named results exposed, and the same workflow available to viewers, Python extensions, and workers |
| Fiji/ImageJ | Mature image-processing ecosystem and familiar inspection tools | Viewer or manual processing steps can become detached from the batch workflow that produced the images | OpenHCS treats Fiji output as a step-level destination for named workflow results |
| napari | Interactive multidimensional image viewing in Python | Inspection can be separate from the executable workflow state | OpenHCS streams selected step outputs to napari while keeping those outputs named in the workflow |
| OMERO and BIOMERO-style systems | Managed image storage, collaboration, and managed/HPC launch surfaces | The analysis still needs to know which images are being used, where outputs came from, which functions ran, and how results can be inspected or rerun | OMERO can provide sources and BIOMERO-style systems can launch OpenHCS workflows, while OpenHCS supplies the image-analysis workflow that runs on those sources |
| Bio-Formats | Broad file readability and metadata access | Opening a file does not always tell the analysis which well, site, channel, z-plane, or timepoint each image represents | Bio-Formats-backed discovery labels images where it can and asks for an explicit mapping when it cannot |
| Generic workflow managers | Scalable and reproducible command execution | They do not know that an image belongs to a particular well, channel, z-plane, mask, measurement table, viewer output, or CellProfiler-style result | OpenHCS supplies the bioimage-specific analysis state and can still report reproducible execution artifacts |

The scope of the current validation is explicit. The CellProfiler importer covers the module and setting subset represented in the parity-tested corpus; modules outside that set require explicit compatibility work before they can be claimed as preserved. Because GPU and array libraries can differ in boundary handling, data type behavior, reductions, randomization, and label handling, we treat them as distinct scientific methods. They are integrated as workflow steps and validated separately from CellProfiler parity. Bio-Formats improves file readability, but opening a file does not always reveal how its images should be assigned to wells, sites, channels, z-planes, or timepoints; OpenHCS asks for an explicit mapping when that information is ambiguous. Persistent workers improve throughput while introducing process-lifecycle, memory, restart, and cache-warmth tradeoffs that must be reported alongside speed. Viewer integration depends on GUI/runtime environment availability, so CI coverage and local GUI-dependent tests should be described separately. Performance claims also depend on benchmark scope and conditions; execution-only time, total wall time, single-thread CPU speed, and many-worker throughput should remain separate.

OpenHCS therefore gives labs a way to keep the tools they trust while making the workflow itself composable, inspectable, extensible, and executable across modern bioimage environments. Trusted CellProfiler `.cppipe` pipelines can be imported as OpenHCS workflows, executed with native-output parity, inspected through viewer and artifact systems, modified with custom Python functions, serialized as editable code, and scaled across persistent worker processes. Under one-sample, one-thread/core, CPU-only conditions, imported CellProfiler workflows ran at least 4x faster than native CellProfiler execution across the tested corpus. In many-sample settings, persistent workers amortized startup and compilation costs while exposing throughput and RAM tradeoffs directly.

## Online Methods

### Pipeline object model and compilation

OpenHCS pipelines are built from source definitions, parameter state, workflow steps, and output policies. Before execution, the pipeline is compiled into runtime contexts. Compilation resolves input sources, validates function requirements, determines intermediate and final outputs, assigns runtime paths, and prepares callable execution hooks. The compiled form is the unit submitted to workers.

### Source schema and source binding

Source schemas describe the experimental identity of input data, including dimensions such as well, site, channel, timepoint, and z-plane when present. Source bindings connect those named workflow sources to local files, managed stores, or virtual sources. This lets analysis code refer to the intended image role rather than a one-off path string.

### Microscope handlers and Bio-Formats source discovery

Microscope handlers convert acquisition-specific file layouts into analysis-ready image identities. Vendor-specific handlers encode known quirks such as nested timepoint and z-plane folders, field remapping, metadata sidecars, pixel-size lookup, channel names, and missing-image policy. The Bio-Formats handler uses Bio-Formats metadata to discover series, channel, z-plane, and timepoint dimensions for broadly supported datasets, then emits the same normalized image labels used by vendor-specific handlers. When Bio-Formats cannot infer well or site labels, OpenHCS requires an explicit source schema rather than silently constructing ambiguous image sets.

### Function registration and memory-backend requirements

OpenHCS functions can declare their memory interface using decorators such as `@numpy`, `@cupy`, `@pyclesperanto`, `@torch`, `@tensorflow`, or `@jax`. These declarations record the expected input and output memory types. During execution, OpenHCS converts image payloads between compatible backends when required. Function signatures are used to expose parameters in the GUI and generated Python representation.

### Runtime outputs and materialization

OpenHCS treats intermediate images, labels, measurements, relationships, grids, and files as named workflow outputs. Output policies determine whether a value is kept in memory, written to disk, stored in Zarr, streamed to napari, streamed to Fiji, compared against a reference, or discarded after execution.

### Viewer and storage backends

Viewer and storage integrations are treated as destinations for workflow outputs or sources for workflow inputs. napari and Fiji outputs receive named workflow results. OMERO, local disk, memory, and Zarr-backed stores participate in source and output handling according to the configured runtime path.

### Worker execution

OpenHCS uses process-level worker execution for isolation, progress reporting, viewer separation, and throughput scaling. Persistent workers can amortize startup, imports, compilation, and warmup across repeated samples. Worker-level benchmarks report wall time, execution time, sample count, worker count, and memory use.

### CellProfiler import

CellProfiler `.cppipe` files are parsed into module blocks and settings. Infrastructure modules are mapped into source schemas and source bindings. Processing modules resolve to CellProfiler-compatible functions. Each resolved module becomes an OpenHCS workflow step with a runtime adapter that provides access to CellProfiler-style images, object labels, measurements, grids, and relationships while storing outputs as OpenHCS workflow results.

### Parity comparison

Native CellProfiler is run for each benchmark workflow to generate reference outputs. OpenHCS then imports and runs the same `.cppipe` file. Outputs are compared through equivalence checks for images, object labels, measurement rows, relationships, scalar values, and materialized files as appropriate for the workflow. Numeric comparisons use declared absolute and relative tolerances. Non-numeric identifiers and categorical values are compared exactly unless a specific CellProfiler-compatible normalization is documented.

### Performance benchmarking

To isolate algorithmic speedup from hardware scaling, the primary benchmark condition uses one sample, one thread/core, CPU-only execution, and no batching. Timing is reported as execution-only time and total wall time. Execution-only time isolates module/runtime execution after startup and compile overhead. Total wall time includes startup, imports, compile, preparation, and execution. Throughput benchmarks are reported separately using multi-sample execution and worker-level parallelism.

### Benchmark corpus

The target benchmark corpus contains 30 `.cppipe` workflows drawn from official benchmark, example, tutorial, and public third-party workflow sources. The corpus table lists each workflow, source category, source URL or citation, dataset source, assay family, the image-analysis behavior it tests, the output files it produces, module coverage, parity status, single-thread/core speedup, and throughput status where available.

### Reproducibility package

The benchmark runner emits the OpenHCS commit, CellProfiler version or commit, Python version, operating system, CPU model, GPU model if used, RAM, pipeline manifest, dataset manifest, checksums where available, native CellProfiler timing, OpenHCS timing, parity reports, and raw CSVs used to generate figures. Benchmark figures are regenerated from saved CSV files without rerunning CellProfiler.

## Draft Figure Captions

### Figure 1. Scaled acquisition should not make analysis the bottleneck

Automated microscopy produces many wells, sites, channels, z-planes, and timepoints. Those images become segmentation masks, measurements, quality-control decisions, review tasks, reruns, and output tables. A biological analysis often spans copied parameters, exported files, detached viewers, custom scripts, batch jobs, and managed image stores. OpenHCS keeps these pieces together so existing tools can work from the same runnable pipeline.

### Figure 2. Imported CellProfiler workflows are preserved, checked, and accelerated

The benchmark manifest feeds native CellProfiler and OpenHCS runs. OpenHCS parses `.cppipe` files into image sources, image-name-to-file matching, CellProfiler-compatible workflow steps, named outputs, and saved results. Native CellProfiler outputs provide the parity reference. The same figure separates output parity, module coverage, constrained one-sample CPU-only speedup, cold-run overhead, many-sample persistent-worker throughput, and RAM scaling.

### Figure 3. OpenHCS workflows remain editable, extensible, inspectable, and executable

This multi-panel figure shows the platform as users encounter it: a pipeline editor with configured function steps; GUI and generated Python views of the same step; function-pattern editing for per-invocation behavior; source and microscope-handler binding; step-level napari/Fiji output; ordinary Python, GPU, and deep-learning functions as workflow steps; and worker execution. These views and destinations all refer to the pipeline being edited and run.

## Supplementary Table Captions

### Supplementary Table 1. Benchmark corpus

Each row lists one `.cppipe` workflow, source category, source URL or citation, dataset source, assay family, what kinds of image-analysis behavior it tests, what output files it produces, CellProfiler modules used, native CellProfiler runtime, OpenHCS execution runtime, total OpenHCS runtime, speedup, parity status, and notes. Parity status and speedup are separate columns so performance is never reported without correctness context.

### Supplementary Table 2. CellProfiler module coverage

Each row lists one CellProfiler module class, import status, explicitly parity-tested status, theoretically covered status for modules sharing implemented abstractions, not-covered status, accelerated path where relevant, backend used, unsupported settings or features, and notes.

### Supplementary Table 3. Worker/RAM scaling

Each row lists a throughput condition, pipeline group, sample count, worker count, core count, peak RAM, RAM per worker, total wall time, samples per hour, speedup versus native CellProfiler, and speedup versus OpenHCS single-worker execution.

## Code and Data Availability

OpenHCS source code: `https://github.com/OpenHCSDev/OpenHCS`.

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
