# OpenHCS: a composable bioimage workflow platform

**Working draft.** Numbers marked `TODO` should be replaced only from final benchmark artifacts. Public adoption, funding, citation, and company-use statements require source verification before submission.

## Plain-Language Summary

Bioimage workflows rarely live inside one program. A lab may keep images in OMERO, Zarr-backed storage, or acquisition folders, run a trusted CellProfiler pipeline, inspect masks in napari or Fiji, add a custom Python quality-control step, export measurement tables, and later repeat the same analysis across many wells. Each tool is useful, but the workflow can become fragmented: the images, parameters, masks, measurements, viewer outputs, and batch jobs stop clearly referring to the same analysis.

OpenHCS keeps those pieces connected as one workflow record. The same analysis can be edited in the GUI, exported as Python, extended with ordinary Python functions, connected to viewers by enabling napari or Fiji on selected steps, and executed across worker processes. CellProfiler import is the strictest validation case: OpenHCS preserves trusted `.cppipe` workflows including their image-loading semantics, reproduces native CellProfiler outputs under parity checks, and runs the tested workflows at least `TODO: 4x` faster under deliberately constrained single-thread CPU-only conditions.

## Abstract

Bioimage workflows increasingly span more than image analysis: they combine image metadata, multidimensional arrays, segmentation masks, measurement tables, managed image stores, interactive viewers, legacy pipeline formats, custom Python functions, and heterogeneous CPU/GPU backends. Existing tools remain essential, but their boundaries often turn one scientific workflow into disconnected GUI settings, scripts, exported files, viewer state, and batch jobs. We present OpenHCS, a composable bioimage workflow platform that keeps these pieces connected as one inspectable and executable workflow record. OpenHCS imports trusted CellProfiler pipelines with their loading semantics intact, streams selected step outputs to napari or Fiji when those step configurations are enabled, edits parameters through GUI and generated Python views of the same state, adds ordinary Python functions with declared memory backends, and scales execution across workers without changing what the workflow means. As a stringent validation, OpenHCS imports CellProfiler `.cppipe` workflows, reproduces native CellProfiler outputs under declared parity checks, and achieves at least `TODO: 4x` execution speedup across `TODO: 33` benchmark workflows under one-sample, one-thread/core, CPU-only conditions. OpenHCS provides a practical route for carrying trusted bioimage analyses into a composable, inspectable, and extensible execution platform rather than preserving them as static legacy files.

## Introduction

Bioimage analysis is rarely contained in one program. A lab may store images in OMERO, Zarr-backed storage, or acquisition folders, run a trusted CellProfiler pipeline, inspect masks in napari or Fiji, add a custom Python quality-control step, export measurement tables, and later repeat the workflow across many wells or a small focused dataset. Each tool is useful, but the scientific workflow is the thing that has to remain coherent: the same images, channels, parameters, masks, measurements, and outputs must refer to the same analysis state.

The problem is not that these tools are inadequate. ImageJ/Fiji, CellProfiler, napari, OMERO, Zarr-backed image storage, scientific Python, GPU libraries, and workflow systems each solve important parts of biological image analysis and computational reproducibility [Schneider2012; Schindelin2012; Carpenter2006; McQuin2018; Sofroniew2019; Allan2012; Moore2021; vanDerWalt2014; Haase2020; Koster2012; DiTommaso2017; Galaxy2020; Galaxy2024]. The failure mode appears between them. Parameters are copied between GUIs and scripts, intermediate images are saved as files whose provenance is no longer clear, viewers inspect masks that may not correspond to the current run, and custom code loses the dimensional and metadata assumptions carried by the original workflow.

OpenHCS addresses this boundary problem by keeping the workflow itself as the shared record. Images, source metadata, parameters, functions, intermediate masks, measurement tables, exported files, viewer streams, and worker execution are named parts of one workflow. A GUI edit, generated Python file, viewer request, imported CellProfiler module, or custom Python function all target the same analysis instead of creating parallel versions. This prevents imported workflows from becoming dead ends: they can be inspected, inherited, overridden, extended, serialized, and rerun through the same state model.

Internally, OpenHCS uses typed source schemas, function contracts, runtime artifacts, storage backends, memory-backend conversion, and process-isolated workers. These terms are implementation details of a simple user-facing guarantee: before a run starts, OpenHCS checks what data exist, what each step consumes and produces, where outputs will go, which memory backend is required, and which worker will execute the work.

CellProfiler compatibility is the strictest test of this model because `.cppipe` files encode years of trusted biological image-analysis practice. OpenHCS does not treat them as opaque files or run them in a separate sidecar. It compiles them into normal OpenHCS workflows, compares outputs against native CellProfiler, and reports speed only after parity is established. CellProfiler parity is not the standard for every OpenHCS analysis; it is the preservation test for imported CellProfiler workflows. Native OpenHCS functions and backend-specific algorithms can be used when a lab intentionally chooses a different analysis method.

We show that OpenHCS keeps bioimage workflows coherent across legacy pipelines, Python functions, viewers, storage backends, and worker execution. We validate this by importing a broad CellProfiler workflow corpus with loading semantics intact, reproducing native outputs under output-parity checks, measuring constrained CPU-only speedups, and demonstrating how the same workflow record supports step-level viewer output, generated Python, custom functions, managed sources, and scalable execution.

## Results

### OpenHCS keeps one workflow record across tools

OpenHCS starts from a practical observation: the workflow is more important than any one interface. The implemented path imports real CellProfiler `.cppipe` files with their image-loading semantics intact, enables napari or Fiji output on selected steps, adds Python functions through the same workflow-step mechanism, and runs parity, single-thread speed, and worker-scaling benchmarks. In many systems, those pieces become separate records. OpenHCS keeps them attached to one workflow record.

In an OpenHCS workflow, images, channels, parameters, intermediate masks, measurements, viewer outputs, generated Python, and worker execution remain connected. Step-level napari and Fiji configurations determine which step outputs are streamed. When enabled, OpenHCS launches or reuses the viewer on the configured port and sends images while the pipeline runs. These actions change or inspect the same workflow rather than creating a GUI version, a script version, a viewer version, and a batch-runner version.

This matters because labs already have trusted tools. OpenHCS is not designed to replace CellProfiler, Fiji, napari, OMERO, Zarr-backed storage, or Python. It is designed to let a lab keep using them together. If a workflow was built in CellProfiler, OpenHCS preserves the pipeline's loading, metadata, and analysis semantics. If a collaborator needs Python, the same workflow can be exported as readable code. If a mask needs inspection, enabling viewer output on that step sends it to napari or Fiji during execution. Native OpenHCS workflows can additionally use explicit source bindings for local and managed sources.

The imported or edited workflow is therefore not a compatibility artifact frozen in place. It becomes more inspectable and extensible than the original file: parameters have provenance, intermediates can be requested by name, custom functions can be inserted, and execution can move from a one-sample comparison to many-well worker throughput.

### OpenHCS checks the workflow before execution

OpenHCS does not simply run a list of Python calls. Before execution, it checks what data exist, what each step consumes, what each step produces, where outputs should go, and which worker process will run the work. These checks move common bioimage failures to the beginning of the run instead of discovering them after hours of processing.

| What OpenHCS checks | Why the user cares | Failure avoided |
|---|---|---|
| Image sources and dimensions | The workflow knows which well, site, channel, timepoint, and z-plane each input represents | Wrong channel, missing image, path-specific assumptions |
| Source bindings | A named image in the workflow points to the intended physical or managed source | Reusing stale images or the wrong acquisition folder |
| Python or imported functions | Each step has a defined callable and expected inputs | Hidden scripts that no longer match the documented workflow |
| Memory/backend requirements | A function receives the array type it was written for | Silent NumPy/GPU conversions or backend-specific surprises |
| Intermediate results | Masks, labels, measurements, relationships, and files are named outputs | Uninspectable module state and stale intermediate files |
| Output destinations | A result can be saved, compared, streamed, or discarded deliberately | Files or viewer outputs that do not match the current run |
| Worker execution | Work runs in the intended process with clear progress and lifetime | Ambiguous process ownership, repeated startup cost, unsafe viewer state |

The same model makes parameters traceable instead of fragile. A threshold, channel name, output setting, or source mapping can be defaulted, inherited, locally changed, or generated from an imported setting. The GUI can show where a value came from and what changed. Generated Python reconstructs the same workflow state, so a visual edit can become reviewable source code instead of an opaque GUI file.

This is the practical value of the underlying state system. The user does not need to think about state machinery to benefit from it. They see that clearing a field means "inherit," entering a value means "override," and generated code explains the current workflow. A wet-lab user can remain in the GUI; a computational collaborator can review Python; both are looking at the same analysis.

### Ordinary Python functions become workflow steps

OpenHCS is not a fixed module catalog. Ordinary Python functions enter the workflow as steps and keep the benefits of parameter editing, validation, viewer output, generated code, and worker execution. Function signatures provide parameter information, and memory annotations declare whether a function expects NumPy, Numba, CuPy, CuCIM, JAX, PyTorch, TensorFlow, pyclesperanto, or another supported backend.

This gives users an escape hatch that does not leave the platform. An imported CellProfiler workflow can be extended with a short Python function that computes an assay-specific quality-control image. A native OpenHCS workflow can mix CPU image processing, GPU-compatible array operations, and table-producing functions when those choices are appropriate for the analysis. The function author writes the scientific operation rather than a separate GUI, serializer, viewer bridge, worker protocol, and benchmark adapter.

Backend-specific algorithms are workflow extensions, not automatically CellProfiler replacements. CellProfiler parity is required when OpenHCS claims to preserve a CellProfiler module. A CuCIM, pyclesperanto, JAX, PyTorch, or TensorFlow function can also be used intentionally as a different analysis method. In both cases, the result remains inside the same workflow record: parameters are visible, intermediates can be inspected, outputs can be materialized, and execution can be scaled.

### Managed stores and viewers stay attached to the workflow

Many microscopy groups already organize data in managed stores such as OMERO. Others work from local disks, Zarr-backed stores, or acquisition folders. OpenHCS treats these as workflow sources rather than side archives. The workflow records which images are being analyzed, how they map to biological and acquisition dimensions, and which outputs were produced from them.

Viewer integration follows the same principle. napari and Fiji outputs are not one-off side effects hidden inside processing code. They are enabled through step configuration. When napari or Fiji streaming is enabled for a step, OpenHCS launches or reuses the viewer on the configured port and streams that step's images while the pipeline runs. The same mask can be saved, compared against a reference, streamed to napari, or sent to Fiji because it remains a named result of the workflow.

This is important for collaboration. A computational user may run the analysis on a workstation or server, while a wet-lab user thinks in projects, images, channels, masks, and measurements. OpenHCS keeps those views connected. CellProfiler-imported workflows preserve their encoded image-loading behavior; native OpenHCS workflows can use explicit local or managed source bindings; napari, Fiji, generated Python, and benchmark outputs remain destinations or views of one workflow instead of separate places where the same experiment is partially duplicated.

The implementation status is concrete. OMERO integration is exercised in CI across multiple versions on each push. Fiji and napari are implemented viewer integrations; Fiji requires a GUI-dependent test environment and is tested outside CI, while napari can be exercised through the available automated/viewer path. The difference is test environment, not conceptual support.

### CellProfiler workflows compile into normal OpenHCS workflows

CellProfiler import validates whether OpenHCS can preserve an existing, mature workflow format rather than only run native examples. A `.cppipe` file contains module settings, image names, object names, measurement expectations, display choices, and output behavior. OpenHCS parses that file, maps image-loading and metadata modules to image sources, maps processing modules to CellProfiler-compatible functions, and stores images, objects, measurements, relationships, grids, and materialized outputs as named workflow results.

| CellProfiler concept | OpenHCS representation | Why this matters |
|---|---|---|
| `.cppipe` file | Compiler dialect input | Trusted workflows enter OpenHCS without manual rewriting |
| Images, Metadata, NamesAndTypes | Image sources and source mappings | Image identity is checked before execution |
| Processing module | OpenHCS workflow step with a CellProfiler-compatible callable | Module behavior runs through the normal workflow model |
| Images and objects | Named image and label outputs | Intermediates can be stored, streamed, compared, or reused |
| Measurements | Named measurement outputs | Tables remain tied to the workflow that produced them |
| Relationships and grids | Named non-image outputs | Object relationships and geometry are preserved explicitly |
| SaveImages and ExportToSpreadsheet | Output destinations and format writers | Files are produced from workflow results, not hidden side effects |
| Module settings | Traceable parameter state | Settings can be edited, inherited, serialized, and audited |

Compatibility is not defined as "the file imports." A pipeline counts only when native CellProfiler outputs and OpenHCS outputs match under the declared comparison policy. Numeric tolerances, label identities, object relationships, measurement rows, and materialized files are compared separately where relevant. Speed is reported after output parity is established.

### OpenHCS reproduces CellProfiler outputs across a broad benchmark corpus

The CellProfiler benchmark corpus is designed to test generality rather than a single curated demonstration. The target set contains `TODO: 33` `.cppipe` workflows: `TODO: 18` official benchmark or example pipelines, `TODO: 7` official tutorial pipelines, and `TODO: 8` public workflows found outside the official example set. Official examples test known CellProfiler semantics, tutorials test user-facing workflows, and public third-party pipelines test whether the importer generalizes beyond examples used during development.

The corpus covers common bioimage and HCS analysis patterns: object identification, object filtering, size and shape measurement, texture measurement, colocalization, image math, illumination correction, grid-based object assignment, object tracking, object-to-image conversion, object overlays, image export, table export, and specialized morphology such as worm untangling. Each workflow is assigned manifest-backed categories such as source category, assay family, semantic pressure, and output pressure. These categories make the benchmark interpretable without relying on filename heuristics.

For each workflow, native CellProfiler produces reference outputs. OpenHCS imports and runs the same `.cppipe` file, then compares the corresponding outputs under the declared equivalence policy. A workflow is counted as passing only when the comparison reports no unresolved differences. The final benchmark table reports the exact pass table, module coverage, unsupported settings or features, speedup, and any tolerated numerical differences.

### OpenHCS is at least 4x faster under constrained single-thread CPU-only conditions

The primary performance benchmark uses the least favorable setting for OpenHCS: one sample, one thread/core, no GPU acceleration, no multiprocessing advantage, and no batching. This condition separates execution speed from additional cores, GPU kernels, larger batch size, and amortized throughput over many wells.

The final figures report execution-only timing separately from total wall time. Execution-only timing measures the part of the run most directly comparable to CellProfiler module execution. Total wall time includes startup, imports, compilation, preparation, and execution. Both numbers matter: execution-only timing tests the analysis engine, while total timing reflects a one-off user run.

Across the final benchmark target, OpenHCS reports the minimum, median, mean, and maximum speedup for constrained single-thread/core execution. The quantitative headline is at least `TODO: 4x` execution speedup on every tested workflow, with stronger average and best-case speedups reported from the final benchmark CSVs. Any workflow below the target remains visible in development reports until optimized or excluded with a source-level reason.

The single-thread CPU-only result is the floor. It shows that trusted CellProfiler-compatible workflows can be preserved and accelerated without invoking GPU acceleration or parallel scaling. Throughput and backend-specific acceleration are additional layers on top of this matched baseline.

### Many-well throughput uses persistent workers

OpenHCS is also evaluated in the way high-content screening is often used: many samples, persistent worker processes, and enough RAM to keep workers alive across repeated work. In that setting, import cost, compile cost, and backend warmup are paid once and divided across many samples.

The throughput model is:

`effective_seconds_per_sample = (worker_startup + compile + warmup) / samples_per_worker + execution_seconds_per_sample + output_seconds_per_sample`

As samples per worker increase, fixed costs become less important and measured throughput approaches steady-state execution. Scaling is reported per resource, not only per wall-clock speedup. Each worker adds CPU capacity but also memory pressure because it may hold imported libraries, compiled kernels, open stores, cached arrays, and intermediate outputs. The throughput figures therefore report samples per hour, worker count, sample count, peak RAM, and approximate RAM per worker.

Small concurrency tests are not overinterpreted. One sample on two workers is not an HCS throughput benchmark because one worker has little or no work to do. More interpretable conditions assign enough samples to each worker for repeated work, such as `1 worker x N samples`, `2 workers x N samples`, and `3 workers x N samples`, where `N` is large enough that startup and warmup no longer dominate.

## Discussion

OpenHCS is built around a simple claim: bioimage workflows need to remain coherent as they move between tools. The system is not defined by any single integration. CellProfiler import, OMERO and Zarr-backed source handling, Fiji and napari inspection, custom Python functions, backend memory conversion, generated Python, reactive parameter state, and persistent workers all depend on the same underlying idea: the workflow remains one record with named sources, parameters, functions, intermediate results, outputs, and execution plans.

This is why compatibility does not become a dead end. A preserved CellProfiler workflow can still be inspected, edited, extended, serialized, streamed to viewers, and executed through workers. A custom Python function can enter the same workflow model. A managed image source can remain attached to the analysis. A backend-specific algorithm can be used when a lab intentionally chooses that method. These are not separate product stories; they are consequences of keeping the workflow coherent across tool boundaries.

The CellProfiler benchmark gives the platform a demanding validation case. It asks whether OpenHCS can preserve a mature external workflow format, reproduce native outputs, and improve execution speed under deliberately constrained conditions. Passing that test does not make OpenHCS only a faster CellProfiler runner. It shows that a trusted legacy workflow can enter a broader platform without losing scientific meaning.

There are limitations. The CellProfiler importer covers the module and setting subset represented in the parity-tested corpus. Some modules have semantics tightly coupled to CellProfiler internals and require explicit compatibility work. Backend-specific algorithms require careful validation when they are claimed as replacements for existing methods. GPU libraries can differ in boundary handling, dtype behavior, reductions, and label semantics. Persistent workers improve throughput while introducing process-lifecycle and RAM tradeoffs. Managed-store deployments also require environment-specific configuration and explicit test evidence.

The field implication is practical. Labs do not have to choose between trusted GUI workflows, managed image stores, interactive viewers, custom Python, reproducible code, and scalable execution. OpenHCS provides a route for keeping those tools while making the workflow itself inspectable, extensible, and faster.

## Methods

### Pipeline object model and compilation

OpenHCS pipelines are built from source definitions, parameter state, workflow steps, and output policies. Before execution, the pipeline is compiled into runtime contexts. Compilation resolves input sources, validates function requirements, determines intermediate and final outputs, assigns runtime paths, and prepares callable execution hooks. The compiled form is the unit submitted to workers.

### Source schema and source binding

Source schemas describe the experimental identity of input data, including dimensions such as well, site, channel, timepoint, and z-plane when present. Source bindings connect those named workflow sources to local files, managed stores, or virtual sources. This lets analysis code refer to the intended image role rather than a one-off path string.

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

The primary benchmark condition uses one sample, one thread/core, no GPU acceleration, and no multiprocessing advantage. Timing is reported as execution-only time and total wall time. Execution-only time isolates module/runtime execution after startup and compile overhead. Total wall time includes startup, imports, compile, preparation, and execution. Throughput benchmarks are reported separately using multi-sample execution and worker-level parallelism.

### Benchmark corpus

The target benchmark corpus contains `TODO: 33` `.cppipe` workflows: `TODO: 18` official benchmark/example pipelines, `TODO: 7` official tutorial pipelines, and `TODO: 8` public third-party workflows. The corpus table lists each workflow, source category, source URL or citation, dataset source, assay family, dominant semantic pressure, output pressure, module coverage, parity status, single-thread/core speedup, and throughput status where available.

### Reproducibility package

The benchmark runner emits the OpenHCS commit, CellProfiler version or commit, Python version, operating system, CPU model, GPU model if used, RAM, pipeline manifest, dataset manifest, checksums where available, native CellProfiler timing, OpenHCS timing, parity reports, and raw CSVs used to generate figures. Benchmark figures are regenerated from saved CSV files without rerunning CellProfiler.

## Conclusion

OpenHCS is a composable bioimage workflow platform. It keeps images, parameters, intermediate results, viewers, Python functions, managed stores, generated code, and worker execution attached to one workflow record. Trusted CellProfiler `.cppipe` pipelines can be imported as normal OpenHCS workflows, executed with native-output parity, inspected through viewer and artifact systems, modified with custom Python functions, serialized as editable code, and scaled across persistent worker processes.

Under one-sample, one-thread/core, CPU-only conditions, imported CellProfiler workflows run at least `TODO: 4x` faster than native CellProfiler execution across the tested corpus. In many-sample settings, persistent workers amortize startup and compilation costs while exposing throughput and RAM tradeoffs directly. The broader contribution is not only speed. OpenHCS gives labs a way to keep the tools they trust while making the workflow itself composable, inspectable, extensible, and executable across modern bioimage environments.

## Draft Figure Captions

### Figure 1. OpenHCS keeps fragmented bioimage workflows as one workflow record

Bioimage workflows often span CellProfiler, Fiji/ImageJ, napari, OMERO, Zarr-backed storage, Python notebooks, exported files, and batch execution. OpenHCS keeps images, metadata, parameters, intermediate results, viewer outputs, generated Python, and workers attached to one workflow record instead of separate tool-specific records.

### Figure 2. OpenHCS checks sources, functions, outputs, and execution before running

Source definitions, parameter state, function requirements, output destinations, and worker execution are resolved before a run starts. This catches common workflow failures such as wrong channels, stale intermediates, hidden parameter copies, incompatible memory backends, and outputs that no longer match the current run.

### Figure 3. Ordinary Python functions and backend-specific algorithms enter the same workflow

A Python function can become a workflow step through signature-derived parameters and declared memory/backend requirements. NumPy, Numba, CuPy, CuCIM, JAX, PyTorch, TensorFlow, and pyclesperanto functions can participate in the same workflow model when their requirements are explicit.

### Figure 4. Step-level viewer output and managed sources stay attached

Imported CellProfiler workflows preserve their encoded loading semantics, while native workflows can use explicit local or managed source bindings. napari and Fiji output are enabled on individual steps; OpenHCS launches or reuses the viewer on the configured port and streams that step's images during execution.

### Figure 5. CellProfiler `.cppipe` files compile into normal OpenHCS workflows

CellProfiler modules and settings are parsed from the `.cppipe` file. Loading and metadata modules define image sources and mappings; processing modules become CellProfiler-compatible workflow steps; images, objects, measurements, relationships, grids, and saved files become named OpenHCS outputs. Native CellProfiler outputs provide the reference for parity comparison.

### Figure 6. Benchmark validation separates output parity, speed, overhead, and throughput

The benchmark manifest feeds native CellProfiler and OpenHCS runs. Native CellProfiler defines reference outputs. OpenHCS imports and runs the same workflows. Output parity, execution timing, total wall time, throughput, RAM, and category summaries remain separate report layers.

### Figure 7. Single-thread CPU speedup and many-sample throughput

The primary speed result uses one sample, one thread/core, no GPU, and no multiprocessing advantage. Throughput figures then show how persistent workers amortize startup and compile costs across samples while reporting worker count, samples per hour, and RAM.

### Figure 8. GUI editing, generated Python, and parameter provenance share one state

The GUI, generated Python, and runtime execution refer to the same workflow state. A parameter can be inherited, locally overridden, cleared back to a parent value, exported as Python, edited, re-imported, and executed without creating a detached script-only workflow.

## Supplementary Table Captions

### Supplementary Table 1. Benchmark corpus

Each row lists one `.cppipe` workflow, source category, source URL or citation, dataset source, assay family, semantic pressure, output pressure, CellProfiler modules used, native CellProfiler runtime, OpenHCS execution runtime, total OpenHCS runtime, speedup, parity status, and notes.

### Supplementary Table 2. CellProfiler module coverage

Each row lists one CellProfiler module class, import status, parity-test status, accelerated path, backend used, unsupported settings or features, and notes.

### Supplementary Table 3. Worker/RAM scaling

Each row lists a throughput condition, pipeline group, sample count, worker count, core count, peak RAM, RAM per worker, total wall time, samples per hour, speedup versus native CellProfiler, and speedup versus OpenHCS single-worker execution.

## Code and Data Availability

OpenHCS source code: `https://github.com/OpenHCSDev/OpenHCS`.

Benchmark scripts, manifests, raw timing CSVs, parity reports, figure-generation scripts, and generated figures: `TODO: repository path or archive DOI`.

CellProfiler pipeline sources and dataset acquisition manifests: `TODO: manifest path`.

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
