# OpenHCS: a composable bioimage workflow platform

**Working draft.** Public adoption, funding, citation, company-use statements, and final integration validation require source/code verification before submission.

## Abstract

High-throughput microscopy can produce biological images faster than they can be analyzed, inspected, and converted into reproducible evidence. OpenHCS is an open-source bioimage workflow platform that keeps source identity, parameters, intermediate results, viewer inspection, custom functions, and worker execution in one executable analysis. Typed source bindings and artifact contracts carry dimensional identity through compilation, process-isolated execution, plate-scoped export, and napari/Fiji review. Imported CellProfiler pipelines provide reference analyses on biological image datasets. Across 30 workflows from official CellProfiler examples, tutorials, and benchmark supplements, OpenHCS reproduced the declared native output values with no unresolved differences. Among 29 workflows with completed native timing, every workflow executed at least 4.03-fold faster (median, 7.39-fold) under one-sample, one-thread/core, CPU-only conditions. GUI edits, generated Python, capability-declared Model Context Protocol (MCP) tools, CellProfiler Analyst-compatible export, viewer outputs, and persistent workers operate on the same public workflow model. OpenHCS preserves established biological analyses while reducing execution time for larger screens, repeated quality-control cycles, and method development.

## Introduction

Image analysis is where microscopy becomes biological evidence. A screen or time-course may begin with automated acquisition, but the experiment becomes interpretable only after cells, organelles, tissues, tracks, intensities, textures, morphologies, and quality-control failures have been identified and reviewed. As acquisition throughput increases, analysis and review often become the slowest and most fragile part of the workflow.

The bottleneck includes runtime, inspection, extension, provenance, and repeated execution. Researchers need to adapt an analysis when an assay changes, inspect intermediate masks when results look wrong, add assay-specific logic, rerun the same workflow across many samples, and explain which parameters produced a table or figure. These needs often span high-throughput batch tools, plugin systems, commercial workflows, open viewers, managed image stores, Python functions, and reproducible execution systems.

Fiji/ImageJ, CellProfiler, Icy, BioImageIT, napari, OMERO, Zarr-backed image storage, Bio-Formats, scientific Python, GPU libraries, and workflow systems each solve distinct parts of biological image analysis and computational reproducibility [Schneider2012; Schindelin2012; Carpenter2006; McQuin2018; deChaumont2012; Prigent2022; Sofroniew2019; Allan2012; Moore2021; BioFormats; vanDerWalt2014; Haase2020; Koster2012; DiTommaso2017; Galaxy2020; Galaxy2024]. The boundary problem appears when an analysis crosses those tools. Parameters are copied between GUIs and scripts, intermediate images are saved without clear provenance, viewers inspect masks that may not correspond to the current run, and custom code loses dimensional context such as channel, timepoint, z-plane, field, or well.

OpenHCS treats the workflow itself as the shared record. The user-facing pipeline remains an ordered list of steps; images, source metadata, parameters, functions, intermediate masks, measurement tables, exported files, viewer streams, and worker execution are named parts of that workflow. Imported workflows remain editable and executable: they can be inspected, inherited, overridden, extended with Python, serialized as generated code, accessed through a capability-declared agent interface, streamed to viewers, compared against references, and rerun through the same state model.

Before a run starts, OpenHCS checks which data exist, what each step consumes and produces, where outputs will go, which memory backend is required, and which worker will execute the work. Declared source bindings, callable and artifact contracts, typed compiled plans, storage backends, memory conversion, a typed runtime-value store, and process-isolated workers implement these checks.

CellProfiler equivalence provides the strictest test of this model. A `.cppipe` file contains image-loading rules, module settings, image names, object names, measurement expectations, display choices, and output behavior. OpenHCS compiles `.cppipe` files into editable workflows, compares their outputs against native CellProfiler, and reports speed only after equivalence is established. Native OpenHCS functions and backend-specific algorithms remain distinct methods unless they are evaluated against their own reference.

Validation spans the complete execution path from source discovery or import through inspection, extension, equivalence testing, and scaled execution. The validation corpus applies official CellProfiler workflows to biological image datasets spanning DNA damage, cell and tissue morphology, Cell Painting, translocation, wound healing, tracking, imaging flow cytometry, yeast screening, and worm phenotyping. The same workflow model supports step-level viewer output, generated Python, custom functions, managed sources, CellProfiler Analyst export, agent-guided authoring and review, and persistent workers.

## Results

### One executable workflow retains analysis state across tools

Official CellProfiler `.cppipe` workflows enter OpenHCS with their image-loading semantics intact. Source images resolve to well, site, channel, z-plane, and timepoint identities. Processing steps produce named images, object labels, measurements, relationships, grids, and files. Selected outputs can be streamed to napari or Fiji during execution, exported as spreadsheets or CellProfiler Analyst-compatible databases, extended with Python functions, inspected through MCP clients, compared against native CellProfiler references, and executed across wells with persistent workers (Figure 1).

OpenHCS records these operations as one ordered workflow and one workflow state. A source binding, GUI edit, revision-checked MCP edit, generated Python file, viewer request, imported CellProfiler module, custom Python function, or worker submission updates or projects the same public declarations. The compiler derives internal source and artifact dependencies from that ordered list; those dependencies are not a second user-authored graph. The workflow therefore retains provenance for image identity, parameter state, intermediate results, output destinations, viewer streams, and execution conditions.

The public executable boundary consists of `PipelineConfig` and a list of `FunctionStep` declarations, whether authored in the GUI, Python, a `.cppipe` import, or an MCP client. Compilation resolves those declarations once into per-step source universes and bindings, callable invocations, artifact producer-consumer edges, memory conversions, execution scopes, materialization targets, and viewer destinations. Workers consume the resulting typed plans and record typed runtime values; they do not reconstruct source, artifact, or grouping semantics from filenames or parameter names.

Compile-time checks expose common setup failures before long runs. Image inputs must resolve to the intended well, site, channel, z-plane, and timepoint. Each imported module or Python function must receive the inputs it expects. CPU and backend requirements must be declared. Output policies must specify whether masks, measurements, files, viewer outputs, and comparison artifacts are kept, written, streamed, compared, or discarded.

Parameter state remains traceable without requiring every option to remain visible. A threshold, channel name, output setting, or source mapping can come from an imported setting, a parent/default configuration, or a local override. Forms display the effective inherited defaults and let users expose only the fields they need to override; clearing an override returns the field to inherited state. The GUI, generated Python, and reflected MCP configuration schema project the same declarations, so a domain expert can remain in the GUI while a computational collaborator reviews generated code or an agent validates a proposed quality-control step.

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
| SaveImages, ExportToSpreadsheet, and ExportToDatabase | Named materialization and terminal plate-scoped export steps | Image files, tables, CPA-compatible SQLite databases, and `.properties` files are produced from typed workflow results, not hidden side effects |
| Module settings | Traceable parameter state | Settings can be edited, inherited, and reviewed in the workflow record |

Preservation is defined by output equivalence under a declared comparison profile. The official-30 manifest uses value-output comparison: exported tables and database values are compared for every workflow, and image artifacts are compared when the native reference contains only images. A workflow passes only when the enabled comparisons report no unresolved differences. Speed is reported only for passing workflows (Figure 2).

The validation corpus contains 22 workflows and associated images from the official CellProfiler 3 example collection, seven workflows from the official CellProfiler tutorial collection, and one workflow from the CellProfiler 4 benchmark supplement [CellProfilerExamples; CellProfilerTutorials; Stirling2021]. These materials were created and distributed by the CellProfiler project and its contributors; OpenHCS uses them as external biological reference analyses. The assays include DNA-damage measurement, human and Drosophila cell morphology, tumor morphology, Cell Painting morphology and quality control, protein translocation, wound healing, time-lapse tracking, imaging flow cytometry, colocalization, positive-cell classification, yeast screening, and *C. elegans* phenotyping. The corresponding operations include segmentation, filtering, intensity and texture measurement, illumination correction, spatial relationships, tracking, image and table export, and specialized worm morphology. Each corpus row retains its official CellProfiler source, workflow, images, assay category, and original citation where available.

Native CellProfiler produced the reference outputs for each workflow. OpenHCS imported and ran the same `.cppipe` file on the same biological images, then compared corresponding output values under the declared equivalence policy. All 30 workflows reached an accuracy fraction of 1.0, with no unresolved differences under the enabled checks. The benchmark artifacts retain the comparison profile, pass table, module coverage, unsupported-feature status, speedup, and tolerated numerical differences.

The current executable compatibility report contains 30 supported `.cppipe` cases, 471 CellProfiler module instances, 58 unique CellProfiler module names, and 5,640 setting rows. All setting rows are covered. The corpus exercises 54 absorbed processing modules plus four source/infrastructure modules, with no missing processing module and no known-invalid absorbed processing module; 32 additional registered processing modules remain outside the tested corpus. Version-specific parity artifacts record the CellProfiler and OpenHCS versions used for each comparison; new CellProfiler versions receive their own declared compatibility checks before output preservation or speed is reported for those versions.

`ExportToDatabase` is translated as an executable terminal step over the same typed measurements and source provenance, rather than as a benchmark-only postprocessing script. The implemented CellProfiler Analyst route writes a self-contained SQLite database and matching `.properties` files with image, object, location, grouping, relationship, channel-display, and optional thumbnail information [Jones2008]. It does not currently generate a CellProfiler Analyst `.workspace` file, support non-SQLite databases, or implement every historical filter and aggregation setting; those requests remain explicit compatibility boundaries.

### Reference-equivalent workflows execute at least fourfold faster

The primary performance benchmark constrains execution to one sample, one thread/core, CPU-only execution, and no batching. This condition separates the CellProfiler-compatible execution path from additional cores, GPU kernels, larger batch size, and amortized throughput over many wells.

The benchmark reports execution-only timing separately from total phase timing. Execution-only timing measures the part of the run most directly comparable to CellProfiler module execution. Total phase timing includes startup, imports, compilation, preparation, and execution, reflecting a one-off user run.

Twenty-nine workflows had completed native execution timings. Their median native execution time was 5.39 s, compared with 0.653 s for OpenHCS. Execution-only speedup was at least 4.03-fold for every one of these workflows, with a median of 7.39-fold. Total phase speedup was lower because startup and compilation are included; the median total phase speedup was 3.42-fold. The remaining wound-healing row equaled the configured 900-s native timeout ceiling without an explicit completion flag and is excluded from timing summaries pending a final benchmark rerun.

The single-thread CPU-only result establishes a performance floor for imported CellProfiler workflows. The measured reduction applies directly to repeated biological-analysis tasks such as screening additional wells, rerunning quality control, and iterating segmentation or measurement settings when execution dominates the analysis cycle. GPU-backed functions and backend-specific algorithms are supported as intentional OpenHCS methods, but they are not part of the CellProfiler-preserving speed result unless separately benchmarked against an appropriate reference.

### Typed sources, viewers, code, and agent access remain attached to the run

Many microscopy groups organize data in managed stores such as OMERO. Others work from local disks, OME-Zarr/NGFF stores, or acquisition folders exported by vendor acquisition software. OpenHCS treats those inputs as workflow sources. `SourceBindingsConfig` is the public authority for declared semantic ingestion: typed filters bound the file universe, metadata rules extract well, site, channel, z-plane, timepoint, or experiment fields, named bindings assign biological roles and component identity, and step-local bindings select the ordered subset required by one step. Compilation records the resolved source universe and binding plan, while runtime image values retain the selected plane provenance. Analysis code therefore refers to an intended image role rather than reparsing a path string.

Microscope and store handlers convert acquisition-specific layouts into analysis-ready image identities before source bindings select semantic inputs. ImageXpress-style exports can place timepoints and z-planes in nested folders rather than encoding every dimension in each filename. Opera Phenix exports can use acquisition-order field indices rather than spatial field order and can omit images when autofocus fails. OpenHCS builds an analysis-ready view of the plate, normalizes field labels, and fills missing Opera Phenix images with black placeholders where the handler can infer the expected grid. Bio-Formats-backed discovery labels CZI, OME-TIFF, and other readable microscopy planes by series, channel, z-plane, and timepoint. A registered OME-Zarr adapter reads declared NGFF image and plate metadata, axes, channel labels, spatial scale, wells, and array addresses. Ambiguous well or site identity requires an explicit binding rather than a guessed label.

Viewer integration follows the typed-output model. napari and Fiji outputs are enabled through step configuration. When streaming is enabled, OpenHCS launches or reuses the viewer, verifies typed readiness, sends outputs with distinct producer and source identities, waits for deferred updates to settle, and checks lifecycle cleanup. The same object-label result can remain available to a later step, materialize as ROI shapes, be compared against a reference, or be streamed for review without being reclassified by its filename or array dtype. Napari exposes typed layer, payload, image-sample, and ROI summaries that can be reconciled with live measurement rows; Fiji participates in the common readiness and settlement protocol but does not expose the same live state-inspection controls.

OpenHCS exposes ordinary Python functions as workflow steps with the same parameter editing, validation, viewer output, generated code, and worker execution used by imported modules. Function signatures expose editable parameters. Memory-backend declarations record whether a function expects arrays from a NumPy-like CPU library, a GPU array library, or a deep-learning framework. An imported CellProfiler workflow can therefore be extended with an assay-specific quality-control function, and a native OpenHCS workflow can mix CPU image processing, GPU-compatible array operations, and table-producing functions when those choices are appropriate for the analysis.

Local agent clients access this model through an MCP server whose tools are projected from registered capability declarations [MCP]. The surface supports source and function discovery, reflected configuration schemas, pipeline authoring and validation, headless compilation and execution, structured progress and artifact inspection, viewer review, and revision-checked attachment to a separately running desktop. Capability metadata owns tool input/output contracts, mutability, side effects, data exposure, and transport availability; clients discover the installed surface rather than relying on a copied tool list. The local stdio server is headless and receives only explicitly granted read/write roots. A separate hosted HTTP surface is narrower and read-only by design and cannot expose a user's local UI, filesystem, or runtime processes.

Each integration preserves a distinct role in the biological analysis and has a corresponding evidence boundary.

| Existing tool or platform | Preserved role | OpenHCS integration | Reported evidence |
|---|---|---|---|
| CellProfiler | Established modular bioimage analysis and `.cppipe` workflows | CellProfiler pipelines run through OpenHCS with loading semantics preserved and named results exposed | 30-workflow value-output equivalence corpus; speed reported only after equivalence |
| CellProfiler Analyst | Interactive exploration and classification of image-derived measurements | Plate-scoped export renders CPA-compatible SQLite tables and `.properties` files from typed image, object, measurement, and relationship artifacts | Focused database/properties projection and materialization tests; no `.workspace` generation or non-SQLite route |
| Fiji/ImageJ | Familiar inspection and image-processing environment | Fiji receives selected step outputs as named workflow results | Implemented viewer destination; GUI/runtime environment requirements reported separately |
| napari | Python-native multidimensional image and ROI inspection | napari receives selected step outputs with producer, source, component, label, and ROI identities attached | Automated readiness, payload, state, ROI, settlement, and cleanup paths; full functional corpus validation is reported separately from headless parity |
| OMERO and BIOMERO-style systems | Managed image storage, collaboration, and launch surfaces | OMERO can provide workflow sources and managed environments can launch OpenHCS workflows | OMERO support tested across multiple versions; launch-surface support should cite exact tested versions |
| Bio-Formats | Broad pixel and metadata readability | Bio-Formats-backed discovery labels images where metadata support safe inference and requires explicit mapping when metadata are ambiguous | Source discovery uses the same normalized image-label model as vendor handlers |
| OME-Zarr/NGFF | Cloud- and object-store-oriented multidimensional image representation | A registered store adapter projects declared plate/image metadata, axes, channels, scale, and array addresses into the source-binding model | Unit and ZMQ integration tests cover source projection and code-mode round trips |
| Microscope vendor folders | Acquisition-specific plate layouts and metadata conventions | Vendor handlers normalize wells, fields, channels, z-planes, timepoints, and missing-image behavior before analysis starts | ImageXpress and Opera Phenix quirks are handled explicitly |
| Python, GPU, and deep-learning libraries | Assay-specific operations and backend-specific methods | Python callables become workflow steps with declared parameters, memory expectations, outputs, viewer policies, and worker execution | Integrated as native methods; CellProfiler equivalence requires separate validation |
| MCP-compatible agent clients | Assisted source onboarding, authoring, execution, and evidence review | Registered capabilities project live schemas and structured operations over local stdio, with a separate restricted hosted boundary | Capability registry, protocol, clean-wheel, UI-bridge, headless-execution, and viewer-inspection tests |
| Generic workflow managers | Scalable and reproducible command execution | OpenHCS supplies bioimage-specific source identity, named outputs, viewer destinations, and parity artifacts inside an executable workflow | Worker benchmarks report throughput, queue depth, and RAM tradeoffs |

### Many-well throughput uses persistent workers

OpenHCS is also evaluated in the way high-content screening is often used: many samples, persistent worker processes, and enough RAM to keep workers alive across repeated work. In that setting, import cost, compile cost, and backend warmup are paid once and divided across many samples rather than paid again for every well.

OpenHCS throughput was measured directly in replicated-well runs over all 30 workflows. Comparisons to CellProfiler were projected by multiplying each workflow's measured native single-sample execution time by the number of wells; native CellProfiler was not rerun as a matched multi-process workload. The projected summaries exclude the unresolved wound-healing timing. Across the remaining 29 workflows, minimum and median execution speedups were 7.22-fold and 9.88-fold with two cores and four wells per core, 10.8-fold and 14.2-fold with three cores and four wells per core, and 12.3-fold and 16.7-fold with four cores and eight wells per core. At four cores, increasing queue depth from one to eight wells per core raised median projected execution speedup from 12.8-fold to 16.7-fold.

Scaling is reported per resource, not only per wall-clock speedup. Each worker adds CPU capacity but also memory pressure because it may hold imported libraries, compiled kernels, open stores, cached arrays, and intermediate outputs. In the four-core queue-depth sweep, median peak RAM increased from 3.98 GB at one well per core to 4.25 GB at eight wells per core, with maximum peak RAM reaching 14.6 GB.

## Discussion

OpenHCS preserves existing bioimage workflows by keeping sources, parameters, intermediate outputs, viewers, generated code, and worker execution in one provenance-tracked workflow. Thirty official CellProfiler workflows applied to their associated biological image data supply the strongest preservation test. Imported `.cppipe` workflows reproduce native CellProfiler outputs under declared equivalence checks and retain named results that can be inspected, extended, serialized, streamed to viewers, and executed through workers.

CellProfiler and OpenHCS both present an ordered sequence of analysis steps; the distinction is not a linear pipeline versus a free-form graph. CellProfiler supplies the established module interface and scientific ecosystem, while OpenHCS keeps the visible editor compact through inherited defaults and makes the resolved consequences inspectable as generated Python, source bindings, callable and artifact contracts, compiled dependencies, materialization targets, and runtime values. Icy Protocols and BioImageIT provide closer comparisons for graphical workflow composition and integrated image-data processing [deChaumont2012; Prigent2022]. OpenHCS differs by making the same public declarations authoritative across its forms, code, CellProfiler import, compiler, workers, viewers, and agent interface.

Compatibility also creates an extension path. A preserved CellProfiler workflow can be modified with an ordinary Python function, routed to napari or Fiji for step-level inspection, exported for CellProfiler Analyst, reviewed as generated Python, authored or inspected through MCP, or run across persistent workers. Native OpenHCS workflows can use the same source, output, viewer, function, agent, and worker mechanisms without requiring a CellProfiler origin. Backend-specific GPU or deep-learning methods remain distinct scientific methods unless a separate parity or validation target is defined for them.

The equivalence result validates OpenHCS as an execution environment for the 30 biological analyses. Native CellProfiler supplies the version-matched reference behavior, while the official workflows and images supply the biological assay context. Assay-specific biological validation remains attached to the originating workflow rather than being redefined by the execution platform.

The measured speedup makes preservation consequential for experiment scale. Screening, longitudinal imaging, parameter refinement, and quality-control reruns repeatedly apply the same analysis to additional images. Reducing execution time without changing the declared outputs increases the analysis that can be completed within a fixed computational window when execution is the limiting phase.

The CellProfiler importer covers the module and setting subset represented in the equivalence-tested corpus. Modules outside that set require explicit validation before preservation is reported. CellProfiler Analyst support is limited to the implemented SQLite/`.properties` route. Bio-Formats and OME-Zarr improve readability, while ambiguous well, site, channel, z-plane, or timepoint identity requires explicit source binding. Viewer integration depends on GUI/runtime environment availability, and local MCP access depends on explicit filesystem grants and an available desktop bridge for UI operations. Persistent workers add process-lifecycle, memory, restart, and cache-warmth tradeoffs. Performance results are therefore separated into execution-only time, total wall time, single-thread CPU speed, measured OpenHCS throughput, and projected comparison to serial native CellProfiler execution.

Output equivalence and constrained execution timing establish a preservation-plus-performance baseline. Under one-sample, one-thread/core, CPU-only conditions, all 29 workflows with completed native timing executed at least 4.03-fold faster after equivalence was established. Persistent workers amortized startup and compilation costs in many-sample runs while exposing throughput and RAM tradeoffs directly. OpenHCS provides a migration path for retaining established CellProfiler analyses while adding provenance, viewer inspection, custom functions, generated code, and scalable execution.

## Online Methods

### Pipeline object model and compilation

The public OpenHCS pipeline consists of a `PipelineConfig` and an ordered list of `FunctionStep` declarations. Before execution, ObjectState-backed configuration is resolved once and the compiler creates a typed `CompiledStepPlan` for each step. These plans record source universes and bindings, ordinary main-flow dependencies, callable invocations, exact artifact input occurrences and outputs, component grouping, memory conversions, execution scope, paths and backends, materialization targets, and enabled viewer destinations. A compiled execution bundle joins the plans to worker assignments and runtime context; workers consume that bundle rather than reconstructing declaration semantics.

### Source binding and source projection

`SourceBindingsConfig` contains typed source filters, metadata extraction rules, named source bindings, grouping metadata, and explicit image-plane sources. `StepSourceBindingsConfig` selects the named views used by one step. The compiler resolves these declarations into a source-universe plan and an ordered binding plan, including component identity and alias-matching rules. Virtual-workspace projection keeps canonical OpenHCS paths separate from backend addresses and optional physical files, and runtime image metadata carries source-plane provenance into downstream artifacts. The former source-schema layer is not part of the current model.

### Microscope handlers and Bio-Formats source discovery

Microscope handlers convert acquisition-specific file layouts into analysis-ready image identities. Vendor-specific handlers encode known quirks such as nested timepoint and z-plane folders, field remapping, metadata sidecars, pixel-size lookup, channel names, and missing-image policy. Bio-Formats-backed discovery emits normalized plane records for broadly readable containers. The OME-Zarr adapter reads declared NGFF plate or image structure, axes, channel labels, pixel scale, and array addresses through a registered store backend. When a handler or store cannot infer a semantic well, site, or channel role safely, OpenHCS requires an explicit source binding rather than silently constructing an ambiguous image set.

### Function registration and memory-backend requirements

OpenHCS functions declare their memory interface through backend annotations for NumPy-like CPU arrays, GPU array libraries, OpenCL-based image-processing libraries, or deep-learning frameworks. These declarations record the expected input and output memory types. During execution, OpenHCS converts image payloads between compatible backends when required. Function signatures expose parameters in the GUI and generated Python representation.

### Runtime outputs and materialization

OpenHCS treats intermediate images, labels, measurements, relationships, grids, tables, metadata, and files as typed artifacts. Callable contracts declare exact semantic inputs and outputs; compilation creates producer-consumer edges and records whether each input occurrence is satisfied by a source binding, ordinary main flow, metadata, or a prior runtime producer. Workers validate and record values in `RuntimeValueStore` under artifact, execution-axis, and component-group identity. Runtime availability is separate from persistence: output policies independently determine whether a value is checkpointed, materialized to a configured store, streamed to napari or Fiji, compared against a reference, or discarded after its consumers finish.

### Viewer and storage backends

Viewer and storage integrations are treated as destinations for workflow outputs or sources for workflow inputs. napari and Fiji receive compiled output routes with separate producer, source, and component metadata. Viewer reuse requires a typed readiness response; completion requires acknowledged transport, settlement of deferred updates, and lifecycle cleanup. Non-persistent napari runs can capture typed layer and payload state before shutdown. OMERO, local disk, memory, Zarr-backed stores, and the read-only OME-Zarr array source participate through registered storage owners rather than backend-name conditionals in the compiler.

### Worker execution

OpenHCS uses process-level worker execution for isolation, progress reporting, viewer separation, and throughput scaling. Persistent workers can amortize startup, imports, compilation, and warmup across repeated samples. Worker-level benchmarks report wall time, execution time, sample count, worker count, and memory use.

### Agent access through MCP

The OpenHCS MCP server projects tools generically from registered `AgentCapabilityDeclaration` classes. Each declaration owns its typed request and result contract, service binding, mutability and side effects, data exposure, security requirements, and allowed transports. Local stdio profiles expose different declared workflow groups for desktop-assisted, headless, authoring, or full development use. UI operations attach to a separately running application through an authenticated, revision-checked bridge; headless authoring and execution do not claim a visible desktop state. Filesystem reads and writes are restricted to configured roots. A separate hosted HTTP projection includes only explicitly opted-in, read-only capabilities and uses isolated server-side workspaces.

### CellProfiler import

CellProfiler `.cppipe` files are parsed into module blocks and settings. Setup modules contribute ordinary source bindings. Processing and export modules resolve through registered CellProfiler module declarations into public OpenHCS workflow steps. During compilation, declaration-owned invocation contracts provide exact images, object labels, measurements, grids, relationships, execution scope, and runtime adaptation while preserving the public callable shown in the GUI, Python, and MCP surfaces.

### CellProfiler Analyst export

Imported `ExportToDatabase` modules execute once per plate after axis-scoped work. Their callable contract selects exact image, object, measurement, relationship, thumbnail, and grouping artifacts from the merged runtime-value store. A typed projection builder constructs CellProfiler Analyst image, object, experiment, and relationship tables; renderers emit a self-contained SQLite database and one or more `.properties` files as one materialized file bundle. Unsupported non-SQLite databases, custom filter rows, `.workspace` generation, and unimplemented historical aggregation settings fail or remain documented compatibility gaps rather than producing inferred side outputs.

### Parity comparison

Native CellProfiler is run for each benchmark workflow to generate reference outputs. OpenHCS then imports and runs the same `.cppipe` file. The official-30 manifest enables value-output comparison. Exported tables and database values are compared for every workflow; image outputs are additionally compared when the native reference profile contains only images. CPA SQLite tables and `.properties` values participate when those outputs are present. Numeric comparisons use declared absolute and relative tolerances; non-numeric identifiers and categorical values are compared exactly unless a specific CellProfiler-compatible normalization is documented. Equivalence is therefore semantic under a declared typed policy, not a claim of universal byte-for-byte identity. Each report records the enabled artifact classes, CellProfiler version or commit, and OpenHCS commit.

### Performance benchmarking

The primary benchmark condition uses one sample, one thread/core, CPU-only execution, and no batching. Timing is reported as execution-only time and total wall time. Execution-only time isolates module/runtime execution after startup and compile overhead. Total wall time includes startup, imports, compile, preparation, and execution. The wound-healing native duration equals the configured 900-s timeout ceiling, while the summary row lacks an explicit completion flag. This row is excluded from timing statistics pending a final rerun with explicit completion or censoring metadata. Throughput benchmarks are reported separately using multi-sample execution and worker-level parallelism. GPU-backed functions are reported only as OpenHCS method support unless a separate benchmark defines the GPU hardware, backend version, reference behavior, and comparison policy.

The benchmark runs use local or explicitly mounted paths recorded in the benchmark manifest. Cloud-bursting or network-filesystem-latency performance requires a benchmark that records the storage environment, path form, cache state, and worker placement.

### Benchmark corpus

The corpus contains 22 workflows and associated image sets from the official CellProfiler 3 examples repository, seven workflows and image sets from the official CellProfiler tutorials repository, and one workflow from the supplement to the CellProfiler 4 performance study [CellProfilerExamples; CellProfilerTutorials; Stirling2021]. The CellProfiler project and the cited dataset contributors retain authorship and provenance for these materials. The corpus table lists each workflow, official source URL, immutable source revision for the final benchmark release, original data citation where available, assay family, image-analysis behavior, output files, module coverage, equivalence status, single-thread/core speedup, and throughput status.

The executable acquisition declarations pin the official CellProfiler examples source to `4972b59e670a4ae96c3d453803c92eeff378d054`, the CellProfiler tutorials source to `264a8155da21a2d468051f78211bed2e580a8934`, and the CellProfiler 4 benchmark supplement to `40abc2e600fd46b74c213999dd25c5245048dc92`.

### Reproducibility package

The benchmark runner emits the OpenHCS commit, CellProfiler version or commit, Python version, operating system, CPU model, GPU model if used, RAM, pipeline manifest, dataset manifest, checksums where available, native CellProfiler timing, OpenHCS timing, parity reports, and raw CSVs used to generate figures. Benchmark figures are regenerated from saved CSV files without rerunning CellProfiler.

## Draft Figure Captions

### Figure 1. Sources, tools, agents, viewers, code, and workers remain one workflow

Automated microscopy produces wells, sites, channels, z-planes, and timepoints that become masks, measurements, quality-control decisions, reruns, and output tables. OpenHCS keeps typed source bindings, parameters, named artifacts, CPA exports, viewer destinations, generated code, MCP operations, and worker execution attached to one public runnable workflow. Diagram source: `paper/figures/rendered/fig01_field_integration_gap.svg`.

### Figure 2. Official CellProfiler biological workflows are preserved and accelerated

Representative official CellProfiler biological images, native outputs, OpenHCS outputs, and difference views make the preservation target visible. The examples include at least one cellular assay and one morphologically distinct assay such as wound healing, yeast, or worm phenotyping. These image panels require selected workflows to be rerun with image-output comparison enabled; they do not infer image equivalence from the value-output corpus result. The benchmark manifest feeds native CellProfiler and OpenHCS runs. OpenHCS parses `.cppipe` files into image sources, image-name-to-file matching, CellProfiler-compatible workflow steps, named outputs, and saved results. Quantitative panels separate output equivalence, module coverage, constrained one-sample CPU-only speedup, cold-run overhead, measured OpenHCS throughput, projected native comparison, and RAM scaling. Every image panel credits the official CellProfiler source and original dataset citation. Diagram sources: `paper/figures/rendered/fig03_cellprofiler_import_path.svg` and `paper/figures/rendered/fig04_benchmark_validation_structure.svg`.

### Figure 3. GUI edits, generated Python, agent operations, and Python functions modify the same workflow

The pipeline editor, generated Python representation, reflected MCP schemas and revision-checked operations, source bindings, step-level napari/Fiji output, ordinary Python functions, backend-specific functions, and worker execution all target the same public declarations. Diagram sources: `paper/figures/rendered/fig07_typed_state_bidirectional_editing.svg` and `paper/figures/rendered/fig06_backend_extensibility.svg`.

## Supplementary Table Captions

### Supplementary Table 1. Benchmark corpus

Each row lists one `.cppipe` workflow, official CellProfiler source collection, immutable source revision, source URL, original dataset citation where available, assay family, image-analysis behavior, output files, CellProfiler modules, native CellProfiler runtime or timeout status, OpenHCS execution runtime, total OpenHCS runtime, speedup, equivalence status, and notes. Equivalence, timing, and timeout status remain separate columns.

### Supplementary Table 2. CellProfiler module coverage

Each row lists one registered CellProfiler module class, whether the 30-workflow corpus exercises it, its source/infrastructure or processing role, importability, declared contract and backend, unsupported settings or features, and notes. Processing modules outside the corpus remain explicitly untested rather than being inferred covered from family similarity.

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

The OpenHCS benchmark manifest records acquisition paths and maps every workflow to its original source. Its executable acquisition declarations pin immutable revisions for all three source collections; the archived benchmark release additionally retains checksums, source licenses, and original dataset citations.

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
- `[Jones2008]` Jones et al. CellProfiler Analyst: data exploration and analysis software for complex image-based screens. BMC Bioinformatics 9, 482 (2008). DOI: 10.1186/1471-2105-9-482.
- `[MCP]` Model Context Protocol contributors. Model Context Protocol specification. `https://modelcontextprotocol.io/specification/`.
- `[CellProfilerExamples]` CellProfiler contributors. CellProfiler example pipelines and associated biological images. `https://github.com/CellProfiler/examples`, revision `4972b59e670a4ae96c3d453803c92eeff378d054`.
- `[CellProfilerTutorials]` CellProfiler contributors. CellProfiler tutorial pipelines and associated biological images. `https://github.com/CellProfiler/tutorials`, revision `264a8155da21a2d468051f78211bed2e580a8934`.
- `[Schneider2012]` Schneider, Rasband and Eliceiri. NIH Image to ImageJ: 25 years of image analysis. Nature Methods 9, 671-675 (2012). DOI: 10.1038/nmeth.2089.
- `[Schindelin2012]` Schindelin et al. Fiji: an open-source platform for biological-image analysis. Nature Methods 9, 676-682 (2012). DOI: 10.1038/nmeth.2019.
- `[deChaumont2012]` de Chaumont et al. Icy: an open bioimage informatics platform for extended reproducible research. Nature Methods 9, 690-696 (2012). DOI: 10.1038/nmeth.2075.
- `[Prigent2022]` Prigent et al. BioImageIT: Open-source framework for integration of image data management with analysis. Nature Methods 19, 1328-1330 (2022). DOI: 10.1038/s41592-022-01642-9.
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
