# OpenHCS: preserving CellProfiler workflows while unifying the high-content screening ecosystem

**Working draft.** Numbers marked `TODO` should be replaced only from final benchmark artifacts. Public adoption, funding, citation, and company-use statements require source verification before submission.

## Plain-Language Summary

Many biology labs already have working CellProfiler pipelines. Those pipelines are not just files; they contain years of practical decisions about how to identify cells, measure objects, export tables, and trust results. OpenHCS keeps those workflows meaningful while bringing them into a modern high-content screening environment. The same analysis can run faster, show intermediate masks and images, connect to viewers such as napari and Fiji, use managed image stores such as OMERO or OME-Zarr, combine with custom Python functions, and scale across many wells. The GUI, generated Python, runtime state, and LLM-assisted editing all refer to the same workflow. A biologist can keep the analysis they already trust, while OpenHCS handles the engineering needed for modern HCS: organized inputs, inspectable outputs, reproducible edits, worker processes, and future CPU/GPU backends.

## Abstract

High-content screening has outgrown the software model that made it accessible. Modern screens combine wells, fields of view, channels, z-planes, timepoints, segmentation masks, measurement tables, viewers, storage systems, legacy analysis formats, custom functions, and heterogeneous compute backends. Existing tools remain essential: CellProfiler provides trusted modular image-analysis pipelines, Fiji/ImageJ provides a deep microscopy plugin ecosystem, napari provides interactive multidimensional viewing, OMERO and OME-Zarr provide data-management and storage conventions, and CPU/GPU libraries provide fast numerical computation. Their strengths usually live in separate systems. OpenHCS brings these roles into one workflow environment. A pipeline is checked before execution, functions describe the kind of image data they expect, parameters remain editable and traceable, and images, masks, measurements, files, and viewer outputs are treated as named workflow products rather than hidden side effects. Managed image stores can become workflow sources, generated Python can become a readable and re-importable workflow representation, and LLM assistance can operate over the real function registry and typed state rather than detached scripts. OpenHCS imports CellProfiler `.cppipe` workflows as normal OpenHCS workflows and executes them with CellProfiler-output parity. In a deliberately constrained CPU-only benchmark using one sample, one core, no GPU, and no multiprocessing advantage, OpenHCS achieved at least `TODO: 4x` execution speedup across `TODO: 33` CellProfiler pipelines, spanning `TODO: 18` official examples, `TODO: 7` official tutorials, and `TODO: 8` public third-party workflows. Imported workflows can then be inspected in viewers, modified with custom Python functions, serialized as editable code, and scaled across persistent workers. OpenHCS protects trusted legacy workflows while making them part of a modern HCS platform.

## Introduction

High-content screening is now a software problem as much as an imaging problem. The experimental unit is no longer one image or one folder. A typical experiment contains wells, fields of view, channels, z-planes, timepoints, treatments, replicates, segmentation masks, measurement tables, quality-control images, viewer overlays, and intermediate results that need to be inspected or regenerated. The scientific question is biological, but the analysis works only when all of these moving parts stay aligned.

The field has strong tools because earlier generations of bioimage software solved real problems. ImageJ and Fiji made biological image processing extensible and widely available through plugins, macros, and a large user community [Schneider2012; Schindelin2012]. CellProfiler made image-analysis pipelines modular and reproducible for biologists, especially for tasks such as illumination correction, object identification, object measurement, and spreadsheet export [Carpenter2006; McQuin2018]. OMERO gave microscopy groups a serious system for storing, organizing, and sharing imaging data and metadata [Allan2012]. napari brought modern Python-based multidimensional viewing, making it easier to inspect images, masks, points, and overlays during analysis [Sofroniew2019]. scikit-image, SciPy, OpenCV, and related libraries provide much of the shared numerical foundation used by scientific Python image analysis [vanDerWalt2014]. GPU and accelerator libraries such as CuPy, JAX, PyTorch, TensorFlow, pyclesperanto, and CuCIM provide faster computation for workloads that can use them [Haase2020; CuPy; JAX; PyTorch; TensorFlow; CuCIM]. Workflow systems such as Snakemake, Nextflow, and Galaxy provide reproducible orchestration for many computational domains [Koster2012; DiTommaso2017; Galaxy2020; Galaxy2024]. Modern HCS depends on all of these contributions. Their separation is the limitation.

The separation becomes visible during ordinary lab work. A CellProfiler pipeline can define a trusted segmentation and measurement protocol while the user also needs napari inspection, Fiji-compatible ROI outputs, one custom Python function, Zarr-backed intermediates, multi-worker plate execution, and traceable parameter changes. Each handoff is manageable once. Across a full screen, the handoffs become the workflow.

Table 1 summarizes the software roles that high-content screening workflows commonly need and the integration gap that motivates OpenHCS.

**Table 1. Field software strengths and the integration boundary addressed by OpenHCS.**

| System or ecosystem | Field role | What it gives HCS users | Boundary that remains in practice | OpenHCS integration point |
|---|---|---|---|---|
| CellProfiler [Carpenter2006; McQuin2018] | Modular biological image-analysis pipelines | Trusted `.cppipe` workflows, object measurement, table export, GUI-accessible analysis | Pipeline semantics are tied to the CellProfiler runtime and are hard to compose with custom execution backends | `.cppipe` compiler dialect, CellProfiler-compatible runtime adapter, parity comparison |
| ImageJ/Fiji [Schneider2012; Schindelin2012] | Extensible microscopy image processing | Plugins, ROI conventions, interactive inspection, broad lab familiarity | Fiji outputs and interactive inspection often sit outside batch HCS runtime state | PolyStore streaming backend, ImageJ-compatible ROI/artifact outputs |
| napari [Sofroniew2019] | Python-native multidimensional viewing | Interactive image, label, and point visualization in scientific Python | Viewer state and processing state are usually connected manually | Process-isolated streaming backend through PolyStore/ZMQRuntime |
| OMERO/OME [Allan2012] | Microscopy data management and metadata | Server-side image storage, metadata, sharing, data model conventions | Analysis execution and interactive pipeline state are separate from data management | Source schema, VFS/runtime store abstraction, ZMQRuntime-compatible remote execution path |
| OME-Zarr/Zarr [Moore2021] | Chunked multidimensional image storage | Scalable array storage and cloud/HPC-friendly layout | Storage layout does not by itself define pipeline semantics or function contracts | PolyStore backend selected by runtime artifact and materialization plans |
| scikit-image/SciPy/OpenCV [vanDerWalt2014] | CPU image-processing foundations | Mature numerical and image-processing algorithms | Function calls lack HCS dimensional, provenance, and artifact semantics by default | Decorated functions become FunctionSteps with dimensional and memory contracts |
| CuPy/CuCIM/JAX/PyTorch/TensorFlow/pyclesperanto [Haase2020; CuPy; CuCIM; JAX; PyTorch; TensorFlow] | GPU and accelerator backends | Fast array operations, deep-learning models, OpenCL/CUDA image kernels | Backend transitions require framework-specific glue and parity-sensitive dtype/boundary handling | ArrayBridge conversion and compiler-selected backend variants |
| Snakemake/Nextflow/Galaxy and related workflow systems [Koster2012; DiTommaso2017; Galaxy2020; Galaxy2024] | General workflow orchestration | Reproducible task graphs, batch execution, HPC/cloud compatibility | They orchestrate tasks but do not define HCS image/object/measurement semantics | OpenHCS produces typed HCS execution units that can be run locally, remotely, or through workers |

OpenHCS is the layer that lets these systems work together without losing workflow meaning. A pipeline is treated as an executable scientific protocol, not just a list of buttons or script calls. Images keep their experimental identity: well, site, channel, timepoint, z-plane, and source. Masks, object labels, measurement tables, grids, exported files, and viewer overlays are named workflow products. Parameters remain editable values with a known source: defaulted, inherited, locally changed, or generated from imported CellProfiler settings. Under the hood, OpenHCS uses typed state, function contracts, storage backends, memory backends, and worker processes. The user-facing point is simpler: the same workflow can be inspected, edited, serialized, benchmarked, and executed without becoming several disconnected versions of itself.

The structure matches HCS workflow practice. A user can begin from a known CellProfiler pipeline, replace one module with a custom Python function, inspect intermediate images in napari, stream ROIs to Fiji, change a parameter in the GUI, export the current workflow as Python, edit it in an IDE, re-import it, and run the same analysis across many wells with multiprocessing or GPU acceleration. In most systems, those actions cross tool boundaries and semantic boundaries. In OpenHCS, they remain operations on the same underlying pipeline object.

CellProfiler compatibility grounds the architecture in existing practice. A `.cppipe` workflow carries module settings, image semantics, object relationships, measurement tables, display choices, and years of biological trust. OpenHCS implements CellProfiler import as a compiler dialect. A CellProfiler pipeline is parsed, compiled into normal OpenHCS pipeline structure, connected to OpenHCS runtime artifacts, executed through normal FunctionStep machinery, and checked against native CellProfiler output.

The benchmark uses the most constrained OpenHCS condition: one sample, one CPU core, no GPU acceleration, no multiprocessing, and no batching. Startup, compilation, warmup, and execution timing are separated. Native CellProfiler produces the reference outputs. OpenHCS imports the same `.cppipe` files and reproduces those outputs under declared tolerances. The same scientific workflow semantics run faster inside the OpenHCS execution model.

The same imported workflow then enters the rest of OpenHCS: typed state management, bidirectional GUI-code workflows, custom function insertion, multi-backend memory conversion, VFS-backed runtime artifacts, viewer streaming, multiprocessing, persistent workers, and GPU backend selection.

## Results

### A familiar CellProfiler workflow becomes an editable OpenHCS workflow

OpenHCS begins from a familiar object: a CellProfiler `.cppipe` file. The pipeline is parsed into modules and settings, then compiled into OpenHCS pipeline structure. Image-loading and metadata modules become source schema and source-binding configuration. Processing modules become FunctionSteps. CellProfiler images, objects, measurements, relationships, and materialized files become OpenHCS runtime artifacts. The imported workflow remains a CellProfiler-compatible analysis protocol while entering the OpenHCS object model.

The imported workflow is editable from several surfaces. A user can inspect the imported steps in the GUI, change a setting, clear a local override to inherit a parent value, add a custom Python function, stream an intermediate image to napari or Fiji, export the workflow as Python, edit the generated code, and re-import it. The operations act on the same compiled pipeline semantics. They do not create separate GUI, script, viewer, and batch-runner versions of the analysis.

The same workflow can then run in several execution modes. A one-sample run gives the most direct comparison with native CellProfiler. A many-well run assigns work to persistent worker processes. A viewer run streams selected artifacts to napari or Fiji. A future backend-selected run can resolve compatible functions to NumPy/Numba, CuPy/CuCIM, pyclesperanto, or JAX variants at compile time. After import, the CellProfiler workflow becomes a normal OpenHCS workflow with CellProfiler-compatible semantics.

The value is easiest to see as an ordinary lab story. A user receives a published `.cppipe` file for nuclei segmentation and texture measurement. They import it, run the native CellProfiler comparison once, and confirm that the OpenHCS workflow reproduces the expected outputs. They inspect the accepted-object mask in napari, notice that one channel needs a different illumination correction, clear a local threshold override so it inherits the plate-level value, and insert a short Python function that computes an assay-specific quality-control image. The modified workflow can be exported as Python for review, re-imported by another user, and run across many wells without creating a separate GUI version, script version, viewer version, and batch version of the analysis.

In that story, speed is only one consequence. The more important change is that the workflow keeps its identity while moving through different surfaces. The `.cppipe` file is not discarded. The GUI is not a separate front end over hidden state. Python is not a detached reproduction of a clicked workflow. napari and Fiji are not manual side effects. Each surface edits, inspects, or executes the same underlying pipeline object.

### OpenHCS checks workflow meaning before the run starts

Before running an analysis, OpenHCS checks the meaning of the workflow. It does not simply order Python calls. It determines which images exist, which wells, sites, channels, timepoints, and z-planes identify them, which functions can use them, which kind of array memory is required, which images or masks or tables will be produced, which outputs should be saved or streamed to a viewer, and which worker process can execute the work.

This distinction matters because HCS failures often come from meaning rather than syntax. A script can run while silently using the wrong channel, overwriting an intermediate image, mixing object scopes, exporting a stale table, or sending a viewer overlay that no longer corresponds to the current parameters. OpenHCS moves those concerns into explicit workflow objects that are checked together.

| Workflow object | Question answered before execution | Typical failure avoided |
|---|---|---|
| Source schema | Which wells, sites, channels, timepoints, and z-planes exist? | Channel swaps, missing files, path-specific assumptions |
| Source binding | Which physical or virtual payload corresponds to each semantic image? | Accidental reuse of stale images or wrong acquisition folders |
| FunctionStep | Which callable or callable chain is executed at this point? | Hidden scripts that no longer match the documented workflow |
| Processing contract | Does the function consume a plane, stack, object set, table, or artifact? | Applying 2D logic to stacked or grouped data incorrectly |
| Memory contract | Which backend memory type does the callable require and return? | Implicit NumPy/GPU conversions and backend-specific surprises |
| Runtime artifact | What image, label set, relationship, measurement, grid, or file is produced? | Hidden module state and uninspectable intermediates |
| Materialization plan | Which artifacts are saved, streamed, compared, or discarded? | Output files that do not correspond to the executed workflow |
| Worker context | Where does the work run and what state is reused? | Repeated startup cost, unsafe viewer state, ambiguous process ownership |

The checked workflow becomes a contract between the biological analysis and the execution system. It keeps high-level assay intent connected to low-level execution details without forcing the user to manage those details by hand.

### OpenHCS keeps image analysis, measurements, viewers, and code in one workflow

OpenHCS uses a check-then-run architecture. Pipeline construction is interactive and editable, but execution begins only after the platform has resolved the pipeline into a validated run plan. That plan includes input paths, image identity, function compatibility, memory compatibility, artifact production, output destinations, and worker setup. Errors that would otherwise appear after hours of processing are moved to the front of the run.

The process resembles preparing an experiment rather than launching a loose script. OpenHCS knows which sources exist, which outputs will be produced, which functions can accept which array memory types, which results need to be saved or streamed, and which workers can execute the plan. The user still thinks in biological workflow terms: load the images, correct illumination, identify objects, measure features, export tables, inspect masks. The system carries the dimensional and runtime bookkeeping.

The central processing unit is the FunctionStep: one function, or a small chain of functions, placed into the workflow. Each function describes the type of data it expects. A NumPy function, a CuPy function, a pyclesperanto function, a PyTorch model, a JAX function, or a custom lab function can appear in the same pipeline when its declared requirements match the surrounding workflow. ArrayBridge handles array conversion, making backend transitions explicit and inspectable.

Dimensional structure is also part of the execution model. HCS images are grouped by experimental axes such as well, site, channel, timepoint, and z-plane. OpenHCS separates the function's local array behavior from the experiment-level grouping behavior. A function that operates on a 2D plane can be applied across planes; a function that consumes a stack can receive the stack; a function that emits measurement rows can declare those rows as runtime artifacts. Implicit dimensional bookkeeping moves out of scripts and into validated runtime structure.

The pipeline representation can be manipulated from several interfaces without changing meaning. The GUI edits the same objects that Python code executes. Generated Python reconstructs the same pipeline state. Custom functions enter through the same memory decorators and signature introspection as built-in functions. LLM-assisted pipeline generation is constrained by the actual function registry and type signatures rather than free-form text generation.

Editable workflows remain reproducible because every entry point targets the same compiled object. A wet-lab user can stay in the GUI. A computational user can edit Python. A methods developer can register new functions. An LLM assistant can search the actual callable registry. The entry points converge on one compiled object rather than producing parallel workflow formats.

### Parameters become traceable instead of fragile

OpenHCS configuration is typed runtime state exposed through the GUI. It is built on hierarchical resolution, provenance, dirty tracking, and code generation. A configuration value can be unset at a local scope and inherited from a parent scope. The UI can show the inherited value, show where it came from, and update related windows when a parent value changes. Clearing a field means "inherit"; entering a value means "override." The user interaction matches the data model.

HCS pipelines are rarely flat lists of parameters. Defaults apply across plates, pipelines, steps, and individual function invocations. Function parameters are extracted from Python signatures and participate in the same state system as dataclass configuration objects. A lab can register a function and immediately receive a configurable UI without manually building a parameter form. The same state can be serialized as Python, reviewed in version control, edited by hand, and re-imported.

The UI layer is an editing surface over the same typed objects used by the runtime. Cross-window updates, inheritance previews, saved/live state separation, undo/redo behavior, and generated code make complex HCS workflows visible and editable without breaking executable structure.

Workflow reuse depends on knowing what changed. A pipeline imported for one assay can receive a different threshold method, measurement export, channel mapping, or viewer output in the next assay. In a flat settings file, those changes are simple to make and costly to audit. In OpenHCS, the changed value has a scope, a type, and a provenance path. The platform can show whether a value is local, inherited, defaulted, or user-modified.

The same state model also gives LLM-assisted workflows a concrete boundary. The assistant does not invent an analysis by writing arbitrary script text. It can operate over available functions, signatures, dataclass fields, memory decorators, and pipeline objects. Suggested edits still become typed state and compiled FunctionSteps. This makes the assistant another editing surface over the workflow rather than a second untyped automation system.

### The UI is a teaching surface, not only a control panel

OpenHCS uses the user interface to teach the workflow structure. Many biologists learn image analysis by clicking through modules, changing thresholds, inspecting masks, and comparing outputs. That style of learning is powerful, but it becomes fragile when the GUI hides where a value came from or when the clicked workflow cannot be translated into readable code. OpenHCS keeps the interactive workflow and the executable workflow synchronized.

The interface can show a parameter as inherited, defaulted, locally changed, or generated from an imported CellProfiler setting. That makes the workflow behave more like a versioned object than a flat form. A user can see which values are unchanged, which values were edited for a specific step, which values came from a parent scope, and which edits have not yet been applied to a saved or generated representation. This gives wet-lab users a practical mental model for workflow state: not only "what is the current value?", but "why is this the current value, where did it come from, and what changed?"

Bidirectional GUI-code conversion adds another teaching layer. A user can build or modify a workflow in the GUI, export it as Python, and read the generated source as an explanation of the workflow. A computational user can edit the Python and re-import it without creating a separate script-only pipeline. The code is not merely an export format; it is a pedagogical bridge between visual workflow editing and reproducible programmatic analysis.

This matters for collaboration. A wet-lab scientist can send a generated Python workflow to a computational collaborator. The collaborator can inspect the actual functions, parameters, and scopes rather than reverse-engineering a screenshot or a saved GUI file. The edited workflow can return to the GUI with the same structure intact. In practice, OpenHCS makes the workflow teachable: values have provenance, edits have scope, code has a path back to the UI, and runtime products remain connected to the parameters that produced them.

### Intermediate images, masks, and tables become named workflow products

Many image-analysis systems expose final outputs but hide the intermediate objects that make those outputs scientifically interpretable. In HCS, the intermediate state is often the evidence: corrected images, threshold masks, accepted objects, rejected objects, parent-child relationships, outlines, grids, quality-control images, and measurement rows. OpenHCS treats these as runtime artifacts rather than incidental values inside a module call.

A runtime artifact has a semantic role, a scope, and a destination policy. It can be kept in memory, written to disk, stored in Zarr, streamed to napari, exported to Fiji-compatible outputs, compared against a native CellProfiler reference, or discarded after execution. The same artifact does not need separate code paths for saving, viewing, benchmarking, and debugging. Those are materialization choices over the same produced value.

This is the difference between "a module saved a file" and "the workflow produced an object label artifact that was materialized as an image, compared for parity, and streamed for inspection." The latter is slower to say but much safer to build on. It gives downstream tools a stable object to request and gives the benchmark a stable object to compare.

| Runtime artifact class | Examples | Why it matters qualitatively |
|---|---|---|
| Image artifact | corrected image, enhanced image, object-to-image output | Intermediate image state can be inspected and reused |
| Label artifact | nuclei, cells, worms, masks, accepted/rejected objects | Segmentation is a typed output, not only pixels in a display |
| Measurement artifact | image measurements, object measurements, table rows | Numeric facts remain tied to the workflow that produced them |
| Relationship artifact | parent-child objects, neighbor relationships, tracking links | Non-image biological structure is preserved explicitly |
| Grid or geometry artifact | plate grid, object outlines, ROIs, overlays | Viewer and export outputs share a semantic source |
| Materialized file artifact | CSV, JSON, PNG/TIFF, ROI-like output | Files are products of the artifact system, not hidden side effects |

### OpenHCS unifies the tools biologists already use

OpenHCS is built from reusable scientific Python libraries, but the user-facing goal is straightforward: a biologist should not need a different workflow record for every tool. The image viewer, saved output, measurement table, Python function, imported CellProfiler module, and worker process should all refer to the same analysis state.

Each lower-level library handles one piece of that problem. ObjectState tracks parameter values, inheritance, local overrides, and provenance [ObjectState]. ArrayBridge moves arrays between NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto when a function needs a different memory backend [ArrayBridge]. PolyStore treats disk, memory, Zarr, napari, and Fiji as destinations for workflow products [PolyStore]. ZMQRuntime keeps workers and viewers in separate processes while still reporting progress [ZMQRuntime]. pyqt-reactive provides the live desktop forms [pyqtReactive]. pycodify turns workflow objects into editable Python source [pycodify]. python-introspect reads function signatures so custom functions can become configurable workflow steps [pythonIntrospect]. metaclass-registry lets backends and adapters register themselves when their classes are defined [metaclassRegistry].

Together, these pieces let OpenHCS be modular without feeling fragmented. The platform is a domain assembly of reusable components: parameter state, image functions, storage destinations, memory conversion, workflow products, viewer streaming, and process-isolated execution. This decomposition makes CellProfiler import possible without making CellProfiler special. A `.cppipe` pipeline becomes one dialect that enters normal OpenHCS workflow structure.

Viewer integration follows the same principle. napari and Fiji are not treated as one-off side effects hidden inside processing functions. They are streaming backends. A pipeline can save data to disk, stage it in memory, write it to Zarr, or stream it to a viewer through the same storage abstraction. Intermediate results, ROI outputs, and quality-control images can be inspected during or after execution without changing the analysis function itself.

CellProfiler, Fiji, napari, OMERO, Zarr, and custom Python functions keep their natural roles rather than being forced into one lowest-common-denominator interface. OpenHCS supplies the layer that lets those roles compose: pipeline semantics, state semantics, memory semantics, artifact semantics, and worker semantics.

This design avoids making integration synonymous with import/export. Import/export integration can move data between tools, but it usually cannot say whether a parameter, object label, image plane, measurement table, viewer overlay, and worker process all refer to the same workflow state. OpenHCS integration is semantic integration. Tools remain recognizable, but their outputs and inputs become typed parts of a shared execution model.

### OMERO and institutional image stores become workflow sources, not side archives

Many microscopy groups already use OMERO or OME-oriented storage to organize images, metadata, users, projects, and shared datasets. In ordinary analysis workflows, the data-management system often sits beside the analysis environment. Images are downloaded or mounted, processed elsewhere, and exported results are copied back or stored separately. That pattern works, but it separates the institutional source of truth from the analysis state.

OpenHCS treats managed image stores as workflow sources and artifact destinations. The source schema records the experimental identity of the data: plate, well, site, channel, timepoint, z-plane, and metadata fields. The runtime store and VFS layers let analysis code refer to those sources through OpenHCS semantics rather than through one-off local paths. This makes OMERO-style integration more than file access. The workflow can know that an image came from a managed dataset, which semantic channel it represents, which analysis artifacts were produced from it, and which outputs should be streamed, saved, compared, or returned to a managed storage location.

This is important for shared lab environments. A computational user may run the analysis on a workstation or server, while a wet-lab user thinks in projects, plates, wells, images, and measurements. OpenHCS keeps those views connected. OMERO, OME-Zarr, local disk, napari, Fiji, and benchmark outputs become destinations and sources in one workflow model rather than separate places where the same experiment is partially duplicated.

### LLM assistance is constrained by the real workflow model

OpenHCS also changes what AI assistance can safely mean in scientific image analysis. A general chatbot can suggest code, but free-form generated scripts are difficult to trust when they are detached from the actual pipeline, available functions, parameter types, memory backends, and runtime artifacts. OpenHCS gives an assistant a constrained substrate: the real function registry, real Python signatures, typed configuration objects, memory decorators, source schemas, and current workflow state.

An assistant can therefore help a user search available functions, explain what a parameter does, draft a custom function, suggest where a quality-control step should be inserted, or translate a GUI-edited workflow into readable Python. The result still becomes typed state and FunctionSteps that the normal OpenHCS compiler checks before execution. The assistant does not replace the workflow model; it operates through it.

This matters pedagogically as well as practically. A user can ask why a workflow is using a particular threshold, where an inherited value came from, what changed between two workflow states, or how a generated Python block corresponds to the GUI. The answer can refer to actual workflow objects rather than generic image-analysis advice. In this form, LLM support is not a separate automation layer. It is a guided interface to the same inspectable workflow state used by the GUI, code generator, runtime, and benchmark.

### Imported workflows support several lab-facing use cases

CellProfiler import preserves familiar workflows while expanding their execution context. A lab can import a published `.cppipe` file, run it against native CellProfiler for parity, and keep using the same analysis without rewriting it. The imported workflow can then be inspected step-by-step, modified in the GUI, exported as Python, or combined with new Python functions written for a particular assay.

The same workflow can support quality control. Intermediate images, labels, overlays, and measurement tables become runtime artifacts rather than hidden module state. A failed segmentation can be streamed to napari, ROI-style artifacts can be sent to Fiji-compatible outputs, and measurement changes can be traced to the setting or function that produced them.

The same workflow can also support scale. A one-sample run provides the matched comparison with CellProfiler. A many-well run assigns work to persistent processes. A backend-selected run can use NumPy/Numba by default and move compatible steps to GPU-oriented backends when parity-tested variants exist. The user-facing workflow remains the same while the execution mode changes.

### Each engineering layer removes one recurring lab problem

OpenHCS is a platform because the user-facing workflow depends on several lower-level engineering layers working together. These layers are not exposed to make the system sound complex. They exist because each one removes a recurring problem that labs otherwise solve manually.

| Engineering layer | Lab problem it removes | User-visible consequence |
|---|---|---|
| ObjectState | Parameters scattered across widgets, configs, defaults, and runtime overrides | Inherited values, local overrides, provenance, undo/redo, GUI/code consistency |
| FunctionStep contracts | Functions with implicit dimensional and memory assumptions | Compile-time validation before long runs |
| ArrayBridge | Backend-specific array conversion code | NumPy, CuPy, JAX, PyTorch, TensorFlow, and pyclesperanto functions can share one pipeline model |
| PolyStore | Separate code paths for disk, memory, Zarr, napari, Fiji, and runtime artifacts | Outputs and viewers become selectable backends rather than custom side effects |
| ZMQRuntime | Worker startup, progress reporting, viewer isolation, and remote execution handled ad hoc | Persistent workers, process-isolated viewers, OMERO-side execution paths, progress streaming |
| pycodify | GUI workflows trapped in opaque serialized state | Executable Python becomes a durable, editable workflow representation |
| python-introspect | Handwritten parameter adapters for functions and dataclasses | Custom functions receive UI forms and contract analysis from signatures |
| metaclass-registry | Manual plugin registries that drift or miss subclasses | Backends, module adapters, and handlers register through class definition |

A custom function can be discovered through the registry, inspected through its signature, configured through typed state, validated by the compiler, converted to the right memory type by ArrayBridge, executed by a worker, and streamed to a viewer through PolyStore. The function author writes the analysis function rather than the surrounding infrastructure.

### CellProfiler pipelines compile into normal OpenHCS pipelines

CellProfiler import is implemented as a compiler path rather than a separate runner. The `.cppipe` file is parsed into modules and settings. Infrastructure modules such as image loading and metadata are mapped into OpenHCS source schema and source binding. Processing modules resolve to absorbed CellProfiler-compatible Python functions. Runtime artifacts represent images, object labels, measurements, relationships, grids, and materialized outputs.

Each imported module becomes an OpenHCS FunctionStep with a CellProfiler-compatible runtime adapter. The adapter gives the function access to the image, object, and measurement semantics expected by CellProfiler modules while storing the results as typed OpenHCS runtime values. The runtime boundary stays narrow. CellProfiler semantics are preserved, but the execution still passes through normal OpenHCS function invocation, artifact planning, memory conversion, and multiprocessing machinery.

**Table 2. CellProfiler concepts and their OpenHCS runtime representation.**

| CellProfiler concept | OpenHCS representation | Why this matters |
|---|---|---|
| `.cppipe` file | Compiler dialect input | Legacy workflows enter OpenHCS without manual rewriting |
| Images module / file loading | Source schema and source binding | File discovery becomes part of OpenHCS path planning and validation |
| Metadata / NamesAndTypes | Typed source fields and channel/source mappings | Image identity is preserved before module execution begins |
| Processing module | FunctionStep with CellProfiler-compatible callable | Module behavior runs through normal OpenHCS invocation |
| Image set | Runtime image artifact | Intermediate images can be stored, streamed, compared, or reused |
| Objects / labels | Runtime object artifact | Segmentation outputs remain typed and comparable |
| Measurements | Runtime measurement artifact | Tables are generated from declared measurement semantics |
| Relationships / grids | Runtime relationship or grid artifact | Non-image CellProfiler state is preserved alongside image outputs |
| SaveImages / ExportToSpreadsheet | Materialization plan and format writer | Output files are produced by the OpenHCS artifact system |
| Module settings | Typed configuration state | Settings can be inherited, edited, serialized, and audited |

The compiler path avoids a common failure mode in interoperability software: building a second runtime beside the main one. A separate CellProfiler runner would not inherit the OpenHCS platform. Imported workflows can be inspected, modified, extended, benchmarked, and accelerated using the same mechanisms as native OpenHCS workflows.

Compatibility is therefore not defined as "the file imports." Compatibility means that the imported workflow has the same scientific meaning under the supported module and setting subset. A threshold module must produce the same threshold behavior. An object-identification module must preserve label, object, and measurement semantics. A measurement module must preserve table meaning, not only column names. Save and export modules must materialize outputs from the same runtime artifacts that produced the measurements. This is why parity is attached to the compiler path rather than treated as a separate after-the-fact smoke test.

The importer also deliberately keeps CellProfiler-specific behavior at the edge. CellProfiler module settings, naming conventions, and measurement expectations are translated into OpenHCS objects, but the rest of the system should not become CellProfiler-shaped. Once compiled, the workflow should look like any other OpenHCS workflow: source schema, FunctionSteps, artifacts, typed state, backend contracts, and materialization plans. That boundary is what lets OpenHCS preserve CellProfiler while still supporting Fiji, napari, OMERO, custom functions, and future GPU variants.

### OpenHCS reproduces CellProfiler outputs across a broad `.cppipe` benchmark set

The CellProfiler benchmark set is designed to test generality rather than a single curated demonstration. The target set contains `TODO: 33` `.cppipe` workflows: `TODO: 18` official benchmark or example pipelines, `TODO: 7` official tutorial pipelines, and `TODO: 8` public workflows found outside the official example set. Official examples test known CellProfiler semantics, tutorials test user-facing workflows, and public third-party pipelines test whether the importer generalizes beyond examples used during development.

The corpus covers common HCS analysis patterns rather than one repeated assay. It includes workflows centered on object identification, object filtering, size and shape measurement, texture measurement, colocalization, image math, illumination correction, grid-based object assignment, object tracking, object-to-image conversion, object overlays, image export, table export, and specialized morphology such as worm untangling. The categories exercise different parts of CellProfiler semantics: image arithmetic, label geometry, object relationships, measurement tables, thresholding, display/materialization, and non-image runtime state.

The corpus also separates source types. Official examples provide stable reference workflows maintained near CellProfiler itself. Tutorials represent the workflows users encounter when learning the tool. Public third-party pipelines test the importer against workflows not selected only because they were convenient for OpenHCS development. The benchmark covers workflow style, module mix, and dataset organization.

Benchmark grouping is declared in the dataset manifest rather than inferred from filenames. Each workflow receives three independent tags: source category, assay family, and dominant semantic pressure. Source category records where the `.cppipe` came from. Assay family records the biological or imaging context at a coarse level. Semantic pressure records the part of the runtime most likely to expose incompatibility.

| Grouping axis | Example values | Purpose in the benchmark |
|---|---|---|
| Source category | official example, official tutorial, public third-party workflow | Separates maintained CellProfiler examples from user-facing tutorials and external pipelines |
| Assay family | cell/nuclei morphology, intensity screening, colocalization, tracking, worm morphology, illumination correction, histology/large image, neuron morphology | Keeps the performance plots interpretable for wet-lab readers without making each assay label too specific |
| Semantic pressure | thresholding, object labeling, object relationships, measurement tables, image math, image export, grid geometry, temporal linkage, specialized morphology | Identifies which runtime semantics are being tested when a pipeline passes parity |
| Output pressure | image artifacts, object labels, measurement CSVs, relationships, overlays, saved images, mixed artifacts | Separates pure image-processing speed from table/output-heavy workloads |

For each pipeline, native CellProfiler is run to produce reference outputs. OpenHCS then imports the same `.cppipe`, runs the corresponding OpenHCS pipeline, and compares output measurements, labels, image-derived values, and materialized artifacts according to declared equivalence tolerances. A pipeline is counted as passing only when the semantic comparison reports no unresolved differences. Numeric tolerances are reported explicitly and are kept separate from exact table or categorical matches.

The current benchmark target is full parity across all `TODO: 33` pipelines. The first fully plotted benchmark set contains `TODO: 18` official pipelines. The expanded set includes the additional official tutorials and public online pipelines. The benchmark table reports the exact pass table, the module coverage table, and any explicitly excluded modules or disabled CellProfiler features.

### OpenHCS is faster under deliberately constrained CPU-only conditions

The primary performance benchmark uses the least favorable setting for OpenHCS: one sample, one CPU core, no GPU acceleration, no multiprocessing advantage, and no batching. The condition separates execution speed from extra cores, GPU kernels, larger batch size, and amortized throughput over many wells.

The benchmark reports execution-only timing separately from total wall time. Execution-only timing measures the part of the run most directly comparable to CellProfiler module execution. Total timing includes startup, import, compile, preparation, and execution. Execution-only timing asks how fast the workflow runs once invoked. Total timing asks what a user experiences for a cold run.

The figures separate cold-run timing from execution timing. A cold OpenHCS run includes Python import time, optional backend imports, pipeline compilation, Numba compilation, OpenCV initialization, worker setup, and output-store preparation. Those costs are real user costs for a one-off run. They differ from per-sample analysis cost. In HCS, the same worker normally processes many wells or sites. Fixed costs are paid once and then divided over the number of samples assigned to that worker.

The throughput model can be stated plainly:

`effective_seconds_per_sample = (worker_startup + compile + warmup) / samples_per_worker + execution_seconds_per_sample + output_seconds_per_sample`

As `samples_per_worker` increases, fixed costs become less important and the measured speedup per core approaches the execution-only speedup. One-sample results are conservative for OpenHCS because they include the least amortization and the least opportunity to benefit from persistent workers. The platform model is most natural for many-well screens.

Across the final benchmark target, OpenHCS reports the minimum, median, mean, and maximum speedup for single-core execution. The current target is at least `TODO: 4x` execution speedup on every tested pipeline, with stronger average and best-case speedups reported from the final benchmark CSVs. Any pipeline below the target remains visible in the development reports until optimized or excluded with a source-level reason.

The CPU-only result is the floor under deliberately constrained conditions. Multiprocessing, persistent workers, and GPU backend support are additional capabilities layered on top of the matched single-core result.

### HCS throughput benefits from persistent workers and well-level parallelism

High-content screening usually contains many wells. OpenHCS is also evaluated in the way it is meant to be used: many wells, persistent workers, and enough RAM to keep workers alive across repeated work. In that setting, import cost, compile cost, and backend warmup are amortized across many samples.

OpenHCS supports process-level parallelism rather than relying on Python threads for compute scaling. Worker processes can execute wells independently, and GPU scheduling can assign devices to workers when GPU backends are used. A persistent ZMQ worker model further separates one-time worker setup from steady-state execution. Workers can load libraries, receive compiled work, warm runtime hooks, and then process many submitted wells without paying the full startup cost each time.

Scaling is reported per resource, not only per wall-clock speedup. Each additional worker adds CPU capacity and also adds memory pressure, because the worker may hold imported libraries, compiled kernels, open stores, cached arrays, and intermediate outputs. The benchmark reports throughput alongside peak RAM and an approximate RAM-per-worker slope. The capacity curve identifies how many samples per hour a machine can process for a pipeline class under a given worker and RAM budget.

The sample-per-core axis determines whether the throughput test represents HCS. A single sample on two workers is usually not meaningful because one worker has little or no work to do. Two workers on two samples is a minimal concurrency test, not an HCS throughput test. Better conditions are `1 worker x N samples`, `2 workers x N samples`, and `3 workers x N samples`, where `N` is large enough that each worker receives repeated work. A 16-sample or 24-sample condition is more interpretable than a 2-sample condition because startup and warmup no longer dominate the denominator.

The scaling model has three regimes. If per-sample execution dominates, wall time improves close to the number of workers until I/O, memory bandwidth, RAM pressure, or load imbalance becomes limiting. If startup dominates, speedup is poor for small sample counts and improves as more samples are assigned per worker. If RAM is insufficient, paging and cache pressure can flatten or reverse the benefit of parallelism. Joint reporting of speed, samples per worker, and RAM identifies the active regime.

Throughput benchmarks are reported as a second layer, separate from the single-core result. The single-core benchmark demonstrates execution under matched constraints. The throughput benchmark shows OpenHCS operating as an HCS platform rather than as a one-sample runner.

### GPU backend support is an architectural extension, not a requirement for the CPU result

OpenHCS already supports multiple memory backends through ArrayBridge and memory decorators. Native OpenHCS functions can declare NumPy, CuPy, pyclesperanto, PyTorch, TensorFlow, or JAX memory semantics. The current CellProfiler-compatible implementation is mostly NumPy and Numba because CellProfiler itself and the absorbed module implementations are NumPy-oriented. Backend selection is a compiler concern rather than a CellProfiler-specific rewrite.

The clean path to GPU-backed CellProfiler modules is compiler-selected callable variants. A CellProfiler module resolves to a backend-compatible function at compile time: NumPy/Numba for parity defaults, CuPy or CuCIM for NumPy-like GPU image operations, pyclesperanto for morphology and OpenCL image-processing primitives, and JAX for pure dense kernels where JIT compilation can be prepared before timed execution. The runtime still sees a normal OpenHCS callable with declared memory types.

The CPU benchmark stands independently of future GPU support. GPU support increases the ceiling and broadens hardware use, while the CPU result shows that CellProfiler-compatible workflows can be preserved and accelerated without using a GPU. GPU acceleration extends the platform rather than explaining the benchmark.

## Discussion

OpenHCS addresses a gap that has become visible as high-content screening has matured. The field has trusted image-analysis tools, powerful viewers, GPU libraries, storage systems, and workflow engines, but the integration burden remains high. Users often move between GUI pipelines, scripts, viewers, file exports, custom functions, and batch execution by hand. Each boundary introduces a chance to lose metadata, duplicate parameters, break reproducibility, or silently change semantics.

OpenHCS makes those boundaries explicit and composable. A pipeline is compiled. A function declares a memory contract. A storage destination is a backend. A viewer is a streaming backend. A measurement table is a runtime artifact. A parameter can be inherited, overridden, inspected, serialized, and regenerated. A custom function can enter the same registry as built-in functions. A legacy CellProfiler workflow becomes an imported dialect rather than a foreign file that must be manually rewritten.

The CellProfiler benchmark validates the architecture against a mature baseline whose module semantics encode years of biological image-analysis practice. Its public workflows cover many common HCS patterns. OpenHCS reproduces those outputs and runs them faster under one-core CPU-only constraints across official examples, tutorials, and public third-party workflows.

The benchmark differs from a conventional speed test in its validation structure. Output parity is a primary requirement. The constrained CPU benchmark separates execution speed from hardware and batching. Throughput scaling is reported separately. GPU backend support is reported only where a specific variant has parity evidence.

The platform also changes how HCS workflows can be shared. A CellProfiler pipeline can be imported and preserved. An OpenHCS pipeline can be edited visually or as Python source. Custom functions can be registered directly by a lab, with typed configuration forms generated from signatures. LLM support can operate over the actual function registry and type information rather than producing detached scripts. Users get several entry points without fragmenting the underlying workflow.

The reusable-library structure is also central. ObjectState, ArrayBridge, PolyStore, ZMQRuntime, pyqt-reactive, pycodify, python-introspect, and metaclass-registry each solve a general scientific Python infrastructure problem [ObjectState; ArrayBridge; PolyStore; ZMQRuntime; pyqtReactive; pycodify; pythonIntrospect; metaclassRegistry]. OpenHCS uses them together for HCS, but the abstractions are not limited to microscopy. Scientific software repeatedly needs typed state, backend conversion, storage abstraction, code serialization, process-isolated execution, and reactive parameter editing. OpenHCS demonstrates these pieces in a demanding real application.

There are limitations. The CellProfiler importer covers the module and setting subset represented in the parity-tested corpus. Some modules have semantics that are tightly coupled to CellProfiler internals and require explicit compatibility work. Measurement-heavy modules can remain CPU-bound even when image kernels move to GPU backends. GPU acceleration requires careful parity testing because library defaults differ in boundary handling, dtype behavior, label semantics, and reductions. Persistent workers improve throughput while introducing process-lifecycle considerations. The engineering limits remain visible through typed contracts and benchmark reports.

### What changes when integration becomes semantic

The central change is not that OpenHCS can call several tools from one process. Many systems can call external tools. The change is that OpenHCS gives the workflow a shared semantic object model before the tool boundary is crossed. A viewer receives a runtime artifact. A file writer materializes a runtime artifact. A benchmark compares a runtime artifact. A worker executes a compiled plan over runtime artifacts. A GUI edit changes typed state that will be recompiled into runtime artifacts. The same nouns remain present across the system.

That structure removes several recurring failure modes in HCS work.

| Failure mode | Ordinary workflow pattern | OpenHCS pattern |
|---|---|---|
| Stale intermediate files | A user reruns one step but inspects an older saved image | Intermediates are runtime artifacts tied to the current compiled run |
| Duplicated parameters | Thresholds or channel names are copied into GUI settings, scripts, and notebooks | Parameters live in typed state with inherited/local provenance |
| Hidden viewer edits | A mask is corrected or inspected in a viewer but the workflow record does not know why | Viewer-facing artifacts remain connected to the pipeline object |
| Lost metadata | Files move between tools without well/site/channel identity | Source schema and binding carry identity before processing begins |
| Unclear object scope | Measurements, labels, parents, and children are matched by naming convention | Object, relationship, and measurement artifacts keep explicit roles |
| Backend drift | CPU and GPU paths silently differ in dtype, boundaries, or reductions | Backend variants remain separate until contract-compatible and parity-tested |
| Unreviewable automation | Generated scripts or assistant edits become detached from the GUI workflow | Code generation and LLM edits target the same typed objects and callable registry |

These are qualitative advantages, but they are not cosmetic. They determine whether a fast workflow remains scientifically inspectable after a lab adapts it, scales it, or combines it with another tool.

### OpenHCS is not only a faster CellProfiler runner

The CellProfiler benchmark is a deliberately strict entry point into the platform. It asks whether a mature legacy workflow format can be preserved, compiled, checked, and accelerated without changing scientific outputs. Passing that test is useful by itself, but it is not the full scope of OpenHCS.

A faster runner would keep the input format and optimize execution. OpenHCS keeps the input format and changes the substrate around it. The imported workflow gains typed state, Python serialization, function registration, runtime artifacts, storage and viewer backends, persistent workers, backend memory contracts, and manifest-driven benchmarking. The same mechanism that runs a `.cppipe` file can run custom Python functions, stream artifacts to viewers, or later select GPU-compatible variants.

This matters for adoption. Existing CellProfiler users do not have to abandon their pipelines to enter the OpenHCS model. Existing Python users do not have to stay inside CellProfiler semantics once they enter it. Methods developers can add new functions without writing a new GUI, serializer, worker protocol, viewer pathway, and benchmark adapter each time. The platform gives each user group a natural entry point while keeping the workflow unified underneath.

### Compatibility boundaries handled by the benchmark

OpenHCS treats compatibility as a measured property of a workflow and module set. A CellProfiler module is supported for a benchmark class when its outputs match native CellProfiler under the declared comparison policy. Successful import alone is insufficient. Unsupported settings, partially covered modules, and backend-specific differences remain visible in the corpus and module tables.

| Boundary | How it appears | How OpenHCS handles it |
|---|---|---|
| Unimplemented CellProfiler module | `.cppipe` import cannot resolve a module class | Module remains listed as unsupported until implemented and parity-tested |
| Partially covered module setting | Module imports but a setting path is not represented | Setting is listed in module coverage notes and excluded from broad support wording |
| Numeric tolerance | Floating-point outputs differ at small scale | Absolute and relative tolerances are declared with the comparison report |
| Label/object semantics | Object identifiers, outlines, declumping, or parent-child relationships differ | Object and relationship equivalence checks run separately from raw image comparisons |
| Measurement table semantics | Column order, object scope, image scope, or row identity differs | Measurement facts are compared semantically rather than only by file bytes |
| Materialized outputs | Saved images, ROI files, CSV/JSON outputs, or display artifacts differ | Format writers and artifact comparisons are tested explicitly |
| GPU backend variant | GPU library changes boundary, dtype, reduction, or label behavior | Backend variant remains separate until it passes the same parity tests |
| Worker lifecycle | Persistent workers change startup and memory behavior | Throughput benchmarks report worker count, samples per worker, and RAM |

Legacy workflow formats are preserved as dialects inside modern semantic platforms rather than discarded because their original execution engines were built for an earlier era. OpenHCS applies this pattern to CellProfiler workflows: preserve the scientific output, accelerate the execution, and make the workflow available to a richer ecosystem of state management, viewers, storage, custom functions, multiprocessing, and GPU backends.

### Interpretation for the field

CellProfiler remains central because the `.cppipe` ecosystem encodes trusted biological analysis practice. Fiji and ImageJ remain central because their plugin ecosystem and ROI conventions are part of daily microscopy work. napari remains central because it gives Python users a powerful viewer. OMERO and OME-oriented storage remain central because microscopy data management is a field-level problem. OpenHCS lets these pieces participate in one workflow without reducing them to manual file handoffs.

Acceleration has greater scientific value when it preserves the whole workflow. A faster imported CellProfiler workflow carries more context than a faster isolated module. A faster imported workflow that can be inspected in viewers, modified with custom functions, represented as code, run across workers, and eventually assigned to CPU or GPU backends changes the workflow substrate. The speedup is the first visible consequence of moving legacy workflow semantics into a compiled platform.

The benchmark design keeps the algorithm, hardware, output semantics, and workload separate. Native CellProfiler defines the reference output. OpenHCS matches it. The primary speed result uses one core and one sample. Throughput scaling is measured separately. Startup and compilation are reported separately from execution. RAM is reported with worker scaling. The resulting tables give one-off users, benchmark readers, and HCS users the timing number that matches their use case.

## Methods

### OpenHCS pipeline compilation

OpenHCS pipelines are built from FunctionSteps and configuration objects. Before execution, the pipeline is compiled into immutable runtime contexts. Compilation resolves input source patterns, validates function memory contracts, resolves processing contracts, constructs artifact input and output plans, assigns runtime paths, and prepares callable execution hooks. The compiled form is the unit submitted to workers.

### Function memory contracts

Functions declare their memory interface using decorators such as `@numpy`, `@cupy`, `@pyclesperanto`, `@torch`, `@tensorflow`, or `@jax`. The decorators attach input and output memory metadata and preserve OpenHCS callable metadata. During execution, OpenHCS converts the main image payload between the current memory type and the next callable's required memory type using ArrayBridge. Runtime artifact plans handle special outputs and can use CPU-native structures when appropriate.

### CellProfiler pipeline import

CellProfiler `.cppipe` files are parsed into module blocks and settings. Infrastructure modules are mapped into OpenHCS source schema and source-binding configuration. Processing modules resolve to absorbed CellProfiler-compatible functions. Each resolved module is wrapped as an OpenHCS FunctionStep with a CellProfiler runtime adapter. The adapter provides module access to images, object labels, measurements, grids, and relationships while storing outputs as OpenHCS runtime values.

### Parity comparison

Native CellProfiler is run for each benchmark pipeline to generate reference outputs. OpenHCS is then run on the imported version of the same pipeline. Outputs are compared through semantic equivalence checks for image payloads, object labels, measurement rows, relationships, scalar values, and materialized files. Numeric comparisons use declared absolute and relative tolerances. Non-numeric identifiers and categorical values are compared exactly unless a specific CellProfiler-compatible normalization is documented.

### Performance benchmarking

The primary benchmark condition uses one sample, one CPU core, no GPU acceleration, and no multiprocessing advantage. Timing is reported as both execution-only time and total wall time. Execution-only time isolates module/runtime execution after startup and compile overhead. Total wall time includes startup, imports, compile, preparation, and execution. Throughput benchmarks are reported separately using multi-well execution and worker-level parallelism.

### Benchmark corpus

The target benchmark corpus contains `TODO: 33` `.cppipe` workflows: `TODO: 18` official benchmark/example pipelines, `TODO: 7` official tutorial pipelines, and `TODO: 8` public third-party workflows. The corpus table lists each workflow, source category, source URL or citation, dataset source, assay family, dominant semantic pressure, output pressure, module coverage, parity status, single-core speedup, and throughput status where available. The categories are explicit manifest fields and are not inferred from pipeline filenames during figure generation.

### Reproducibility package

The benchmark runner emits the OpenHCS commit, CellProfiler version or commit, Python version, operating system, CPU model, GPU model if used, RAM, pipeline manifest, dataset manifest, checksums where available, native CellProfiler timing, OpenHCS timing, parity reports, and raw CSVs used to generate figures. The benchmark figures are regenerated from saved CSV files without rerunning CellProfiler.

## Conclusion

OpenHCS provides a semantic execution platform for high-content screening workflows. Trusted CellProfiler `.cppipe` pipelines can be imported as normal OpenHCS workflows, executed with native-output parity, inspected through modern viewer and artifact systems, modified with custom Python functions, serialized as editable code, and scaled across persistent worker processes. Under one-sample, one-core, CPU-only conditions, the imported workflows run faster than native CellProfiler execution across the tested corpus. In many-well settings, persistent workers amortize startup and compilation costs while exposing throughput and RAM tradeoffs directly.

OpenHCS provides a practical path for carrying legacy HCS analysis forward. Existing CellProfiler workflows remain scientifically meaningful. Existing field tools such as Fiji/ImageJ, napari, OMERO, Zarr, scientific Python, and GPU backends keep their natural roles. OpenHCS supplies the typed state, compiled execution, artifact model, backend conversion, and process isolation that let those roles compose in one workflow.

## Draft Figure Captions

### Figure 1. OpenHCS turns a fragmented HCS tool stack into one semantic workflow

Field tools solve distinct parts of high-content screening analysis. CellProfiler provides trusted modular pipelines, Fiji/ImageJ provides plugins and ROI conventions, napari provides Python-native multidimensional viewing, OMERO and OME-Zarr support data management and storage, scientific Python and GPU libraries provide numerical backends, and workflow systems provide orchestration. OpenHCS replaces copied files, duplicated parameters, hidden viewer state, and backend-specific glue with source schemas, typed state, memory contracts, runtime artifacts, and storage/viewer backends. The result is one executable HCS workflow whose meaning is preserved across editing, inspection, benchmarking, and execution.

### Figure 2. Compile-then-execute architecture

OpenHCS represents HCS workflows as compiled semantic dataflow. Source schema, typed state, the function registry, and the CellProfiler dialect feed a compiler that validates FunctionSteps, dimensional behavior, memory contracts, runtime artifacts, materialization plans, and worker execution contexts. ArrayBridge handles backend conversion. PolyStore handles disk, memory, Zarr, napari, and Fiji destinations. Generated Python reconstructs the same pipeline objects edited in the GUI.

### Figure 3. CellProfiler import path

CellProfiler `.cppipe` workflows compile into normal OpenHCS workflows. Modules and settings are parsed from the `.cppipe` file, infrastructure modules define source schema and binding, processing modules resolve to CellProfiler-compatible FunctionSteps, and images, objects, measurements, relationships, grids, and materialized outputs become runtime artifacts. Native CellProfiler outputs provide the reference for parity comparison.

### Figure 4. Benchmark validation structure

The benchmark manifest feeds native CellProfiler and OpenHCS runs. Native CellProfiler defines reference outputs. OpenHCS imports and runs the same `.cppipe` files. Semantic parity, phase timing, throughput, RAM, and category summaries remain separate report layers, so correctness, execution speed, cold-run overhead, and HCS throughput are not collapsed into one ambiguous timing number.

### Figure 5. Throughput amortization

Persistent worker costs are divided by samples per worker, while execution and output costs remain per-sample. The figure distinguishes one-sample cold timing from many-well HCS throughput and shows why startup, compilation, warmup, execution, output, worker count, and RAM need to be reported together.

### Figure 6. Backend extensibility

OpenHCS separates workflow semantics from backend implementation. Current CellProfiler-compatible execution uses NumPy/Numba paths for parity. Compiler-selected backend variants can route compatible functions to CuPy/CuCIM, pyclesperanto, JAX, PyTorch, TensorFlow, or other memory backends through declared memory contracts and ArrayBridge conversion.

### Figure 7. Typed state and bidirectional editing

Typed state keeps GUI editing, Python code, assistant workflows, and runtime execution synchronized. A parameter can inherit from global, pipeline, or step scope; local edits appear in the GUI, generated Python, and runtime context; clearing a value restores inheritance. The UI becomes a teaching surface: it shows where values came from, what changed, and how visual edits map to executable Python. Generated code remains re-importable because it reconstructs the same typed objects rather than creating a detached script.

### Figure 8. Benchmark categories are declared semantics

Benchmark workflows are grouped by declared manifest fields rather than filename heuristics. Source category, assay family, semantic pressure, and output pressure are explicit properties used to organize parity, speed, RAM, and throughput figures. This separates maintained CellProfiler examples, official tutorials, and public third-party workflows while keeping category-level summaries reproducible.

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
