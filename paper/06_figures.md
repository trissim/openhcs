# Figure Plan

**Target:** 8 main figures for a Nature Methods-style platform manuscript.

Editable drafts are in `paper/figures/diagrams/*.dot`; rendered SVG/PNG drafts are in `paper/figures/rendered/`. Rebuild all diagrams and the contact sheet with `python paper/figures/render_diagrams.py`.

## Figure 1. Fragmented Tool Stack To Semantic Workflow

**Source:** `paper/figures/diagrams/fig01_field_integration_gap.dot`
**Rendered:** `paper/figures/rendered/fig01_field_integration_gap.svg`

Field tools solve separate parts of HCS: CellProfiler, Fiji/ImageJ, napari, OMERO/OME-Zarr, GPU libraries, and workflow systems. OpenHCS replaces manual handoffs with source schemas, typed state, memory contracts, runtime artifacts, and storage/viewer backends.

**Main message:** OpenHCS replaces manual file handoffs with typed HCS contracts.

## Figure 2. Compiler And Runtime Architecture

**Source:** `paper/figures/diagrams/fig02_compiler_runtime_architecture.dot`
**Rendered:** `paper/figures/rendered/fig02_compiler_runtime_architecture.svg`

Source schema, typed state, function registry, and CellProfiler dialect inputs enter the OpenHCS compiler. The compiler produces FunctionSteps, runtime artifacts, storage plans, memory conversion, workers, viewer streams, and benchmark outputs.

**Main message:** OpenHCS is a compile-then-execute semantic runtime, not a script wrapper.

## Figure 3. CellProfiler Import Path

**Source:** `paper/figures/diagrams/fig03_cellprofiler_import_path.dot`
**Rendered:** `paper/figures/rendered/fig03_cellprofiler_import_path.svg`

The `.cppipe` file is parsed into module blocks and settings. Infrastructure modules become source schema and binding; processing modules become FunctionSteps; output modules become materialization plans; runtime artifacts are compared against native CellProfiler outputs.

**Main message:** CellProfiler import is a compiler dialect that produces normal OpenHCS workflows.

## Figure 4. Benchmark Validation Structure

**Source:** `paper/figures/diagrams/fig04_benchmark_validation_structure.dot`
**Rendered:** `paper/figures/rendered/fig04_benchmark_validation_structure.svg`

The benchmark manifest feeds native CellProfiler and OpenHCS runs. Native CellProfiler defines reference outputs. OpenHCS imports and runs the same `.cppipe` files. Semantic parity, phase timing, throughput, RAM, and category summaries remain separate report layers.

**Main message:** The benchmark separates correctness, execution speed, cold-run overhead, and throughput.

## Figure 5. Throughput Amortization

**Source:** `paper/figures/diagrams/fig05_throughput_amortization.dot`
**Rendered:** `paper/figures/rendered/fig05_throughput_amortization.svg`

Fixed worker costs are divided by samples per worker; execution and output costs remain per-sample. RAM determines feasible worker count. The capacity curve reports samples per hour, RAM per worker, and speedup versus native CellProfiler.

**Main message:** One-sample timing is conservative; many-well HCS amortizes fixed costs.

## Figure 6. Backend Extensibility

**Source:** `paper/figures/diagrams/fig06_backend_extensibility.dot`
**Rendered:** `paper/figures/rendered/fig06_backend_extensibility.svg`

Workflow semantics remain stable while function memory contracts and ArrayBridge route compatible functions to NumPy/Numba, CuPy/CuCIM, pyclesperanto, JAX, PyTorch, or TensorFlow variants. Backend variants remain subject to the same parity policy.

**Main message:** GPU acceleration is an architectural extension selected by contracts.

## Figure 7. Typed State And Bidirectional Editing

**Source:** `paper/figures/diagrams/fig07_typed_state_bidirectional_editing.dot`
**Rendered:** `paper/figures/rendered/fig07_typed_state_bidirectional_editing.svg`

GUI editing, generated Python, and LLM-assisted construction all target ObjectState. ObjectState resolves inherited/defaulted/local values into compiled runtime contexts and exposes provenance back to GUI and code.

**Main message:** GUI, code, runtime, and assistant workflows converge on one typed state model.

## Figure 8. Benchmark Corpus Categories

**Source:** `paper/figures/diagrams/fig08_benchmark_corpus_categories.dot`
**Rendered:** `paper/figures/rendered/fig08_benchmark_corpus_categories.svg`

Benchmark workflows are grouped by declared manifest fields: source category, assay family, semantic pressure, and output pressure. Figure grouping comes from manifest semantics rather than filename heuristics.

**Main message:** Category-level claims are backed by declared benchmark metadata.

## Next Iteration

- Replace broad labels with final terminology from the manuscript after the benchmark corpus settles.
- Add figure-panel letters if these diagrams become multi-panel composites.
- Re-render SVGs after any DOT edit.
- Convert final SVGs to publication layout in Illustrator/Inkscape only after the argument structure is stable.
