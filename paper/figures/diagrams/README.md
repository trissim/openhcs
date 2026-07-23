# OpenHCS Nature Methods Figure Drafts

Editable source diagrams live in `paper/figures/diagrams/*.dot`.
Rendered SVG/PNG drafts live in `paper/figures/rendered/`.
`paper/figures/rendered/figure_contact_sheet.png` is the fastest review surface.

These are working scientific diagrams, not final art. The goal is to lock the argument structure before polishing typography and icons.

## Draft Set

1. `fig01_field_integration_gap.dot`  
   Field tools solve separate parts of HCS; OpenHCS is the semantic layer that keeps them composable.

2. `fig02_compiler_runtime_architecture.dot`  
   Source schema, typed state, FunctionSteps, runtime artifacts, memory/storage backends, viewers, and workers form one compiled execution model.

3. `fig03_cellprofiler_import_path.dot`  
   CellProfiler `.cppipe` import becomes a compiler dialect that produces normal OpenHCS runtime artifacts and parity outputs.

4. `fig04_benchmark_validation_structure.dot`  
   Native CellProfiler defines reference outputs; OpenHCS imports the same pipelines; parity and speed are reported separately.

5. `fig05_throughput_amortization.dot`  
   Startup, compile, warmup, execution, output, samples per worker, and RAM explain one-sample and many-well performance.

6. `fig06_backend_extensibility.dot`  
   Function memory contracts and ArrayBridge allow backend-selected variants without changing workflow semantics.

7. `fig07_typed_state_bidirectional_editing.dot`  
   GUI editing, generated Python, live runtime state, and inherited/defaulted values converge on one typed state model.

8. `fig08_benchmark_corpus_categories.dot`  
   Benchmark workflows are grouped by source, assay family, semantic pressure, and output pressure rather than filename heuristics.

## Render Command

```bash
python paper/figures/render_diagrams.py
```

The renderer expects Graphviz `dot`, `rsvg-convert`, and Pillow. SVGs are the editable publication sources; PNGs are white-background previews for slides and contact-sheet review.
