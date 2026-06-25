# OpenHCS Nature Methods Draft Review Revision Plan

Source review: `openhcs_nature_methods_draft_review[99].docx`  
Reviewer: Sumner Magruder  
Comment count extracted from `word/comments.xml`: 30

## Revision Posture

Take the review as a request for a major restructure, not a light line edit. The current draft explains architectural features before it establishes the result those features support. The rewrite should:

- Lead earlier with the strongest claims: preserved CellProfiler compatibility, strict output parity, and at least 4x constrained CPU-only speedup.
- Present OpenHCS as an open-source, backwards-compatible, non-opinionated interoperability layer that preserves existing bioimage tools.
- Replace long comma-chain feature lists with named concepts, definitions, figures, and tables.
- Move implementation-heavy lists into Methods, Supplementary Tables, or documentation citations.
- Add visual summaries where prose currently carries too much load.
- Use language that respects domain experts and existing tool developers.

## Adapted Style Constraints

Use the paper style guide as a clarity and scope guide rather than as a theory-paper template. The OpenHCS draft should remain a biomedical software methods paper: direct, concrete, and accessible to cell biologists and image-analysis users. The revision should still follow these transferable rules:

- Write declarative sentences that state what OpenHCS does, what was measured, and what remains bounded.
- Put the strongest verified result early: CellProfiler parity across 30 workflows and at least 4x execution-only speedup under the declared CPU-only condition.
- Define the main object before naming it. For this draft, the object is a provenance-tracked bioimage workflow that keeps sources, parameters, named outputs, viewers, code, and workers connected.
- Split dense sentences. Do not stack source handling, viewer output, Python extension, parity, and worker execution in one sentence.
- Prefer exact scope over hedging. Say which sources, viewers, modules, versions, benchmark conditions, and hardware are validated.
- Avoid hype and meta-paper wording. Replace “upsell,” “punchline,” “key feature,” “this shows,” “the claim is,” and “this paper” with concrete results or objects.
- Use comparison tables for ecosystem positioning. Tables should state what each tool preserves and how OpenHCS interoperates with it, without implying the existing tool is deficient.
- Resolve pronouns. Repeat nouns such as “workflow,” “source binding,” “parity check,” “viewer output,” and “worker execution” when ambiguity is possible.
- Preserve a warmer methods-paper tone where useful. “We measured,” “we benchmarked,” and “we implemented” are acceptable when they identify the experiment or implementation. Avoid self-referential paper narration such as “this manuscript argues.”

## Proposed Manuscript Restructure

### 1. Abstract

Rewrite the abstract with a compact three-paragraph structure.

Paragraph 1: Problem and positioning.

- Microscopy acquisition has scaled faster than analysis, review, and provenance.
- The core problem is not lack of tools; it is the boundary between trusted tools, data formats, viewers, custom code, and execution systems.
- Avoid “analysis should scale” as a normative claim. Use a negative/friction framing: as image acquisition scales, analysis and review often become the bottleneck.

Paragraph 2: OpenHCS object and scope.

- State OpenHCS as an open-source, backwards-compatible, bring-your-own-data and bring-your-own-method workflow platform.
- Suggested core phrase to test: “OpenHCS lets existing bioimage tools, microscope folders, Bio-Formats-readable data, viewers, Python functions, and worker execution remain part of one provenance-tracked analysis.”
- Include “Any microscope / any format / any method” only if the verified scope supports it. Otherwise use the safer “microscope folders, Bio-Formats-readable data, CellProfiler pipelines, Python functions, napari/Fiji, and worker execution.”
- Explicitly mention open source in the abstract.

Paragraph 3: Validation and quantitative result.

- Move CellProfiler parity and 4x speedup into the abstract’s main result sentence.
- Keep correctness before speed: “Across 30 CellProfiler workflows, OpenHCS reproduced native outputs under declared parity checks and achieved at least 4x execution-only speedup under one-sample, one-thread/core, CPU-only conditions.”
- Add one phrase about future compatibility/maintenance: OpenHCS augments existing workflows rather than requiring labs to abandon them.

### 2. Introduction

Introduce the “boundary problem” explicitly before listing tools.

- Define “boundary problem” early: parameters, intermediate masks, source identity, viewer state, custom code, and execution metadata become disconnected as analyses cross tools.
- Reduce the number of long lists. Where lists remain necessary, move them after a named concept rather than opening paragraphs with them.
- Reframe existing tools positively. Fiji, CellProfiler, napari, OMERO, Bio-Formats, and workflow managers each solve valuable problems. OpenHCS preserves their strengths in one runnable analysis.
- Add “provenance” as a primary term alongside reproducibility.
- Avoid “14 standards and now there is a 15th” optics by emphasizing pluralism/interoperability and non-opinionated integration.

### 3. Results Order

Use this order so the reader sees evidence earlier:

1. **OpenHCS organizes the boundary problem into one runnable analysis**
   - Use a figure; prose should introduce the figure rather than carry the full architecture.
   - Define inputs, workflow graph, named outputs, viewer destinations, code representation, and workers.

2. **CellProfiler workflows are preserved under parity checks**
   - Move compatibility/parity up before custom Python and backend detail.
   - Include the compatibility mapping table here or as Supplementary Table 1 if too large.

3. **OpenHCS is at least 4x faster under constrained CPU-only conditions**
   - Move the 4x result much earlier than the current draft.
   - Keep parity-before-speed logic explicit.
   - Add a benchmark figure with parity, speedup, cold-run overhead, and throughput summaries.

4. **Existing data sources, viewers, and Python methods remain attached to the run**
   - Combine microscope handlers, Bio-Formats, OMERO, napari/Fiji, generated Python, and custom functions under interoperability/provenance.
   - Keep heavy backend/decorator details out of the main prose.

5. **Persistent workers scale many-well throughput**
   - Keep after the constrained speed result.
   - State hardware, CPU/GPU status, memory tradeoffs, and queue-depth behavior clearly.

### 4. Figures and Tables

#### New or revised Figure 1: Boundary-to-workflow diagram

Required front-end figure.

Panel structure:

- Left: fragmented analysis world: microscope folders, Bio-Formats/OMERO, CellProfiler `.cppipe`, Python functions, napari/Fiji, exported files, workers.
- Center: OpenHCS workflow graph/DAG with source identity, parameter state, named outputs, and provenance.
- Right: inspected masks, tables/files, generated Python, CellProfiler parity, and scaled worker execution.

Caption should state the scope directly: “OpenHCS keeps sources, parameters, outputs, viewers, code, and worker execution connected as one runnable analysis.”

#### Revised Figure 2: Compatibility and performance evidence

Combine:

- `.cppipe` import path.
- Native CellProfiler reference run.
- OpenHCS run.
- Output parity checks.
- 30-workflow pass summary.
- Execution-only speedup distribution.
- Total/cold-run timing.
- Persistent-worker throughput.

The 4x result should be visible before backend and extension details.

#### Revised Figure 3: User-facing workflow/editability

Show:

- GUI parameter inheritance/override.
- Generated Python representation.
- Step-level viewer output to napari/Fiji.
- Custom Python function as a workflow step.
- Source binding/microscope handler.
- Worker execution.

Use “domain expert” and “computational collaborator,” not “wet-lab user.”

#### Main or supplementary compatibility/interoperability table

Add a table for ecosystem positioning and compatibility scope.

Columns:

- Existing tool/platform.
- Primary strength.
- What OpenHCS preserves.
- What OpenHCS adds at the boundary.
- Validation status in the draft.

Rows:

- CellProfiler.
- Fiji/ImageJ.
- napari.
- OMERO/BIOMERO-style systems.
- Bio-Formats.
- Microscope vendor folders.
- Python/GPU/deep-learning libraries.
- Generic workflow managers.

Tone requirement: phrase the “boundary” column carefully so it cannot be read as putting down existing tools. Prefer “OpenHCS relationship” over “limitation.”

## Section-Level Edit Plan

### Abstract

- Add “open-source.”
- Add compatibility/backwards-compatible framing.
- Add parity and 4x speedup earlier.
- Replace feature inventory with one strong interoperability sentence.
- Avoid “should scale” phrasing.

### Intro

- Insert a short “Boundary Problem” definition.
- Move detailed feature lists later.
- Recast the tool ecosystem paragraph around pluralism: existing tools each solve valuable problems; OpenHCS connects their boundaries.
- Use “provenance” repeatedly and consistently.
- Add a forward reference to Figure 1.

### First Results Subsection

Rename to a clearer, result-oriented heading. Candidate headings:

- “OpenHCS turns fragmented bioimage analysis into one runnable workflow”
- “OpenHCS organizes the boundary between data, tools, code, viewers, and workers”
- “OpenHCS keeps trusted tools connected in one analysis”

Replace the current long representative-run sentence with Figure 1 plus short prose.

### Setup Validation / Parameter State

- Change inheritance phrasing to “enter a value to override; clear a value to inherit.”
- Say inherit from what: preset, parent config, imported setting, or global/default state.
- Replace “wet-lab user” with “assay expert,” “domain expert,” or “biologist.”
- Emphasize that domain experts can stay in the GUI while code-savvy collaborators review generated Python.

### Custom Python / GPU / Deep Learning

- Move the long library list to Methods or a supplementary table.
- In Results, describe this as graph/DAG extensibility: an existing callable can become a workflow node with parameters, memory expectations, named outputs, and viewer/worker integration.
- Treat backend-specific functions as intentional alternative methods, not CellProfiler-preserving claims.

### Sources, Viewers, and Compatibility

- Add an interoperability table.
- Make “forward-thinking, backwards-compatible” the theme: OpenHCS keeps trusted formats/tools usable while allowing extension.
- For microscope handlers, separate verified claims from aspirational claims.
- Add a note or Methods paragraph about path robustness and deployment conditions: symlinks, aliases, network filesystems, SMB/NFS latency, and cloud/HPC execution. Verify what is true before claiming cloud bursting.

### CellProfiler Compatibility

- Replace “mature workflow format” if it reads like a reason to stay in CellProfiler. Possible replacement: “widely used, trusted workflow format.”
- Add reassurance about versions: declare the CellProfiler version/commit used for parity and state how future CellProfiler versions are treated.
- State the operational result: users do not have to rewrite trusted `.cppipe` workflows to inspect, extend, or accelerate them in OpenHCS.

### Benchmark Results

- Move the “at least 4x faster” heading and data earlier.
- Add the CPU-only condition and hardware details near the figure, not buried in Methods.
- If any GPU benchmark is available or feasible, add a secondary GPU/throughput result. If not, explicitly state that the primary benchmark is CPU-only and that GPU-backed functions are supported as methods but not part of the parity-preserving CellProfiler speed claim.
- Include GPU model in Methods/reproducibility package whenever GPU is referenced.

### Discussion

- Lead with provenance/compatibility rather than another feature list.
- Re-emphasize maintenance and non-replacement: OpenHCS augments existing tools and preserves lab investment in them.
- Tone-polish the comparison table and discussion to be respectful to existing tool developers.
- Consider contacting or informally sanity-checking wording with developers of major tools if making direct comparative claims.

### Online Methods

- Move decorator/backend examples here or to supplement.
- Add exact versioning policy for CellProfiler, OpenHCS, Python, OS, CPU, and GPU.
- Add source/path handling details relevant to network/cloud environments only after verification.
- Cite OpenHCS documentation where appropriate.

## Comment-by-Comment Action Map

| Comment ID | Anchor/topic | Required action |
|---|---|---|
| 2 | Abstract feature/value sentence | Strengthen the abstract with verified scope: open-source status, BYO data/methods, Bio-Formats/microscope folders, viewers, Python/GPU/deep learning, CellProfiler parity, and 4x speedup. Avoid burying the main value in a flat list. |
| 3 | “analysis should scale” | Replace normative phrasing with bottleneck/friction framing: acquisition has scaled, but analysis/review/provenance often become the limiting step. |
| 4 | “bottleneck includes runtime…” | Add a figure reference or move this into the Figure 1 boundary-problem schematic. |
| 5 | Long citation/tool list | Avoid “new standard” optics. Reframe as OpenHCS connecting existing tools rather than replacing them. |
| 6 | “boundaries between” | Emphasize non-opinionated pluralism and interoperability. |
| 7 | Long list of connected artifacts | Define “boundary problem” and use named groups instead of serial lists, especially early in the paper. |
| 8 | Abstract `.cppipe` paragraph too dense | Condense abstract heavily; move detailed CellProfiler explanation into Results. |
| 11 | Heading/title | Rename first Results section to a concrete value-focused heading, such as “OpenHCS organizes the boundary problem.” |
| 12 | Section headings | Make skimmable headings more concrete and result-oriented. |
| 13 | “practical observation” / “should scale” | Remove or rephrase “should scale” again; use observed bottleneck. |
| 14 | Representative run sentence | Convert this sentence into Figure 1. Do not make prose carry the whole architecture. |
| 15 | CellProfiler / versioning | Add versioning and environment policy: same data under declared CellProfiler/OpenHCS versions; future CellProfiler versions require declared compatibility checks. |
| 17 | Image inputs / cloud bursting | Verify and describe behavior for symlinks, aliases, NFS/SMB latency, and cloud/HPC storage. Avoid unsupported cloud-bursting claims. |
| 18 | Inherit/override phrasing | Reorder and define: entering value overrides; clearing field inherits from a specific parent/default/imported state. |
| 19 | “wet-lab user” | Replace with “domain expert,” “assay expert,” “biologist,” or similar non-diminutive phrasing. |
| 21 | Custom function/backend list in Results | Move technical list out of early Results; lead Results with 4x performance/parity and put backend mechanics in Methods/supplement. |
| 22 | DAG/workflow builder framing | Consider explicitly describing OpenHCS as a workflow graph/DAG where functions are nodes with declared inputs, outputs, parameters, and memory requirements. |
| 24 | “Image folders, OMERO…” heading | Reframe around backwards-compatible extension: “Forward-compatible analysis without abandoning trusted tools” or similar. |
| 25 | Microscope handler paragraph | Add compatibility/interoperability table. Separate source types, quirks handled, and validation status. |
| 27 | “mature workflow format” | Replace with less off-putting phrasing: “widely used, trusted CellProfiler workflow format.” |
| 28 | Output parity policy | Use this to reassure readers about future CellProfiler versions and declared comparison policy. |
| 30 | Benchmark corpus categories | Cite/link OpenHCS docs or benchmark documentation where helpful. |
| 32 | 4x speed result | Move much earlier and make it a front-line result, not a late subsection. |
| 33 | GPU/batch sentence | Add a figure and specify hardware. If discussing GPU, note GPU model or state GPU was excluded from the parity-preserving benchmark. |
| 36 | Inspect/edit/rerun | Add “provenance” as a central term. |
| 37 | “augments” / maintenance | Promote maintenance/backwards compatibility. OpenHCS builds on existing tools in a fast-changing AI/software environment. |
| 38 | Open-source absent from abstract | Add open-source status to abstract. |
| 39 | Comparison table tone | Audit table wording carefully so it cannot be read as disparaging existing tools; consider external sanity check for comparative claims. |
| 45 | Decorators | Move decorator examples to Methods/supplement or make them less prominent in main text. |
| 52 | CPU benchmark | Anticipate GPU questions: either add a GPU result or clearly explain why the main benchmark is CPU-only and where GPU support fits. |

## Verification Tasks Before Editing the Manuscript

- Confirm the exact OpenHCS source URL and whether “open-source” can be stated without qualification.
- Confirm CellProfiler version/commit used in all parity runs.
- Confirm OpenHCS version/commit used in all benchmark runs.
- Confirm CPU model, RAM, OS, Python version, and GPU model if any GPU result or GPU support claim appears in Results.
- Verify source handling claims for symlinks, network storage, SMB/NFS latency, aliases, and cloud/HPC launch environments before mentioning them.
- Confirm the benchmark corpus counts: 30 workflows, 471 CellProfiler module instances, 58 unique module names, 7,158 setting rows, 53 explicitly covered modules, 28 semantically covered modules, and 8 out of current catalogue scope.
- Confirm the exact meaning of “official30 coverage analysis” and rename if needed for reader clarity.

## Suggested Edit Sequence

1. Rewrite abstract and first two intro pages around boundary problem, provenance, open source, backwards compatibility, and parity-plus-speed.
2. Draft Figure 1 schematic and use it to delete the long representative-run prose.
3. Reorder Results so CellProfiler parity and 4x CPU-only speed appear before custom Python/backend internals.
4. Build the compatibility/interoperability table and revise all comparison wording for respectfulness.
5. Move backend/decorator/library details into Online Methods or supplement.
6. Add versioning, deployment, and reproducibility details after verifying current implementation and benchmark artifacts.
7. Polish headings so skimmers see the result and scope without reading every paragraph.
