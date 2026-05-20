# Nature Methods Comparator Action Plan

## Goal

Turn the comparator scan into concrete edits for the OpenHCS Nature Methods draft. The target is a manuscript that reads like a strong platform paper for biologists: practical bottleneck first, one coherent workflow second, validation evidence third, implementation details only after the value is clear.

Source note: `paper/nature_methods_comparable_papers.md`.

## Implementation Status

Applied to `paper/openhcs_nature_methods_draft.md` and `paper/06_figures.md` on 2026-05-18.

Completed in this pass:

- Abstract simplified to the bottleneck/intervention/evidence spine.
- Figure 1 and Figure 2 reframed around analysis bottleneck and one workflow record.
- Results openings tightened into a proof chain.
- CellProfiler positioned as the strict preservation/performance proof, not the product identity.
- Accessibility examples sharpened around GUI, generated Python, step-level viewer output, and Python QC extension.
- Bio-Formats language remains explicitly gated by TODOs until implementation evidence exists.
- Discussion now includes a direct positioning table.
- Discussion limitations now cover parity scope, backend-specific methods, Bio-Formats semantics, persistent workers, viewer environments, and benchmark-condition scope.
- Supplementary table captions now separate parity from speed and explicitly distinguish tested, theoretically covered, and not-covered modules.
- Main-text architecture jargon reduced where it appeared before Results/Methods detail.

## Action 1: Make The Abstract Less Crowded

Priority: high.

Problem:

- The current abstract has the right spine, but it carries too many examples in one paragraph.
- Comparable papers usually make one bottleneck and one intervention unmistakable before listing capabilities.

Edit:

- Sentence 1: acquisition scales faster than analysis/review.
- Sentence 2: existing tools are valuable, but boundaries fragment analysis.
- Sentence 3: OpenHCS keeps sources, parameters, functions, intermediates, viewers, and workers as one workflow record.
- Sentence 4: name only the strongest capability cluster: CellProfiler import, Python functions, napari/Fiji step output, managed/microscope sources, worker execution.
- Sentence 5: parity and 4x minimum speed result.
- Sentence 6: significance.

Evidence gate:

- Keep `TODO` for final benchmark count and Bio-Formats until implemented.
- Do not mention every backend in the abstract.

## Action 2: Turn Figure 1 Into The Pain Point

Priority: high.

Problem:

- Comparator papers establish why the problem matters before showing the system.
- The figure plan has already moved in this direction, but it should be treated as the first-reader entry point.

Edit:

- Figure 1 title: "Scaled acquisition makes analysis and review the bottleneck."
- Panel A: automated microscopy produces many wells, sites, channels, z-planes, and timepoints.
- Panel B: analysis creates segmentation, measurements, QC, review, reruns, and output tables.
- Panel C: without OpenHCS these split across CellProfiler, Fiji/napari, OMERO/local files, Python scripts, and batch jobs.
- Panel D: OpenHCS keeps those as one workflow record.

Evidence gate:

- Figure should not require knowing CellProfiler internals.
- Avoid compiler/runtime labels in Figure 1.

## Action 3: Add A Direct Positioning Table

Priority: high.

Problem:

- The draft says OpenHCS is not replacing existing tools, but reviewers will still ask how it differs from CellProfiler, Fiji/ImageJ, napari, OMERO, Bio-Formats, and workflow managers.
- A concise table prevents the introduction from becoming an ecosystem catalog.

Placement:

- Discussion or Supplementary Table.

Columns:

- Existing tool/platform.
- Primary strength.
- Boundary where users still hit friction.
- OpenHCS relationship.

Rows:

- CellProfiler: trusted modular bioimage analysis; OpenHCS preserves `.cppipe` workflows, adds named intermediates, Python extension, viewer outputs, workers, and speed.
- Fiji/ImageJ: mature image-processing ecosystem; OpenHCS streams step outputs and can interoperate rather than replace.
- napari: interactive multidimensional viewing; OpenHCS treats viewer output as step-level workflow output.
- OMERO: managed image store; OpenHCS treats managed images as workflow sources.
- Bio-Formats: broad file readability and metadata; planned OpenHCS handler adds workflow-safe source identities where semantics can be inferred.
- Generic workflow managers: scalable/reproducible execution; OpenHCS adds image-domain source dimensions, named intermediates, viewer outputs, CellProfiler semantics, and memory/backend-aware functions.

Evidence gate:

- Keep Bio-Formats row marked planned/TODO until handler is implemented.
- Do not make adversarial claims about existing tools.

## Action 4: Make The Results Read Like A Proof Chain

Priority: high.

Problem:

- The current Results sections are reasonable, but comparable papers are strongest when the reader can see the proof sequence.

Edit sequence:

1. One workflow record across import, viewer output, Python extension, execution, and benchmark.
2. Pre-run checks catch source/function/output/backend mistakes.
3. CellProfiler workflows compile into normal OpenHCS workflows.
4. Parity proves preservation.
5. Single-thread/core CPU speed proves the performance floor.
6. Worker throughput proves scaling.

Evidence gate:

- Each Results section should name what was actually run, tested, or benchmarked.
- Avoid "can" phrasing for implemented features; use "OpenHCS imports", "OpenHCS streams", "benchmarks report".

## Action 5: Separate Platform Claim From CellProfiler Proof

Priority: high.

Problem:

- CellProfiler parity is the strongest validation, but OpenHCS should not read as only a faster CellProfiler runner.

Edit:

- In Abstract and Introduction, state that CellProfiler import is the strictest preservation test.
- In Discussion, state that the broader platform contribution is one workflow record across sources, viewers, Python functions, generated code, backends, and workers.
- In Results, keep CellProfiler parity and speed in dedicated sections after the general workflow claim.

Evidence gate:

- Avoid title or opening sentences that center CellProfiler.
- Keep "composable bioimage workflow platform" as the product identity.

## Action 6: Make Accessibility Concrete

Priority: medium.

Problem:

- Comparator papers like ilastik and BiaPy make accessibility concrete, not abstract.

Edit:

- Add or sharpen examples where a non-technical user can understand the value:
  - inspect a mask by enabling napari or Fiji on a step,
  - change a threshold in the GUI and export the same state as Python,
  - use a vendor plate folder without reorganizing nested folders manually,
  - reuse a trusted `.cppipe` file instead of rewriting it,
  - ask a computational collaborator to add a Python QC function without leaving the workflow.

Evidence gate:

- Avoid saying "no code required" if that overpromises.
- Use "wet-lab user can remain in the GUI; computational collaborator can review generated Python; both views refer to the same workflow."

## Action 7: Tighten Bio-Formats Language

Priority: medium.

Problem:

- Bio-Formats is an important comparator but it solves file readability, not guaranteed workflow-level plate semantics.

Edit:

- Keep TODO language in draft until implementation.
- After implementation, replace with:

> Bio-Formats-backed source discovery extends the microscope-handler system to Bio-Formats-readable datasets by converting series/channel/z/time metadata into normalized OpenHCS source identities, while requiring explicit source schemas when well or site semantics are ambiguous.

Evidence gate:

- Do not claim automatic plate semantics for all Bio-Formats-readable datasets.
- Require tests or fixtures proving normalized source identities and fail-loud ambiguity handling.

## Action 8: Add Expected Reviewer Limitations

Priority: medium.

Problem:

- Broad platform papers are more credible when limitations are explicit.

Edit:

Add a Discussion limitations paragraph covering:

- CellProfiler parity covers the modules/settings represented in the benchmark corpus.
- Backend-specific functions are not automatic CellProfiler replacements.
- Bio-Formats improves readability, but source semantics still require inference or explicit user mapping.
- Persistent workers trade throughput for memory and lifetime management.
- Viewer integration depends on GUI environment availability.

Evidence gate:

- Limitations should be concrete and not apologetic.

## Action 9: Make Benchmark Tables Reviewer-Friendly

Priority: medium.

Problem:

- Comparable benchmark-heavy papers make corpus scope interpretable.

Edit:

- Ensure supplementary benchmark table includes workflow source, assay family, module coverage, parity status, speedup, and notes.
- Ensure module coverage table separates explicitly tested modules, theoretically covered modules, and not-covered modules.
- Make the 4x result visibly a minimum, not just mean/median.

Evidence gate:

- Use final CSVs only.
- Keep parity and speed separate.

## Action 10: Reduce Architecture Jargon In Main Text

Priority: medium.

Problem:

- The current draft still has some implementation terms in main text.

Edit:

- Replace or defer:
  - "runtime artifact" -> "named intermediate result" in Results.
  - "materialization plan" -> "where outputs go" in Results.
  - "FunctionStep" -> "Python function becomes a workflow step" in Results.
  - "ZMQRuntime" -> "worker process" in Results.
- Keep technical names in Methods or Supplement.

Evidence gate:

- A biologist should understand Abstract, Introduction, and Results section openings without knowing OpenHCS internals.

## Proposed Work Order

1. Rewrite abstract to the six-sentence spine.
2. Update Figure 1 and Figure 2 captions/plan.
3. Add positioning table to the draft or supplementary notes.
4. Tighten Results section openings to proof-chain language.
5. Add Discussion positioning and limitations paragraphs.
6. Re-scan for jargon and replace in Abstract/Introduction/Results.
7. Re-check every Bio-Formats claim for TODO/evidence gate status.
8. Re-read against comparator note and MFD accessibility target.

## Done Criteria

- The draft opens with image-analysis bottleneck, not ecosystem naming.
- CellProfiler is framed as the hardest proof, not the product identity.
- Existing tools are treated as valuable and preserved.
- The Results read as an evidence sequence.
- Bio-Formats claims are explicitly gated until implementation.
- A comparison/positioning table exists.
- Main-text architecture terms are reduced or explained through user-facing consequences.
