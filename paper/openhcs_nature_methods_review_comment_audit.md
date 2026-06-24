# OpenHCS Nature Methods Review Comment Audit

Source review: `/home/ts/code/projects/mfd/papers/mfd_platform/reviews/openhcs_nature_methods_draft_review[99].docx`  
Audited manuscript: `paper/openhcs_nature_methods_draft.md`  
Status: all extracted comments are addressed in the revised draft or converted into explicit verification scope.

## Coverage Summary

| Comment | Audit result |
|---|---|
| 2 | Abstract now states open source, BYO source scope, CellProfiler import, napari/Fiji routing, CPU/GPU/deep-learning Python functions, workers, parity, and speedup. |
| 3 | “Should scale” phrasing was replaced by acquisition-outpaces-analysis and bottleneck language. |
| 4 | Bottleneck framing is now tied to Figure 1 and the one-workflow Results section. |
| 5 | Tool ecosystem framing now says existing tools solve distinct parts; OpenHCS connects workflow boundaries rather than becoming another standard. |
| 6 | Non-opinionated/pluralistic positioning appears in the Introduction and interoperability table. |
| 7 | The “boundary problem” and workflow graph are defined before detailed feature lists. Dense list sentences were reduced or moved later. |
| 8 | Abstract is now three paragraphs and no longer carries the detailed CellProfiler explanation. |
| 11 | First Results heading now names the concrete object: one runnable workflow. |
| 12 | Results headings now foreground workflow, parity, speed, interoperability, and throughput. |
| 13 | “Should scale” phrasing was removed. |
| 14 | The representative-run architecture sentence was replaced by a Figure 1 reference and figure caption. |
| 15 | Version-specific parity artifacts and CellProfiler/OpenHCS version recording are stated in Results and Methods. |
| 17 | Cloud/network-filesystem performance is bounded in Methods; no cloud-bursting claim is made without a recorded benchmark. |
| 18 | Inheritance wording now says entering a value overrides and clearing a value returns to inherited state. |
| 19 | “Wet-lab user” was replaced with “domain expert” and “computational collaborator.” |
| 21 | Backend library mechanics moved out of early Results; Results now lead with workflow, parity, and 4.03x speedup. |
| 22 | Workflow graph/DAG language is now explicit in Introduction, Results, and Online Methods. |
| 24 | Abstract and Discussion now use backwards-compatible preservation and extension framing. |
| 25 | A main-text interoperability table was added. |
| 27 | “Mature workflow format” was replaced with “widely used, trusted workflow format.” |
| 28 | Future CellProfiler versions are handled through version-specific parity artifacts and declared compatibility checks. |
| 30 | OpenHCS documentation link was added to Code and Data Availability. |
| 32 | The at-least-4x result now appears in the Abstract and early Results. |
| 33 | Figure 2 covers benchmarking; GPU support is scoped as method support unless a separate benchmark defines hardware, backend version, reference behavior, and comparison policy. |
| 36 | Provenance is now central in the Abstract, Introduction, Results, and Discussion. |
| 37 | Maintenance/backwards-compatible framing was promoted to the Abstract and Discussion. |
| 38 | Open-source status now appears in the Abstract. |
| 39 | The comparison table was rewritten as a preservation/integration/validation-status table to avoid disparaging existing tools. |
| 45 | Specific decorator examples were replaced by backend-annotation language in Methods. |
| 52 | GPU questions are answered by explicit CPU-only benchmark scope and GPU benchmark requirements. |

## Style-Guide Check

The revised draft follows the adapted paper style guide:

- Direct, declarative sentences replace meta-paper narration.
- Main claims are paired with exact scope: workflow count, benchmark condition, parity policy, and CPU-only execution.
- The strongest verified result is visible in the Abstract and early Results.
- Tool comparisons are handled in a table with neutral preservation/integration language.
- Potential overclaims around GPU, cloud execution, and future CellProfiler versions are bounded by verification requirements.
