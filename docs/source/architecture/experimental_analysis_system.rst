Experimental analysis system
============================

Experimental analysis is a standalone post-processing workflow over a
materialised measurement summary and an experimental layout workbook. It joins
measurements to plate annotations, applies control-based normalisation, and
emits analysis tables and visual summaries. It does not participate in pipeline
compilation or execution.

Current ownership
-----------------

``ExperimentalAnalysisConfig`` owns the standalone engine options and all five
filenames used by the directory workflow. It is deliberately not a global or
pipeline configuration. ``ExperimentalAnalysisEngine`` owns format selection
and orchestration; its ``run_directory`` projection derives every input and
output path from that configuration. The desktop action constructs the default
configuration and calls the same engine boundary rather than restating those
filenames or using the deprecated compatibility wrapper.

``NormalizationMethod`` owns the fold-change, control z-score, and
percentage-of-control leaf calculations. The shared analysis module traverses
the experiment once and applies that declaration against each biological
replicate's own control distribution; it does not infer control semantics from
condition-name strings. ``ExperimentalResultFormatStrategy`` leaves own the
CX5 and MetaXpress result readers selected by the workbook's
``ExperimentalAnalysisScope`` declaration. The shared analysis module owns
layout parsing, exclusion projection, and table/plate-grid export.

Pipeline boundary
-----------------

This layer runs after image processing. It does not define pipeline axes,
``group_by`` behaviour, callable locality, or artifact satisfaction. Those
remain compiler and runtime concepts. Its input is a stable materialised table
so the statistical transformation can be reproduced and audited.

See :doc:`analysis_consolidation_system` and
:doc:`../user_guide/experimental_layouts`.
