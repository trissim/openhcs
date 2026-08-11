Analysis consolidation system
=============================

Analysis consolidation is a post-plate operation over materialised tabular
artifacts. It is separate from callable execution and from the compiler's
artifact graph.

``AnalysisConsolidationConfig`` is resolved into each processing context.
After a compiled plate finishes successfully,
``consolidate_analysis_outputs`` reads the configuration from the compiled
contexts, projects the exact table files recorded in that execution's runtime
observations, groups them by materialisation directory, and delegates to the
analysis consolidation functions. It does not discover inputs by scanning a
results directory.

Ownership
---------

- callable or module declarations own artifact production semantics;
- the compiler owns artifact satisfaction and materialisation plans;
- the runtime materialises outputs at the declared locations and records their
  typed addresses;
- the consolidation layer owns well extraction, exclusions, merging, and
  summary filenames.

Consolidation never infers missing artifacts from step names and does not mutate
compiled plans. Manual desktop consolidation uses the same analysis functions
on an explicitly selected results directory.

See :doc:`artifact_contract_system` and
:doc:`../user_guide/analysis_consolidation`.
