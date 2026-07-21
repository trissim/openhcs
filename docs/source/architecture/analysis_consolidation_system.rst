Analysis consolidation system
=============================

Analysis consolidation is a post-plate operation over materialized tabular
artifacts. It is separate from callable execution and from the compiler's
artifact graph.

``AnalysisConsolidationConfig`` is resolved into each processing context.
After a compiled plate finishes successfully,
``consolidate_analysis_outputs`` reads the configuration from the compiled
contexts, discovers the materialized analysis directories, and delegates to the
analysis consolidation functions.

Ownership
---------

- callable or module declarations own artifact production semantics;
- the compiler owns artifact satisfaction and materialization plans;
- the runtime materializes outputs at the declared locations;
- the consolidation layer owns well extraction, exclusions, merging, and
  summary filenames.

Consolidation never infers missing artifacts from step names and does not mutate
compiled plans. Manual desktop consolidation uses the same analysis functions
on an explicitly selected results directory.

See :doc:`artifact_contract_system` and
:doc:`../user_guide/analysis_consolidation`.
