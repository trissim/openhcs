Analysis consolidation system
=============================

Analysis consolidation is a post-plate operation over materialised tabular
artifacts. It is separate from callable execution and does not reopen an output
path to rediscover content already recorded by the runtime.

``AnalysisConsolidationConfig`` is resolved into each processing context.
After a compiled plate finishes successfully,
``consolidate_analysis_outputs`` reads the configuration from the compiled
contexts, projects the exact CSV content recorded in that execution's runtime
observations, groups it by materialisation directory, and delegates to the
analysis consolidation functions. The compiled persistent target supplies one
backend-owned summary destination. Consolidation therefore neither scans a
results directory nor treats a virtual backend address as a host file.

The generic consolidation layer reads ``AnalysisTableSource`` values and writes
through an ``AnalysisSummaryWriter`` ABC. Local desktop consolidation uses a
filesystem writer. Compiled execution uses a FileManager writer bound to the
exact persistent backend and obtains save context through that backend's
``DataSink`` declaration. Per-result and global summaries share this destination;
the global table is merged in memory rather than read back from storage.

Ownership
---------

- callable or module declarations own artifact production semantics;
- the compiler owns artifact satisfaction and materialisation plans;
- the runtime records typed addresses and values produced by the exact
  execution;
- each ``DataSink`` owns contextual persistence arguments for its backend;
- the consolidation layer owns well extraction, exclusions, merging, summary
  rendering, and summary filenames.

Consolidation never infers missing artifacts from step names and does not mutate
compiled plans. A source or destination failure is reported as a plate execution
failure rather than logged and discarded. Manual desktop consolidation uses the
same analysis functions on an explicitly selected results directory.

See :doc:`artifact_contract_system` and
:doc:`../user_guide/analysis_consolidation`.
