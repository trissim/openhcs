Analysis consolidation
======================

OpenHCS can consolidate tabular analysis artifacts after plate execution or as
an explicit desktop task.

From the desktop application, use **Tools > Analysis Consolidation** to:

- consolidate a results directory;
- merge MetaXpress-style summaries;
- concatenate summaries while preserving headers;
- run configured experimental analysis.

Automatic post-plate consolidation is controlled by
``AnalysisConsolidationConfig`` on the pipeline. It selects file extensions,
well extraction, exclusions, and output names. Consolidation consumes
materialized analysis artifacts; it is not an image-processing ``FunctionStep``
and does not reconstruct compiler state.

Keep artifact producers explicit. A callable or module declares its outputs,
the compiler plans their materialization, and consolidation reads the resulting
files after successful execution. See
:doc:`../architecture/analysis_consolidation_system`.
