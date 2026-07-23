Experimental analysis system
============================

Experimental analysis consumes consolidated measurements plus an experimental
layout. It joins measurements to plate annotations, applies the selected
normalization, and emits analysis tables and visual summaries.

The implementation in ``openhcs.formats.experimental_analysis`` owns layout
parsing, well identity normalization, control selection, normalization methods,
and result generation. The desktop workflow supplies explicit input/output
paths and configuration.

This layer runs after image processing. It does not define pipeline axes,
``group_by`` behavior, callable locality, or artifact satisfaction. Those remain
compiler/runtime concepts. Its input should be a stable materialized table so
the statistical transformation is reproducible and auditable.

See :doc:`analysis_consolidation_system` and
:doc:`../user_guide/experimental_layouts`.
