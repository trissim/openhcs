Experimental layouts
====================

Experimental analysis maps well-level measurements to plate metadata such as
treatment, concentration, time point, and controls. Configure the layout in the
desktop analysis workflow, then run analysis on already consolidated results.

A layout is data, not a pipeline axis declaration. It must not change the
meaning of ``variable_components`` or ``group_by``. The processing pipeline
produces typed measurement artifacts; the experimental-analysis layer joins
those measurements with the layout and computes the selected normalization and
summary outputs.

Validate that well identifiers in the layout use the same convention as the
consolidated table before running. Keep raw consolidated results alongside any
normalized output so the transformation remains auditable.

See :doc:`analysis_consolidation` and
:doc:`../architecture/experimental_analysis_system`.
