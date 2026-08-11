Experimental layouts
====================

Experimental analysis maps well-level measurements to plate metadata such as
treatment, concentration, time point, and controls. Configure the layout in the
desktop analysis workflow, then run analysis on already consolidated results.

Run an experimental analysis
----------------------------

1. Start with the downloadable workbook in
   :doc:`../reference/experimental_config_syntax` and enter the design on its
   ``drug_curve_map`` and ``plate_groups`` sheets.
2. Confirm that every workbook well identifier matches the consolidated result
   exactly. Keep the raw consolidated table for comparison.
3. Put the workbook at ``config.xlsx`` and the consolidated table at
   ``metaxpress_style_summary.csv`` in one analysis directory.
4. Choose **Tools > Analysis Consolidation > Run Experimental Analysis** and
   select that directory.
5. Inspect ``compiled_results_normalized.xlsx``, ``compiled_results_raw.xlsx``,
   and the conditionally formatted plate grids in ``heatmaps.xlsx`` beside the
   inputs. If analysis fails, correct the first workbook or input-format error
   before changing the pipeline.

A layout is data, not a pipeline axis declaration. It must not change the
meaning of ``variable_components`` or ``group_by``. The processing pipeline
produces typed measurement artifacts; the experimental-analysis layer joins
those measurements with the layout. The desktop workflow uses replicate-local
fold-change normalization by default. Programmatic callers can instead select
control z-scores or percentage-of-control through
``ExperimentalAnalysisConfig.normalization_method``.

Validate that well identifiers in the layout use the same convention as the
consolidated table before running. Keep raw consolidated results alongside any
normalized output so the transformation remains auditable.

See :doc:`analysis_consolidation` and
:doc:`../architecture/experimental_analysis_system`.
