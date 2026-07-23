# Official30 Presentation Figure Pack

Source data:
- Parity/single-process: `benchmark/results/labmeeting_20260513/official30_well_throughput/data/single_process_summary.csv`
- 1/2/3/4 core comparison: `benchmark/results/labmeeting_20260513/official30_well_throughput/data/core_scaling_well_throughput.csv`
- Variable wells/core comparison: `benchmark/results/labmeeting_20260513/official30_well_throughput/data/wells_per_core_2c3c4c.csv`
- Additional wells/core comparison: `benchmark/results/labmeeting_20260513/official30_well_throughput/data/wells_per_core_4c_6wpc_8wpc.csv`
- Module coverage: `benchmark/results/labmeeting_20260513/official30_well_throughput/figures/module_coverage_semantic_families.csv`

Figure groups:
- `01_parity_by_pipeline.*`
- `02_core_scaling_by_pipeline_plus_average_speedup.*`
- `03_core_scaling_average_with_pipeline_points_speedup.*`
- `04_core_scaling_by_pipeline_plus_average_ram.*` for per-pipeline RAM.
- `04_core_scaling_average_with_pipeline_points_ram.*` for aggregate RAM.
- `05_speedup_summary_by_core_and_wells_per_core.*`
- `06_module_coverage_by_abstraction.*`

Interpretation notes:
- The `1 core` point is same-process, single-well execution latency.
- The `2/3/4 cores` points are native OpenHCS multiprocessing throughput runs over replicated wells.
- Very small pipelines can look non-monotonic in `02_core_scaling_by_pipeline_plus_average_speedup.*` because fixed fork/work-queue/file overhead is not amortized at low well counts.
- Use `05_speedup_summary_by_core_and_wells_per_core.*` to assess throughput scaling as queue depth increases.
