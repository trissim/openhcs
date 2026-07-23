# Lab Meeting Benchmark Artifacts - 2026-05-13

This directory stores the compact CSV and figure artifacts used for the May 13
official30 CellProfiler-vs-OpenHCS benchmark discussion.

## Contents

- `official30_well_throughput/data/single_process_summary.csv`: single-process
  parity and timing summary for the official30 manifest.
- `official30_well_throughput/data/core_scaling_well_throughput.csv`: source
  CSV for the `1/2/3/4 core` presentation figures.
- `official30_well_throughput/data/wells_per_core_2c3c4c.csv`: source CSV for
  the `1..4 wells/core` queue-depth sweep on 2, 3, and 4 cores.
- `official30_well_throughput/data/wells_per_core_4c_6wpc_8wpc.csv`: additional
  4-core queue-depth source rows for 6 and 8 wells/core.
- `official30_well_throughput/figures/`: reusable presentation figure pack
  generated from the CSV inputs above.

## Interpretation Caveat

The `1 core` point in the core-scaling figures is same-process, single-well
execution latency. The `2/3/4 cores` points are native OpenHCS multiprocessing
throughput runs over replicated wells.

That distinction matters for very small pipelines. If a single well executes in
tens of milliseconds, fixed fork/work-queue/file overhead may dominate at low
well counts. Those pipelines can appear non-monotonic in
`02_core_scaling_by_pipeline_plus_average_speedup.*` even when throughput
improves with larger queues. Use
`05_speedup_summary_by_core_and_wells_per_core.*` to assess amortized throughput
scaling.

## Regeneration

```bash
../openhcs/.venv/bin/python scripts/benchmark_cellprofiler_vs_openhcs.py \
  plot-well-throughput-presentation \
  --single-process-summary-csv benchmark/results/labmeeting_20260513/official30_well_throughput/data/single_process_summary.csv \
  --core-scaling-csv benchmark/results/labmeeting_20260513/official30_well_throughput/data/core_scaling_well_throughput.csv \
  --wells-per-core-csv benchmark/results/labmeeting_20260513/official30_well_throughput/data/wells_per_core_2c3c4c.csv \
  --additional-wells-per-core-csv benchmark/results/labmeeting_20260513/official30_well_throughput/data/wells_per_core_4c_6wpc_8wpc.csv \
  --output-dir benchmark/results/labmeeting_20260513/official30_well_throughput/figures
```
