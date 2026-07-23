# Lab Meeting Benchmark Artifacts - 2026-05-05

This directory stores the CSV inputs used for the May 5 lab-meeting figures.

## Files

- `cppipe_cp_vs_openhcs_summary.csv`: native CellProfiler versus OpenHCS runtime and parity summary for the 18 official example pipelines with native CP timings available.
- `throughput_2v16_samples_1v2_jobs_partial/`: partial throughput run comparing 2 versus 16 samples and 1 versus 2 jobs.

## Throughput Caveat

The throughput CSV snapshot is partial and intentionally preserved as evidence of the current benchmark-harness issue. It launches independent adapter/runtime jobs, so it does not yet measure native OpenHCS well-level multiprocessing inside one compiled plate execution.

The next benchmark mode should construct repeated wells/samples inside the OpenHCS plate abstraction, compile once, then execute through `execute_compiled_plate` with `MULTIPROCESSING_AXIS` ownership. That will measure the actual HCS execution model instead of `ProcessPoolExecutor` around repeated independent adapter invocations.
