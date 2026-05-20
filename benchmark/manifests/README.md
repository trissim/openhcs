# Benchmark Manifests

`official30_portable_axis1.json` is the reproducible 30-case CP-vs-OpenHCS
benchmark manifest. It avoids case-level absolute paths by declaring named,
self-materializing roots:

- `CELLPROFILER_EXAMPLES_ROOT`: sparse checkout of official CellProfiler example
  pipelines and datasets. Defaults to `/tmp/cellprofiler_examples`.
- `OPENHCS_BENCHMARK_DATASET_CACHE_ROOT`: auto-acquired dataset cache containing
  registry-backed tutorial/supplement sources. Defaults to
  `/tmp/openhcs_benchmark_dataset_cache_last8`.
- `OPENHCS_AXISONE_SUBSETS_ROOT`: compatibility root for older axis-one
  manifests. The official 30-case manifest uses full dataset-cache roots and
  applies `openhcs_max_axis_count` as a well/sample selector so all sites and
  channels for the selected sample remain present.

The benchmark manifest loader materializes missing acquisition-enabled roots
before resolving case paths. Set `OPENHCS_BENCHMARK_AUTO_ACQUIRE=0` to disable
this and require pre-existing files.

Build or refresh registry-backed datasets directly with:

```bash
python scripts/prepare_cellprofiler_benchmark_datasets.py manifest \
  --cache-root "$OPENHCS_BENCHMARK_DATASET_CACHE_ROOT" \
  --output /tmp/registry_cases.json
```

Run the portable manifest with:

```bash
CELLPROFILER_EXECUTABLE=/path/to/cellprofiler \
python scripts/benchmark_cellprofiler_vs_openhcs.py run \
  --manifest benchmark/manifests/official30_portable_axis1.json \
  --output-dir /tmp/openhcs_cp30_run \
  --native-reference-root /tmp/openhcs_cp_native_refs \
  --no-memory-metric \
  --speedup-target 4
```

The run emits parity/timing CSVs, phase timing, suite metadata, v7-style figures,
and module coverage artifacts including concrete setting coverage and semantic
family coverage.
