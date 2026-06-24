# Benchmark Manifests

`official30_portable_axis1.json` is the reproducible 30-case CP-vs-OpenHCS
benchmark manifest. It avoids case-level absolute paths by declaring named,
self-materializing roots:

- `CELLPROFILER_EXAMPLES_ROOT`: sparse checkout of official CellProfiler example
  pipelines and datasets. Defaults to `~/.cache/openhcs/cellprofiler_examples`.
- `OPENHCS_BENCHMARK_DATASET_CACHE_ROOT`: auto-acquired dataset cache containing
  registry-backed tutorial/supplement sources. Defaults to
  `~/.cache/openhcs/benchmark_datasets`.
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
python scripts/benchmark_cellprofiler_vs_openhcs.py run \
  --manifest benchmark/manifests/official30_portable_axis1.json \
  --output-dir /tmp/openhcs_cp30_run \
  --native-reference-root /tmp/openhcs_cp_native_refs \
  --no-memory-metric \
  --speedup-target 4
```

The native adapter resolves CellProfiler from an explicit executable path,
`CELLPROFILER_EXECUTABLE`, `PATH`, the active Python environment, or tool
environment roots declared through `OPENHCS_BENCHMARK_TOOL_ROOTS`. Local
workspace siblings are scanned through the same tool-root contract, so a
development checkout with `.venv-cellprofiler39`, `.venv-cellprofiler`, or
`.venv` does not need per-run shell setup.

The run emits parity/timing CSVs, phase timing, suite metadata, v7-style figures,
and module coverage artifacts including concrete setting coverage and semantic
family coverage.
