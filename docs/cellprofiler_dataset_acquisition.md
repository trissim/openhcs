# CellProfiler Dataset Acquisition

Benchmark datasets are declared in `benchmark.datasets.registry.DATASET_CATALOG`.
Each row owns the dataset identity, acquisition source, validation rule, and any
dataset-relative `.cppipe` benchmark cases.

## List Available Datasets

```bash
source ../openhcs/.venv/bin/activate
python scripts/prepare_cellprofiler_benchmark_datasets.py list --with-cases
```

## Acquire Data And Generate A Manifest

```bash
source ../openhcs/.venv/bin/activate
python scripts/prepare_cellprofiler_benchmark_datasets.py manifest \
  --dataset-id CellProfiler_tutorials \
  --output ~/.cache/openhcs/cellprofiler_tutorials_manifest.json
```

The generated JSON is directly consumable by:

```bash
python scripts/benchmark_cellprofiler_vs_openhcs.py run \
  --manifest ~/.cache/openhcs/cellprofiler_tutorials_manifest.json \
  --output-root /tmp/openhcs_cellprofiler_tutorials_results
```

## Extension Points

Acquisition and validation are registry-backed strategy families in
`benchmark.datasets.acquire`.

To add another source type, define a `DatasetSourceKind`, a `DatasetSourceSpec`,
and a `DatasetSourceHandler` subclass keyed by `source_kind`.

To add another validation rule, define a `DatasetValidationRule` and a
`DatasetValidationStrategy` subclass keyed by `validation_rule`.
