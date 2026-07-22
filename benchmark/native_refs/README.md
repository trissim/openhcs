# Native CellProfiler Reference Cache

This directory stores setup-local CellProfiler reference outputs used by the
OpenHCS benchmark harness.

The cache is intentionally not portable. It is a backup of native reference
outputs generated on this machine so benchmark runs can be resumed after `/tmp`
is lost without re-running CellProfiler.

Current cache:

- `official30_scoped_rows/`
- Source copied from: `/tmp/openhcs_cp_native_refs_official30_scoped_rows`
- Manifest: `benchmark/manifests/official30_portable_axis1.json`
- Scope: first sampled well per case, official30 benchmark reference outputs
- Scope identity: the selected-well owner derives `wells_include_first1`; each
  case directory carries the matching schema-1 completion/provenance marker.
- File count at copy time: 248
- Size at copy time: about 67 MB
- Platform at copy time: `Linux-7.0.3-arch1-2-x86_64-with-glibc2.43`
- Python used by OpenHCS venv at copy time: `3.11.11`

Use with:

```bash
.venv/bin/python scripts/benchmark_cellprofiler_vs_openhcs.py run \
  --manifest benchmark/manifests/official30_portable_axis1.json \
  --native-reference-root benchmark/native_refs/official30_scoped_rows \
  --require-native-reference
```
