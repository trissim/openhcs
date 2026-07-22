# Native CellProfiler Reference Cache

This directory stores the committed native CellProfiler reference outputs used
by the OpenHCS benchmark harness.

The result payloads and schema-1 completion/provenance markers are portable
parity evidence. Regenerating them still requires the acquired source assets,
and any runtime measurements made while regenerating them remain
machine-specific rather than acceptance evidence.

Current cache:

- `official30_scoped_rows/`
- Source copied from: `/tmp/openhcs_cp_native_refs_official30_scoped_rows`
- Manifest: `benchmark/manifests/official30_portable_axis1.json`
- Scope: first sampled well per case, official30 benchmark reference outputs
- Scope identity: the selected-well owner derives `wells_include_first1`; each
  case directory carries the matching schema-1 completion/provenance marker.
- Committed file count: 230
- Committed size: about 65 MiB
- Platform at copy time: `Linux-7.0.3-arch1-2-x86_64-with-glibc2.43`
- Python used by OpenHCS venv at copy time: `3.11.11`

Use with:

```bash
.venv/bin/python scripts/benchmark_cellprofiler_vs_openhcs.py run \
  --manifest benchmark/manifests/official30_portable_axis1.json \
  --output-dir /tmp/openhcs_official30_parity \
  --native-reference-root benchmark/native_refs/official30_scoped_rows \
  --require-native-reference
```
