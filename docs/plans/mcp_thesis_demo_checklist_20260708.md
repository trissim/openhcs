# OpenHCS MCP Thesis Demo Checklist

## Live Command

Run the no-fallback live rehearsal from `/home/ts/code/projects/openhcs`:

```bash
scripts/mcp_thesis_demo_live.py --runs 3
```

Use `--allow-dirty` only while validating uncommitted local changes:

```bash
scripts/mcp_thesis_demo_live.py --runs 3 --allow-dirty
```

This is the demo authority. It starts the real ZMQ execution server, starts the
real PyQt UI with the UI bridge enabled, drives the UI code-document bridge,
compiles and executes through the canonical ZMQ compiler/orchestrator path,
validates the Napari server, inspects layer payload provenance, and captures UI
and viewer snapshots.

## Verified Result

Verified on 2026-07-08:

```text
mcp_outputs/thesis_demo/live/rehearsals/20260708_020400
```

Three consecutive live runs passed:

```text
run1_20260708_020400: 78.71s, fresh UI/ZMQ restart
run2_20260708_020519: 64.83s, reused live stack
run3_20260708_020624: 65.55s, reused live stack
```

All runs were under the 4 minute live-demo budget. Each run recorded command
JSON evidence under `run_XX/commands/` and a human summary in `summary.md`.

## Required Flow

The script hard-fails unless all of these pass:

1. MCP health is reachable.
2. ZMQ execution server is ready on `7777`.
3. UI bridge descriptor is live and reachable.
4. Generic ZMQ runtime scan sees both `ZMQExecutionServer` and `OpenHCSUiBridgeServer`.
5. `plate_manager.orchestrator_config` is listed, inspected, validated, applied, and reread through the UI bridge.
6. The applied source contains only public OpenHCS state: `plate_paths`, `GlobalPipelineConfig`, per-plate `PipelineConfig`, and `pipeline_data` as `dict[str, list[FunctionStep]]`.
7. Selected plate workflows complete: `init_plate`, `compile_plate`, `run_plate`.
8. The output plate exists and contains TIFF outputs.
9. Napari viewer validates with nonzero payloads and required component labels.
10. Viewer layer navigation and payload inspection return provenance/component context.
11. Plate manager and Napari snapshots are captured.

## Demo Script

For the live seminar, show the UI first, then run:

```bash
scripts/mcp_thesis_demo_live.py --runs 1
```

Expected fresh-run timing is about 80 seconds on the validated machine. The
script leaves the UI, ZMQ server, and Napari process open after success so the
committee can inspect the visible UI, the ZMQ server list, the code document,
the selected plate state, and the Napari layers.

## No Fallback Path

Do not substitute the old smoke script, a benchmark, or a headless-only run for
the thesis demo. A failure means the live demo path is not ready and the failing
JSON evidence file should be inspected directly.
