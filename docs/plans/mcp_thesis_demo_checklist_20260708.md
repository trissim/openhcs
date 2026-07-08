# OpenHCS MCP Thesis Demo Checklist

## Verified Baseline

Verified on 2026-07-08 from `/home/ts/code/projects/openhcs`:

```bash
scripts/mcp_thesis_demo_smoke.sh
scripts/mcp_thesis_demo_smoke.sh --ui-descriptor mcp_outputs/thesis_demo/ui_bridge/ui_bridge_ui-85b00ab1-7adf-4333-8c1e-bd748f196ff4.json
```

This checks MCP health, finds the persistent ZMQ runtime server, generates a
durable synthetic plate under `mcp_outputs/thesis_demo/plate`, writes a
list-only `pipeline_steps` source file, and executes it through MCP into the
canonical ZMQ compiler/executor path. With a pinned descriptor it also checks UI
bridge status, windows, and code-document discovery. The verified output plate
root was:

```text
mcp_outputs/thesis_demo/plate_openhcs
```

## Demo Startup

Start or confirm the persistent ZMQ execution server:

```bash
python -m openhcs.mcp.dev_client runtime-scan --timeout-seconds 20 --json
```

Run the headless MCP smoke:

```bash
scripts/mcp_thesis_demo_smoke.sh
```

For a UI demo, start the UI with bridge descriptor output:

```bash
OPENHCS_ENABLE_UI_BRIDGE=true \
OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR="$PWD/mcp_outputs/thesis_demo/ui_bridge" \
python -m openhcs.pyqt_gui --log-level WARNING
```

Pin the descriptor explicitly when multiple UIs may be running:

```bash
DESCRIPTOR="$(ls -t mcp_outputs/thesis_demo/ui_bridge/ui_bridge_*.json | head -n 1)"
python -m openhcs.mcp.dev_client ui-status --descriptor-file-path "$DESCRIPTOR" --json
python -m openhcs.mcp.dev_client windows --descriptor-file-path "$DESCRIPTOR" --json
python -m openhcs.mcp.dev_client code-documents --descriptor-file-path "$DESCRIPTOR" --json
```

## UI Code-Document Demo

Use `plate_manager.orchestrator_config` as the user-visible bridge between the
UI and MCP. The document should contain public OpenHCS state only:

- `plate_paths`
- `global_config`
- `per_plate_configs`
- `pipeline_data` as `dict[str, list[FunctionStep]]`

It should not import or construct `Pipeline`. The compiler/orchestrator path
receives `PipelineConfig` plus the `FunctionStep` list.

## Talking Points

- OpenHCS exposes a typed analysis state to MCP instead of screen scraping the
  GUI.
- MCP can create or inspect pipeline source, validate it, and send it through
  the same compiler/executor path as UI and headless execution.
- The public pipeline definition is `PipelineConfig + list[FunctionStep]`; UI
  metadata carriers stay inside the UI/ObjectState boundary.
- The demo plate and outputs are under `mcp_outputs/`, not `/tmp`, so the setup
  is reproducible across shell sessions.

## Fallbacks

- If `ui-status` says the UI bridge is unavailable, restart the UI with
  `OPENHCS_ENABLE_UI_BRIDGE=true` and pin the newest descriptor with
  `--descriptor-file-path`.
- If `runtime-scan` cannot find port `7777`, start the ZMQ runtime from the UI
  server manager or launch the runtime server before running the smoke script.
- If the visible UI is already open without a bridge descriptor, do not rely on
  discovery. Restart it with the descriptor directory above or use a known
  descriptor path.
