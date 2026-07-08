#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEMO_ROOT="${OPENHCS_DEMO_ROOT:-$ROOT/mcp_outputs/thesis_demo}"
PLATE_DIR="$DEMO_ROOT/plate"
PIPELINE_SOURCE="$DEMO_ROOT/demo_pipeline.py"
ZMQ_PORT="${OPENHCS_ZMQ_PORT:-7777}"

mkdir -p "$DEMO_ROOT"

"$PYTHON_BIN" -m openhcs.mcp.dev_client health --timeout-seconds 20 --json
"$PYTHON_BIN" -m openhcs.mcp.dev_client runtime-scan --timeout-seconds 20 --json

"$PYTHON_BIN" -m openhcs.mcp.dev_client generate-synthetic-plate "$PLATE_DIR" \
  --grid-rows 1 \
  --grid-cols 1 \
  --tile-width 64 \
  --tile-height 64 \
  --wavelengths 2 \
  --z-stack-levels 1 \
  --num-cells 8 \
  --well A01 \
  --openhcs-format \
  --random-seed 17 \
  --sample-file-limit 5 \
  --json

cat > "$PIPELINE_SOURCE" <<'PY'
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.numpy_processor import gaussian_blur

pipeline_steps = [
    FunctionStep(
        func=(gaussian_blur, {"sigma": 1.0}),
        name="MCP Demo Gaussian Blur",
    )
]
PY

"$PYTHON_BIN" -m openhcs.mcp.dev_client execute-source "$PLATE_DIR" \
  --source-file "$PIPELINE_SOURCE" \
  --port "$ZMQ_PORT" \
  --wait \
  --submit-timeout-ms 15000 \
  --wait-timeout-ms 60000 \
  --json

if [[ "${1:-}" == "--ui-descriptor" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "usage: $0 --ui-descriptor /path/to/ui_bridge_*.json" >&2
    exit 2
  fi
  DESCRIPTOR_PATH="$2"
  "$PYTHON_BIN" -m openhcs.mcp.dev_client ui-status \
    --descriptor-file-path "$DESCRIPTOR_PATH" \
    --timeout-seconds 20 \
    --json
  "$PYTHON_BIN" -m openhcs.mcp.dev_client windows \
    --descriptor-file-path "$DESCRIPTOR_PATH" \
    --timeout-seconds 20 \
    --json
  "$PYTHON_BIN" -m openhcs.mcp.dev_client code-documents \
    --descriptor-file-path "$DESCRIPTOR_PATH" \
    --timeout-seconds 20 \
    --json
fi
