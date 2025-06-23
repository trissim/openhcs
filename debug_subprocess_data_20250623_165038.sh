#!/bin/bash
# OpenHCS Subprocess Debug Script
# Generated: 2025-06-23 16:50:49.067277
# Plates: ['/home/ts/nvme_usb/OpenHCS/mfd-hips-cell-density-ctb-4x_Plate_12994_workspace_outputs']

echo "🔥 Starting OpenHCS subprocess debugging..."
echo "🔥 Pickle file: debug_subprocess_data_20250623_165038.pkl"
echo "🔥 Press Ctrl+C to stop"
echo ""

cd "/home/ts/code/projects/openhcs"

python "/home/ts/code/projects/openhcs/openhcs/textual_tui/subprocess_runner.py" \
    "debug_subprocess_data_20250623_165038.pkl" \
    "debug_status.json" \
    "debug_result.json" \
    "debug.log"

echo ""
echo "🔥 Subprocess finished. Check the files:"
echo "  - debug_status.json (progress/death markers)"
echo "  - debug_result.json (final results)"
echo "  - debug.log (detailed logs)"
