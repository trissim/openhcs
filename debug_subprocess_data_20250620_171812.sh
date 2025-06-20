#!/bin/bash
# OpenHCS Subprocess Debug Script
# Generated: 2025-06-20 17:18:25.077729
# Plates: ['/home/ts/nvme_usb/IMX/mar-20-axotomy-fca-dmso/mar-20-axotomy-fca-dmso-Plate-2_Plate_13054_all/mar-20-axotomy-fca-dmso-Plate-2_Plate_13054']

echo "🔥 Starting OpenHCS subprocess debugging..."
echo "🔥 Pickle file: debug_subprocess_data_20250620_171812.pkl"
echo "🔥 Press Ctrl+C to stop"
echo ""

cd "/home/ts/code/projects/openhcs"

python "/home/ts/code/projects/openhcs/openhcs/textual_tui/subprocess_runner.py" \
    "debug_subprocess_data_20250620_171812.pkl" \
    "debug_status.json" \
    "debug_result.json" \
    "debug.log"

echo ""
echo "🔥 Subprocess finished. Check the files:"
echo "  - debug_status.json (progress/death markers)"
echo "  - debug_result.json (final results)"
echo "  - debug.log (detailed logs)"
