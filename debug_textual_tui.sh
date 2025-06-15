#!/bin/bash

# Debug script for OpenHCS Textual TUI
# Runs the TUI in debug mode and captures logs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 OpenHCS Textual TUI Debug Runner${NC}"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "openhcs/textual_tui/__main__.py" ]; then
    echo -e "${RED}❌ Error: Not in OpenHCS project directory${NC}"
    echo "Please run this script from the OpenHCS project root"
    exit 1
fi

# Create logs directory
LOG_DIR="/tmp/openhcs_debug"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/textual_tui_debug_$TIMESTAMP.log"

echo -e "${YELLOW}📁 Log directory: $LOG_DIR${NC}"
echo -e "${YELLOW}📄 Log file: $LOG_FILE${NC}"
echo ""

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    # Kill any background TUI processes
    pkill -f "python -m openhcs.textual_tui" 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT

echo -e "${BLUE}🚀 Starting OpenHCS Textual TUI in debug mode...${NC}"
echo -e "${YELLOW}💡 Press Ctrl+C to stop and view logs${NC}"
echo ""

# Run the TUI in debug mode and capture output
python -m openhcs.textual_tui --debug 2>&1 | tee "$LOG_FILE"

echo ""
echo -e "${GREEN}✅ TUI session ended${NC}"
echo ""
echo -e "${BLUE}📊 Log Analysis:${NC}"
echo "=================="

# Show log file size
LOG_SIZE=$(wc -l < "$LOG_FILE")
echo -e "${YELLOW}📏 Total log lines: $LOG_SIZE${NC}"

# Show key events in logs
echo ""
echo -e "${BLUE}🔍 Key Events Found:${NC}"

# Check for plate manager events
PLATE_EVENTS=$(grep -i "plate\|watch_plates\|_on_plate_directory_selected" "$LOG_FILE" | wc -l)
if [ "$PLATE_EVENTS" -gt 0 ]; then
    echo -e "${GREEN}📋 Plate Manager Events: $PLATE_EVENTS${NC}"
    echo "Recent plate events:"
    grep -i "plate\|watch_plates\|_on_plate_directory_selected" "$LOG_FILE" | tail -5 | sed 's/^/  /'
else
    echo -e "${RED}❌ No plate manager events found${NC}"
fi

echo ""

# Check for file browser events
BROWSER_EVENTS=$(grep -i "file.*browser\|directory.*selected\|enhanced.*file" "$LOG_FILE" | wc -l)
if [ "$BROWSER_EVENTS" -gt 0 ]; then
    echo -e "${GREEN}📁 File Browser Events: $BROWSER_EVENTS${NC}"
    echo "Recent browser events:"
    grep -i "file.*browser\|directory.*selected\|enhanced.*file" "$LOG_FILE" | tail -5 | sed 's/^/  /'
else
    echo -e "${RED}❌ No file browser events found${NC}"
fi

echo ""

# Check for errors
ERROR_COUNT=$(grep -i "error\|exception\|traceback" "$LOG_FILE" | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo -e "${RED}⚠️  Errors Found: $ERROR_COUNT${NC}"
    echo "Recent errors:"
    grep -i "error\|exception\|traceback" "$LOG_FILE" | tail -5 | sed 's/^/  /'
else
    echo -e "${GREEN}✅ No errors found${NC}"
fi

echo ""
echo -e "${BLUE}📄 Full log available at: $LOG_FILE${NC}"
echo ""
echo -e "${YELLOW}💡 To view full logs:${NC}"
echo "  cat $LOG_FILE"
echo ""
echo -e "${YELLOW}💡 To search logs:${NC}"
echo "  grep -i 'search_term' $LOG_FILE"
echo ""
echo -e "${YELLOW}💡 To follow logs in real-time (run in another terminal):${NC}"
echo "  tail -f $LOG_FILE"
