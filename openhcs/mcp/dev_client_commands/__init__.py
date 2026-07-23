"""Domain command declarations for the MCP dev client."""

from openhcs.mcp.dev_client_commands import knowledge_pipeline as knowledge_pipeline
from openhcs.mcp.dev_client_commands import plate as plate
from openhcs.mcp.dev_client_commands import ui as ui
from openhcs.mcp.dev_client_commands import viewer as viewer

__all__ = (
    "knowledge_pipeline",
    "plate",
    "ui",
    "viewer",
)
