"""Headless agent-facing API for OpenHCS.

This package owns the stable projection used by MCP, future CLIs, and automated
review agents. It intentionally avoids PyQt imports.
"""

from openhcs.agent.dto.common import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
