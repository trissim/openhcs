"""Top-level OpenHCS command dispatcher.

The default command remains the desktop GUI. Named commands provide stable
package-level entry points for installers such as the MCP Registry without
importing GUI or agent runtimes until they are selected.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
import sys


def _load_main(module_name: str) -> Callable[[], int | None]:
    return import_module(module_name).main


def _run_with_arguments(
    entrypoint: Callable[[], int | None],
    arguments: Sequence[str],
) -> int | None:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *arguments]
        return entrypoint()
    finally:
        sys.argv = original_argv


def main(argv: Sequence[str] | None = None) -> int | None:
    """Run the GUI by default, or the explicitly selected package command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["mcp"]:
        return _run_with_arguments(
            _load_main("openhcs.mcp.bootstrap"),
            arguments[1:],
        )
    if arguments[:1] == ["gui"]:
        arguments = arguments[1:]
    return _run_with_arguments(
        _load_main("openhcs.gui_startup"),
        arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
