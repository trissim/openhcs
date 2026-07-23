"""Agent-service stdio guards for MCP-safe tool execution."""

from __future__ import annotations

import contextlib
import os
import sys
import threading


class AgentStdoutRedirect:
    """Redirect noisy library stdout away from MCP stdio transport."""

    _lock = threading.RLock()

    @staticmethod
    @contextlib.contextmanager
    def to_stderr():
        with AgentStdoutRedirect._lock:
            try:
                stdout_fd = 1
                stderr_fd = 2
                os.fstat(stdout_fd)
                os.fstat(stderr_fd)
            except (AttributeError, OSError):
                with contextlib.redirect_stdout(sys.stderr):
                    yield
                return

            saved_stdout_fd = os.dup(stdout_fd)
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stderr_fd, stdout_fd)
                with contextlib.redirect_stdout(sys.stderr):
                    yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout_fd, stdout_fd)
                os.close(saved_stdout_fd)
