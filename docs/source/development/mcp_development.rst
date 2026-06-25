MCP Development Workflow
========================

OpenHCS exposes agent functionality through an MCP stdio server. Stdio servers
are launched and owned by the MCP client, so editing the running server's Python
source does not hot-reload that process. A running OpenHCS MCP process therefore
must fail fast when its loaded source becomes stale, while developers use an
explicit restart workflow for active MCP changes.

Recommended Modes
-----------------

Use two separate modes during development:

* Use a stable installed OpenHCS MCP server as the Codex control plane. This
  can be an editable install from a stable worktree, but it should not be the
  checkout being actively edited. Point it at the working checkout through
  ``OPENHCS_AGENT_READ_ROOTS`` and ``OPENHCS_AGENT_WRITE_ROOTS``.
* Use a separate active-checkout MCP process when editing ``openhcs/mcp`` or
  ``openhcs/agent``. Restart that MCP client/process after source edits before
  relying on non-health tools.

The OpenHCS MCP server intentionally keeps ``openhcs_health_check`` callable
when the source is stale. Health reports process identity, source freshness, and
the stale source paths. All normal tools return a structured
``mcp_server_stale`` error until the process is restarted.

Codex Config Shape
------------------

For local Codex stdio use, prefer an ``exec`` command and short control
timeouts:

.. code-block:: toml

   [mcp_servers.openhcs-dev]
   command = "bash"
   args = ["-lc", "cd /path/to/stable-openhcs && exec .venv/bin/python -m openhcs.mcp"]
   startup_timeout_sec = 5
   tool_timeout_sec = 10

   [mcp_servers.openhcs-dev.env]
   OPENHCS_AGENT_READ_ROOTS = "/path/to/active-openhcs:/tmp"
   OPENHCS_AGENT_WRITE_ROOTS = "/path/to/active-openhcs/mcp_outputs:/tmp"
   XDG_RUNTIME_DIR = "/run/user/1000"

The ``exec`` form makes the Python MCP process the child process observed by
the client instead of leaving an extra shell wrapper. Short timeouts prevent a
broken UI/viewer/runtime control path from becoming a long Codex wait.

What Not To Do
--------------

Do not rely on Python module reload for MCP server changes. FastMCP has already
registered tool function objects and the OpenHCS agent context has already been
constructed.

Do not wrap stdio with file watchers that print status to stdout. Stdio stdout
is the MCP JSON-RPC channel; logs and watcher messages belong on stderr.

Do not make Streamable HTTP the default local-development answer until the repo
has explicit loopback, auth, path-policy, and audit behavior for that transport.
