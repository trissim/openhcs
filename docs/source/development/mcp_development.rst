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
* For active development and testing, prefer ``openhcs-mcp-dev`` or
  ``python -m openhcs.mcp.dev_client``. The dev client starts a fresh stdio MCP
  server from the current checkout for each one-shot command, calls tools through
  the MCP protocol, prints the command's compact renderer (or JSON when
  requested), and closes the subprocess. Its ``shell`` mode instead keeps one
  initialized server for a multi-command development session. It does not depend
  on Codex refreshing its attached MCP process.
  The client incrementally assembles newline-framed JSON responses, so an
  explicitly requested large but bounded projection, such as a complete widget
  tree, is not constrained by Python's stream separator limit.

The OpenHCS MCP server intentionally keeps ``openhcs_health_check`` callable
when the source is stale. Health reports process identity, source freshness, and
the stale source paths. All normal tools return a structured
``mcp_server_stale`` error until the process is restarted.

UI attachment also requires the MCP process and desktop bridge to declare the
same OpenHCS version and bridge protocol. If a UI command reports an endpoint
application or protocol mismatch, restart the MCP process from the same
installed OpenHCS environment as the desktop application. Do not reuse a
descriptor from another checkout or weaken the compatibility check.

For ``selected-workflow --wait``, the development client captures the current
Plate Manager revision before dispatch, waits once for the returned UI
operation receipt, and then polls the Plate Manager state surface. The receipt
establishes that the Qt mutation completed; compile or run completion requires
a later state revision with the requested workflow's terminal row. This keeps a
terminal row from an earlier workflow from satisfying the new wait.

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

Fresh Current-Checkout Client
-----------------------------

Source the active checkout environment, then call the dev client directly:

.. code-block:: bash

   cd /path/to/active-openhcs
   . .venv/bin/activate
   python -m openhcs.mcp.dev_client health
   python -m openhcs.mcp.dev_client tools
   python -m openhcs.mcp.dev_client knowledge
   python -m openhcs.mcp.dev_client authoring-context first_use
   python -m openhcs.mcp.dev_client knowledge-document openhcs_architecture_quick_start --max-chars 4000
   python -m openhcs.mcp.dev_client knowledge-search "viewer"
   python -m openhcs.mcp.dev_client ui-smoke --allow-error-payloads
   python -m openhcs.mcp.dev_client selected-workflow init_plate
   python -m openhcs.mcp.dev_client widget-tree plate_manager
   python -m openhcs.mcp.dev_client viewer-payloads 5565 --include-shape-payloads

The development client uses the ``full`` local surface by default. Pass
``--surface desktop``, ``core``, or ``authoring`` before the command when testing
an installed-client projection. When ``OPENHCS_UI_BRIDGE_DESCRIPTOR`` or
``OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR`` selects a desktop bridge, the development
client forwards that selector to its fresh MCP child. An explicit descriptor
directory remains authoritative and does not fall back to process discovery.

Persistent Current-Checkout Session
-----------------------------------

Use one initialized server when evaluating several commands without paying the
cold-start cost each time:

.. code-block:: bash

   openhcs-mcp-dev shell
   openhcs-mcp-dev> health
   openhcs-mcp-dev> knowledge-search "source bindings"
   openhcs-mcp-dev> ui-status --timeout-ms 2000
   openhcs-mcp-dev> quit

Redirected input is a non-interactive batch, and repeated ``--command`` options
provide the same behavior without a temporary file:

.. code-block:: bash

   openhcs-mcp-dev shell < mcp-smoke-commands.txt
   openhcs-mcp-dev shell --command health --command 'knowledge-search "viewer"'

Blank lines and full-line ``#`` comments are ignored. A fragment such as
``document#section`` remains part of the command. Use ``--stop-on-error`` when a
batch must stop at its first failed command. The persistent process deliberately
retains the stale-source guard: exit and restart the shell after editing watched
OpenHCS source.

Use ``--timeout-seconds`` for the MCP client-side timeout. In shell mode it is
the default for entered commands, while an option on an entered command wins.
UI and viewer tools
also use bounded OpenHCS control timeouts, so a broken bridge or stale viewer
should fail quickly instead of blocking the development loop.

Headless compile and execution submissions apply ``submit_timeout_ms`` as one
budget across execution-server startup, progress registration, task
serialisation, and the control request. Startup progress can refresh the
server's inactivity deadline, but it cannot extend this submission budget. If
the budget expires before the execute request is sent, the tool reports that
known outcome. A timeout after the request is sent remains an unknown outcome;
poll server status before retrying.

For accepted non-blocking jobs, ``openhcs_get_execution_status`` returns the
control-plane lifecycle status together with the submitting client's latest
progress event and monotonic progress sequence. A changing sequence is exact
activity evidence even when the coarse execution status remains ``running``.
Terminal status is cached and releases that client's progress subscription.

The knowledge commands call the same MCP tools exposed to agents:

* ``knowledge`` calls ``openhcs_list_knowledge_documents``.
* ``knowledge-document`` calls ``openhcs_get_knowledge_document`` and accepts an
  optional ``--section-id`` plus ``--max-chars`` bound.
* ``knowledge-search`` calls ``openhcs_search_knowledge``.

The server only serves allowlisted repository documentation through
``openhcs.agent.services.knowledge_base_service``. Clients pass document ids,
not arbitrary filesystem paths.

What Not To Do
--------------

Do not rely on Python module reload for MCP server changes. FastMCP has already
registered tool function objects and the OpenHCS agent context has already been
constructed.

Do not wrap stdio with file watchers that print status to stdout. Stdio stdout
is the MCP JSON-RPC channel; logs and watcher messages belong on stderr.
The installed stdio entrypoint suppresses routine INFO logging so client logs
stay readable; set ``OPENHCS_MCP_VERBOSE=1`` while diagnosing server startup to
restore it.
