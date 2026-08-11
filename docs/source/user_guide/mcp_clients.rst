Using OpenHCS with MCP clients
==============================

OpenHCS exposes a local Model Context Protocol server for supported agent
clients. The local installation contains the processing engine, MCP server, and
desktop UI, while the MCP process itself remains headless and communicates over
stdio.

Local installation
------------------

For normal desktop onboarding, download the native OpenHCS installer for
Windows or macOS and leave **Connect OpenHCS to ChatGPT, Codex, and local AI
agent apps** checked. Setup installs the GUI and MCP runtime together, publishes
one update-stable launcher, and registers that launcher with ChatGPT desktop,
Codex, and other detected supported local clients. Restart the clients after
setup, accept their normal first-use trust prompt if shown, and ask them to use
OpenHCS.

The installer preserves unrelated MCP servers and replaces only the local entry
named ``openhcs``. Re-running Setup updates the private OpenHCS environment
without invalidating the client registration.

Install the combined local environment into an isolated tool environment:

.. code-block:: bash

   pipx install "openhcs[gui,mcp,viz]"

The equivalent persistent ``uv`` installation is:

.. code-block:: bash

   uv tool install "openhcs[gui,mcp,viz]"

The installed MCP commands are ``openhcs mcp`` and ``openhcs-mcp``. Launch the
desktop application separately with ``openhcs``. Starting an MCP client must not
open PyQt windows automatically.

The installed stdio server defaults to the ``desktop`` capability surface. It
keeps normal UI, selected-plate, viewer, plate-data, knowledge,
function-authoring, and pipeline-authoring tools while hiding headless,
runtime-server, fallback-widget, and expert-only tools. The surface is
projected from capability declaration metadata, not a copied tool list.
Advanced users can select another declared surface when configuring the client:

.. code-block:: bash

   openhcs-mcp --surface core       # headless authoring and execution
   openhcs-mcp --surface authoring  # documentation and draft authoring
   openhcs-mcp --surface full       # all local development capabilities

Changing the surface requires restarting the MCP client so it requests the new
tool schemas.

What the agent learns on connection
-----------------------------------

The stdio server publishes first-use instructions in the MCP initialization
handshake; users do not need to paste a separate OpenHCS system prompt into each
client. Those instructions give the agent the compact execution model and tell
it to call ``openhcs_health_check``, then
``openhcs_get_authoring_context`` and ``openhcs_search_capabilities`` before it
chooses an operational route. Capability search filters the selected canonical
registry by workflow, target, role, side effects, or task text and returns a
bounded routing projection. ``openhcs_list_capabilities`` remains available
when the complete selected surface is actually required.

The no-argument authoring-context request defaults to ``kind="first_use"`` and
returns a compact task router. It summarizes the ownership model, tells the
agent how to choose a UI-visible, headless, source-onboarding, authoring, or
viewer-review route, and names the targeted knowledge that can deepen that
route. The request defaults to 16,000 characters. It intentionally does not
embed the complete architecture or every example: request the matching context
kind and then retrieve only its declared source-backed knowledge target.
The bundled Codex plugin reinforces the same handshake through its
``use-openhcs`` skill. Claude Desktop and other MCP clients receive the server
instructions directly from the MCP process.

Before trusting a new client with a real experiment, ask it:

.. code-block:: text

   Check OpenHCS health, read the first-use context, search the current
   capability surface for pipeline authoring and execution, and summarize how
   PipelineDocument, FunctionStep, group_by, variable_components, artifacts,
   source bindings, and UI-visible versus headless execution fit together. Do
   not mutate or execute anything.

A correct response should cite the current tools and capability surface rather
than a remembered tool list. If the health result reports a stale process or
missing packaged resources, restart or repair the MCP installation before
continuing. Installer-managed servers return the update-stable launcher in
``restart_command`` and identify the MCP client as the reconnect owner. Close
the old connection, let the client launch that command, complete a new MCP
initialize handshake, and then retry the blocked operation. A stdio server
cannot replace its own client-owned stream transparently.

The native launch adapter also exposes the installer-owned current-generation
pointer. A running server snapshots that pointer, so an atomic OpenHCS update
is detected even after the old private environment is removed. Health remains
available for diagnosis, while all other capabilities fail closed until the
client reconnects through the stable launcher.

When editing configuration, request ``openhcs_describe_config_schema`` for the
``global``, ``pipeline``, or ``step`` root and follow a returned nested
``path_prefix`` while browsing. The response reports declaring/default
provenance and lazy inheritance as well as the live type and value constraints.
Construct mutation JSON from each field's ``authoring_value_path``; do not pass
the dotted navigation path as a flat key, infer step fields from old examples,
or flatten them into pipeline configuration.

Request the ``ui`` root to inspect process-level desktop settings and their
declaration help. That schema is read-only at the config-draft boundary; change
those fields through the live UI ObjectState so the visible form, history, and
revision token remain authoritative.

Revision protection has two distinct producers. State and code-document reads
return the ``base_revision_token`` used for state mutation. UI action catalogs
return a ``selection_revision_token`` used as
``observed_selection_revision_token`` for the same widget/action/selection.
The generated tool schema names the required producer capability; a state token
is never an action-selection token.

Filesystem access
-----------------

The MCP server accepts local paths only beneath explicitly configured roots:

.. code-block:: text

   OPENHCS_AGENT_READ_ROOTS=/path/to/plates
   OPENHCS_AGENT_WRITE_ROOTS=/path/to/openhcs-outputs

Use the platform path separator when granting multiple roots: ``:`` on Unix
and ``;`` on Windows. Grant the smallest useful directories. A client
installation must not assume access to the home directory or an entire drive.
Rejected paths report the effective readable or writable roots used by the
running server, so an agent can choose a permitted destination without
inspecting OpenHCS source or guessing an unavailable environment variable.

Codex
-----

The repository contains the release plugin under
``packaging/codex/openhcs``. The plugin launches a version-matched PyPI
environment and includes the ``use-openhcs`` workflow skill. During source
development, use the stable-checkout configuration in
:doc:`../development/mcp_development` instead.

Before the plugin is available in a configured marketplace, the supported
Codex CLI fallback is a single local-server registration command:

.. code-block:: bash

   codex mcp add openhcs \
     --env OPENHCS_AGENT_READ_ROOTS=/path/to/plates \
     --env OPENHCS_AGENT_WRITE_ROOTS=/path/to/openhcs-outputs \
     -- uvx --from 'openhcs[gui,mcp,viz]' openhcs-mcp

ChatGPT desktop, the Codex app and CLI, and the Codex IDE extension share the
MCP configuration for the same host. Restart ChatGPT desktop or Codex after
adding or installing the server. In ChatGPT desktop, open **Settings**, then
**MCP servers**, to inspect the connection; after restarting, ``/mcp`` shows
the connected servers.

The native installer performs this local registration automatically. The CLI
command above remains the manual/package-manager fallback and is not required
for installer users.

Claude Desktop
--------------

Claude Desktop releases use the signed ``.mcpb`` artifact generated from
``packaging/mcpb/openhcs``. Its installation form asks separately for a readable
microscopy-data directory and writable output directory. Those choices become
the MCP path policy; the extension does not receive unrestricted filesystem
access.

The native OpenHCS installer can also register its already-installed stable
launcher directly when Claude Desktop is detected, avoiding a second OpenHCS
environment. Restart Claude Desktop after setup. The signed ``.mcpb`` remains
the standalone Claude-directory distribution for users who do not install the
OpenHCS desktop application first.

Other local clients
-------------------

The native installer registers Cursor by preserving its global
``mcpServers`` configuration when Cursor is detected. When the supported Visual
Studio Code command-line interface is available, Setup uses its documented
user-level MCP installation command instead of guessing a private profile path.
Setup also registers the same stable launcher in Gemini CLI's documented user
``settings.json`` and Windsurf Cascade's documented ``mcp_config.json`` when
either client is detected. Each client keeps its own trust and tool-approval
policy, so it may still ask the user to approve OpenHCS the first time the
server starts.

Clients whose public interface cannot safely replace an existing server entry
are not edited heuristically. OpenHCS does not parse human-readable client
output or remove and recreate an entry without a transactional rollback
contract.

GUI attachment
--------------

GUI tools connect to a separately running OpenHCS window through the
authenticated UI bridge. If no bridge is available:

1. Start ``openhcs`` locally.
2. Wait for the main window to finish opening.
3. Ask the agent to discover the bridge again.

Do not paste bridge tokens into prompts or configure a remote client to reach a
local bridge.
