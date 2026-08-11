MCP distribution architecture
=============================

The MCP distribution has one semantic authority and multiple transport and
packaging projections.

Authority and projections
-------------------------

``AgentCapabilityDeclaration`` and its declaration inheritance graph own tool
identity, input type, service binding, side effects, security requirements,
data exposure, and transport availability. Generic MCP binders iterate the
nominal registry and project those declarations into protocol tools.

Standard MCP annotations are derived from the same declaration metadata. Client
plugins, MCPB manifests, Registry metadata, tests, and documentation must not
carry copied tool-name allowlists or independent security classifications.
Tool results use MCP structured content with a JSON-object output schema. Each
tool's protocol metadata also advertises the nominal ``output_contract`` name
from its capability declaration, so wire tooling does not become a second result
type registry.

The resulting ownership flow is:

.. code-block:: text

   AgentCapabilityDeclaration registry
                 |
          generic MCP binder
                 |
                  local stdio
           /           |           \
   registered clients  Codex plugin  Claude MCPB

The same capability registry can also be filtered into the optional hosted
Streamable HTTP lane. Transport availability remains a declaration fact; the
HTTP server does not maintain a tool-name list.

Local process lane
------------------

The local MCP server runs over stdio and is owned by the client process. It
does not import or host the PyQt UI. GUI, viewer, and local runtime capabilities
remain local-only and attach through their authenticated process bridges.

Installing the GUI with the MCP server is a packaging convenience, not a reason
to merge their process lifetimes. MCP startup must remain usable in a headless
environment, and GUI launch must be an explicit human-approved action.

Local client registration
-------------------------

Native desktop installers publish one platform-stable launcher before they
touch agent-client configuration. ``McpClientRegistrationTarget`` is the nominal
owner for the supported local-client projections: each registered leaf owns its
client's detection, configuration path, and format, while generic orchestration
iterates the root registry. Codex TOML and strict JSON clients are updated
atomically with recoverable backups. The shared registered JSON projection owns
Cursor, Gemini CLI, and Windsurf user configuration; Visual Studio Code is
registered through its documented command-line interface rather than a guessed
profile path.

``DesktopDeploymentAuthority`` owns the native launcher, shortcut, and
application-icon projection. Its registered Windows and macOS leaves read the
installed distribution entry points and packaged brand assets. Both initial
setup and the in-application updater call this authority; the PowerShell and
shell bootstrap adapters do not carry parallel launcher or app-bundle
templates.

The projection owns only the ``openhcs`` server entry and preserves every other
client setting. Its command targets the stable launcher, never a
version-stamped private environment, so a verified installer update can switch
the environment without rewriting client configuration. Registration does not
copy capability lists, path-policy roots, client credentials, or agent
instructions. Those remain owned by the MCP initialisation handshake,
``AgentPathPolicy``, and the client itself.

The launcher projects two lifecycle values into the MCP process: one JSON argv
for reconnecting through that same stable adapter and one installer-owned
generation pointer. Windows uses the atomically replaced launcher file as its
pointer; macOS uses the ``current`` environment symlink. The MCP lifecycle
owner snapshots the pointer and reports source drift and install-generation
drift through the same recovery result. It never attempts to replace an
initialised client-owned stdio stream; the result tells the client to close,
relaunch, reinitialize, and retry.

Local surface profiles
----------------------

Local stdio clients can select ``desktop``, ``core``, ``authoring``, or ``full``.
These are registered nominal profile declarations. Their MRO-composed policies
query capability workflow groups, visibility, roles, and runtime requirements;
they never contain MCP tool-name lists. The installed server defaults to
``desktop`` while the development client explicitly uses ``full`` for complete
surface audits. The desktop profile includes both UI-owned workflows and
independent headless execution. Those routes share pipeline declarations but
retain separate state owners: a headless session does not create Plate Manager
rows or ObjectState history. Capability discovery and bound tools/resources are
filtered by the same ``AgentCapabilitySurfaceSelection`` instance.

Hosted HTTP lane
----------------

``openhcs-mcp-http`` builds a stateless Streamable HTTP server from capabilities
that explicitly include ``CapabilityTransport.HOSTED_STREAMABLE_HTTP``. The
hosted registry must contain only read-only tools; server construction fails if
a future hosted declaration is mutating. UI, local execution, viewer control,
and local filesystem capabilities therefore remain absent unless their owning
declarations are deliberately reclassified.

``McpHttpResourceServerSettings`` owns the deployment policy. It permits either
a public read-only surface or OAuth introspection with a required tenant subject
and scopes, validates secure URLs and origin/host restrictions, and projects the
resulting security scheme into each hosted tool. Invocation audit records carry
the declared capability identity and outcome without recording bearer tokens.

Progressive authoring context
-----------------------------

Authoring guidance is a registry projection, not one static prompt.
``AuthoringContextDeclaration`` owns each requested ``kind``, its task route,
and the knowledge targets that may deepen that route.
``AuthoringContextSection`` owns the sections that compose those declarations.
``AgentAuthoringContextService`` iterates both registries and renders only the
facets belonging to the selected declaration.

The ``first_use`` context is therefore a compact router. It identifies the
core model, asks the client to choose one task route, and points to targeted
source-backed knowledge. It does not attempt to embed the whole architecture or
every example. ``AuthoringContextRequest.max_chars`` defaults to 16,000 and the
service truncates at that explicit request boundary. A client that needs more
context requests the matching declared route or one of its declared knowledge
targets instead of increasing a second hardcoded curriculum.

Configuration-schema projection
-------------------------------

The MCP configuration schema has three roots: ``global``, ``pipeline``, and
``step``. Global and pipeline roots reflect their dataclass declarations. The
step root comes from ``AbstractStep.config_classes_by_field_name()``, which
reflects the keyword-only config declarations on the nominal step constructor;
the agent service does not maintain a copied list of step config fields.

``ConfigFieldSchema`` carries the projected path and type together with enum or
registry values, lazy/inheritable state, the declaring type, default origin,
and any nested schema path. Clients should first request the root family map and
then request a returned ``path_prefix``. These schema records describe the live
declarations; they are not a second configuration model and must not be turned
into a hand-maintained field catalogue.

Long-running local-tool liveness
--------------------------------

Long-running MCP behaviour is declared with the capability that owns the
operation. ``AgentCapabilityDeclaration`` can state a progress interval and
whether the operation is safe to move to a worker thread. The generic FastMCP
binder consumes those fields, emits standard MCP progress when the request
supplies a progress token, and chooses the declared execution context. A
worker-safe operation can emit periodic liveness updates while its task runs.
The binder does not switch on tool names or maintain a second list of slow
operations.

Source-backed orchestrator-session creation remains on the MCP process's main
thread because its import and compiler-facing setup is thread-sensitive. The
binder emits the standard ``started`` progress event before entering that
synchronous section, and the service logs its source-parse start and
completion. The event loop cannot emit periodic progress while the synchronous
main-thread operation is blocked. In verbose mode a separate bounded
``faulthandler`` timer writes a delayed stack diagnostic to stderr without
altering the MCP result. None of those signals is scientific percent
completion or success; the client must still wait for the actual typed result.

Session creation and submitted execution are separate phases. The development
``execute-source`` command first creates the source-backed session, then submits
that exact session. Automation that must remain observable over a long run
submits without an opaque aggregate wait and polls the typed execution-status
capability until a terminal state. The initial progress event and the
start/completion diagnostics expose the pre-submit phase; execution status,
cancellation, results, and viewer settlement remain runtime-service
responsibilities.

Distribution metadata
---------------------

``openhcs.__version__`` is the release-version authority.
``scripts/sync_mcp_release_metadata.py`` projects it into the Codex plugin,
Claude MCPB, and ``server.json``. Its dependency-free phase validates only
directly readable package/version structure. After the built wheel is installed,
``--capability-requirements`` imports the canonical selected capability registry
and verifies that all declaration-owned required extras are projected. These
checked artifacts are install-surface metadata only; they do not own OpenHCS
behaviour.

Canonical MCP knowledge documents remain in the documentation tree. The wheel
build deterministically projects the manifest-declared sources into package
resources, so installed servers do not depend on a source checkout and there is
no manually maintained second document corpus.

Related documentation
---------------------

- :doc:`../user_guide/mcp_clients`
- :doc:`code_ui_interconversion`
- :doc:`external_foundations`
