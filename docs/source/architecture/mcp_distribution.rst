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
            /           \
      local stdio     hosted HTTP
        /    \             |
     Codex   MCPB     remote connectors

Local process lane
------------------

The local MCP server runs over stdio and is owned by the client process. It
does not import or host the PyQt UI. GUI, viewer, and local runtime capabilities
remain local-only and attach through their authenticated process bridges.

Installing the GUI with the MCP server is a packaging convenience, not a reason
to merge their process lifetimes. MCP startup must remain usable in a headless
environment, and GUI launch must be an explicit human-approved action.

Local surface profiles
----------------------

Local stdio clients can select ``desktop``, ``core``, ``authoring``, or ``full``.
These are registered nominal profile declarations. Their MRO-composed policies
query capability workflow groups, visibility, roles, and runtime requirements;
they never contain MCP tool-name lists. The installed server defaults to
``desktop`` while the development client explicitly uses ``full`` for complete
surface audits. Capability discovery and bound tools/resources are filtered by
the same ``AgentCapabilitySurfaceSelection`` instance.

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
into a hand-maintained field catalog.

Hosted lane
-----------

The hosted transport is a distinct resource-server boundary. Only declarations
whose owning capability marks them as remotely available may be bound. The
hosted service requires OAuth validation, tenant-isolated path policy and
credentials, bounded requests, and durable job state. It must never expose the
local UI bridge, viewer processes, or arbitrary host paths.

``HostedTransportCapabilityMixin`` is the opt-in boundary. The generic binder
queries the resolved declaration metadata for the requested transport; it does
not carry a tool-name list. The initial hosted surface is intentionally
read-only and limited to packaged knowledge, architecture projection, function
discovery, configuration-schema reflection, and its filtered capability
registry.

``openhcs.mcp.http`` projects one hosted resource-server configuration into
FastMCP. It uses stateless Streamable HTTP, OAuth token introspection, exact
issuer/subject/audience/scope/expiry checks, DNS-rebinding protection, explicit
Host and Origin policy, and mandatory tenant path roots. TLS, denial-of-service
limits, secret storage, and durable audit retention belong to the surrounding
deployment boundary.

The generic tool guard reports invocation outcomes to an optional transport
observer. Hosted HTTP projects those declaration identities into token-free
structured audit events; it does not inspect names or record tool arguments.

Distribution metadata
---------------------

``openhcs.__version__`` is the release-version authority.
``scripts/sync_mcp_release_metadata.py`` projects it into the Codex plugin,
Claude MCPB, and ``server.json``. These checked artifacts are install-surface
metadata only; they do not own OpenHCS behavior.

The browser plugin is generated only after a hosted endpoint exists.
``scripts/build_hosted_mcp_plugin.py`` projects the synchronized product
metadata and remote URL into a temporary release artifact; protocol capability
availability still comes from the declaration registry at server startup.

Canonical MCP knowledge documents remain in the documentation tree. The wheel
build deterministically projects the manifest-declared sources into package
resources, so installed servers do not depend on a source checkout and there is
no manually maintained second document corpus.
