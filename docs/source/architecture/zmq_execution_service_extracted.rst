ZMQ execution transition
========================

ZMQRuntime owns generic transport; OpenHCS owns its compiled-execution adapter.
The generic ``EndpointApplication`` declaration owns exact expected-versus-
observed compatibility, and the connection retains the readiness
``PongResponse`` carrying the observed identity. OpenHCS supplies one versioned
application declaration for execution and UI endpoints. The UI observes the
resulting compatibility value and may request a state-preserving endpoint and
desktop restart. No second probe, copied version field, or UI-owned connection
flag participates in the decision.

Each ``EndpointConnectionAttempt`` owns its cancellation token. Cancelling that
attempt raises ``EndpointConnectionCancelledError`` and disconnects a
connection that won a late readiness race. OpenHCS can therefore accept GUI
teardown during endpoint startup without treating it as a server failure,
matching text, or retaining a parallel connection flag.

``TransportEndpoint`` also owns the configured data/control port pair and the
exact subset currently occupied. Proof-gated stale cleanup and forced local
release descend through the selected transport declaration as well. Launchers
and test harnesses use that typed endpoint authority instead of duplicating the
control-port offset, reconstructing a pair, or branching on transport mode.

ZMQRuntime also owns the execution status transition boundary. Terminal states
are immutable, a cancellation request addresses one execution, and queued
cancellation does not interrupt an unrelated running execution. OpenHCS extends
the generic interruption hook with its exact orchestrator and worker ownership;
inline and threaded work then stops at the next cooperative boundary.

OpenHCS execution submission reuses ZMQRuntime's monotonic
``OperationDeadline`` across endpoint startup, progress registration, task
serialisation, and the execute request. ``OpenHCSZMQConfig`` owns this dedicated
submission budget separately from the shorter budget for status, stop, and
other control requests. Startup activity may refresh the endpoint inactivity
deadline, but it cannot extend the caller's total submit budget. OpenHCS reports
separately whether preparation expired before the execute request or the
request was sent without a reply, so callers do not invent an execution
identifier or retry an unknown outcome blindly.

An accepted headless job retains the exact client that submitted it. Status
polling reuses that client, projects its ZMQRuntime-owned latest progress
observation, and disconnects it after caching a terminal response. It does not
recreate clients for polling or mirror transport progress in an OpenHCS-owned
registry.

The launcher's dedicated capability-preparation mode executes
``FunctionCatalogPreparation`` before importing or constructing the execution
server. A cold registry cache therefore has one preparation process and cannot
recursively start execution-server launchers while server modules are still
being imported. Both that process and a newly spawned execution server receive
the environment projected by ArrayBridge's ``MemoryType`` declarations. The
same declaration-owned framework requirements therefore govern parent
admission, catalogue preparation, and server startup; OpenHCS does not rebuild
NVIDIA wheel paths in either launcher.

``RegistryService`` admits an optional backend only after that backend's own
registry declaration proves its runtime warm-up and complete module inventory.
The admitted inventory remains fixed for the interpreter lifetime and is reused
on both sides of persistent-cache preparation. A partially installed optional
runtime therefore stays absent instead of reappearing after a failed import and
invalidating an otherwise usable catalogue.

For native callables, local nominal declarations own any catalogue-module
projection. Only declarations on that module's public surface enter the
browsable catalogue; an explicitly transported private decorated callable can
still be reconstructed from its own contract. Cache identity includes the
current source revision and framework admission context; a failed projection
publishes no partial catalogue, and clearing the service removes every derived
lookup view. The transport service therefore consumes one declaration-derived
catalogue instead of synchronizing a second function registry.

Catalogue control messages preserve the endpoint's complete membership revision
across full and filtered pages. Detail reads and callable-reference reads require
that revision, and a reference contains the registry-owned canonical key and
processing contract. Desktop and local MCP consumers can therefore resolve one
selected callable without importing every catalogue member or reclassifying its
semantics locally.

Batch shutdown is likewise bound to the exact typed endpoint, including its host
and transport. Cleanup cannot reinterpret a non-default TCP endpoint through a
local transport default.

See :doc:`external_foundations`.
