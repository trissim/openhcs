Streaming boundary and viewer integration
=========================================

OpenHCS treats viewer streaming as an integration surface. It owns microscopy
semantics and application policy, but it does not duplicate generic storage or
transport machinery.

Ownership
---------

``zmqruntime``
  Owns ZMQ endpoints, control/data request contracts, ping/pong readiness,
  acknowledgments, generic viewer process registration, and lifecycle state.

``polystore``
  Owns streaming backends and the reusable wire-payload builders for source,
  producer, display, and component metadata.

``openhcs``
  Owns registered viewer declarations, compilation into step plans, exact
  microscopy source and producer semantics, platform launch policy, and the
  Napari/Fiji adapters that project received batches into application state.

See :doc:`external_foundations` for the extracted-package boundary. This page
documents the OpenHCS integration and cross-package invariants; it does not
reproduce the internal class architecture of either dependency.

Compiled streaming path
-----------------------

Viewer streaming is a compiled output path, not a directory scan performed by
the viewer.

1. A concrete ``StreamingConfig`` declaration owns the viewer identity,
   endpoint, display policy, and visualizer factory. The compiler iterates the
   root registry and records enabled instances on ``CompiledStepPlan``.
2. Function execution contextualizes each returned value according to its exact
   compiled artifact plan. Main-flow images are recorded in the execution-local
   ``StepOutputManifestStore``; non-main-flow artifacts remain typed runtime
   values selected by their compiled artifact edges.
3. Output finalization streams actual main-flow records from that manifest.
   Artifact outputs use their ``ArtifactMaterializationTargetPlan`` instead.
   A stream-only target can therefore publish an object-label artifact without
   inventing a persistent file or pretending that it is an image output.
4. ``StreamComponentMessageExtraAuthority`` combines the configured viewer
   surface with producer identity, source metadata, and display-axis semantics.
   PolyStore serializes and dispatches that request through its registered
   streaming backend.
5. The OpenHCS viewer adapter receives the typed payload, while generic ZMQ
   request/reply and acknowledgment behavior remains in ZMQRuntime. Napari then
   routes by declared producer and component domains and applies layer updates on
   the Qt thread.

This path has two independent identities:

producer identity
  Identifies the compiled step and exact output surface that emitted the value.
  ``FunctionStepOutputProducerIdentityAuthority`` derives it from the compiled
  plan and the returned-output or artifact context.

source identity
  Identifies the source image plane or source domain represented by the value.
  It comes from runtime payload provenance and compiled source relations.

One must not be reconstructed from the other. A produced image can retain a
source plane while having a new producer, and multiple output occurrences can
share a display name without sharing producer identity. The output manifest is
lineage for values produced by this execution; stale files that merely match an
output directory pattern are not candidates.

At the receiver boundary, PolyStore owns projection mechanics through
``StreamProducerIdentity.projection_key``. Item-aligned producer identities and
declared component projection modes determine the receiver route; duplicate
projected coordinates are rejected rather than overwritten. OpenHCS contributes
the compiled producer/source/component facts to that owner contract and must not
recreate its projection key with filenames, titles, or local tuple conventions.

Object labels and image context
-------------------------------

Object-label values remain object-label artifacts through recording,
materialization, and viewer dispatch. The artifact-type materialization strategy
owns their automatic ROI/shape representation. Core streaming code does not
branch on a CellProfiler module name or infer labels from an ndarray dtype.

When a callable deliberately renders an image from labels, the compiled
``source_context_source`` relation selects the label artifact as the output's
context source. Runtime contextualization then uses the registered
``ObjectLabelImageOutputSourceContextStrategy``. For an unselected volumetric
invocation, including a depth-one volume, that strategy requires the declared
plane axis, requires source-plane provenance cardinality to match it, and
validates the rendered output shape. A true scalar 2D label image remains
scalar. Array rank alone never creates source identity.

See :doc:`runtime_value_system` for the general output-context and projection
rules and :doc:`artifact_contract_system` for compiled artifact relations.

Viewer startup and readiness
----------------------------

``ManagedViewerLifecycleMixin`` is OpenHCS's application lifecycle boundary;
ZMQRuntime's ``ViewerStateManager`` remains the process-wide instance registry.
An existing endpoint is reusable only after a typed control reply proves that a
compatible viewer is ready. Endpoint files or bound ports are discovery facts,
not readiness evidence.

Napari owns its widgets and layer mutation on the Qt main thread. Its detached
entry point constructs the viewer, enters the Qt event loop, and uses an initial
Qt callback to install the recurring message timer before binding the ZMQ
endpoints. Thus an advertised endpoint can service control traffic; socket
publication cannot race ahead of a live event loop. ZMQRuntime then proves
readiness with the strict PING/PONG contract described in the
`ZMQRuntime viewer-streaming documentation
<https://github.com/OpenHCSDev/ZMQRuntime/blob/v0.2.8/docs/source/architecture/viewer_streaming_architecture.rst>`_.

Detached Qt environment policy is also application-owned. The singular
``ViewerProcessPlatform`` declaration selects the platform behavior before the
child starts. On Linux it selects XCB when the caller has not selected another
Qt platform, disables X11 shared-memory transport for the detached process, and
disables Mesa's blocking vertical-blank wait. The latter is a liveness
constraint: a renderer blocked on vblank can starve the same Qt loop that must
answer readiness and data messages. These settings belong to the detached
viewer launch policy, not to global application initialization or a benchmark
workaround.

Settlement and transient viewer evidence
----------------------------------------

Transport acceptance is not proof that a deferred Qt layer update succeeded.
Napari records an exception from a scheduled route update against that exact
route while keeping the event loop alive. The ``settle`` control action starts
or observes an incremental drain of the remaining debounced updates. Each
control reply carries a typed ``ViewerSettleProgress`` record with a phase,
completed and total update counts, and the currently active route. Napari
schedules only one route per Qt callback so the event loop remains available to
answer control traffic while a large viewer state is settling.

The caller polls that progress until it reaches ``complete`` or ``failed``. Its
timeout is a **no-progress deadline**, not a cap on total settlement time:
advancing the completed count renews the deadline. A large legitimate transfer
can therefore take longer than the configured interval while still failing a
viewer that is genuinely stalled. Fiji uses the same wire contract and reports
terminal progress for its synchronous update path. Every recorded route must
succeed before settlement completes, so an exception crosses the asynchronous
Qt boundary instead of becoming a silent log message or an unrelated timeout.

Compiled plate execution follows this order on success:

1. settle every execution viewer;
2. read typed state from non-persistent viewers while their endpoints are live;
3. attach the state evidence to the execution result;
4. stop non-persistent viewers and release their owned endpoints.

Functional streaming tests can consequently assert actual image and label/shape
layers, route identity, component domains, and nonzero payload summaries even
when the viewer is intentionally closed at the end of the run. Persistent
viewers remain governed by their configured lifecycle policy.

Failure boundaries
------------------

- Compilation fails when a streaming declaration or required source metadata is
  invalid.
- Output contextualization fails when a declared source relation, plane axis,
  provenance count, or value shape is inconsistent.
- Readiness fails unless the control endpoint returns the typed PONG required by
  policy within one deadline.
- Stream settlement fails on any deferred layer-route exception.
- Stream settlement also fails when the typed completed-update count makes no
  forward progress for the configured deadline; total elapsed settlement time
  is not itself a failure.
- Non-persistent execution fails cleanup if its viewer remains active; runtime
  acceptance also verifies that its owned endpoint was released.

Do not repair these boundaries with port-existence readiness, copied viewer-name
tables, filename-derived producer identity, shape-based source inference,
transport threads that mutate Qt state, or retries that hide a blocked event
loop.
