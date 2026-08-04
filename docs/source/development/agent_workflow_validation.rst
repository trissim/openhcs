Agent workflow validation
=========================

OpenHCS client registration and autonomous workflow performance are different
claims.  A registered client target proves that the installer can project the
local MCP launcher into that client's configuration.  It does not prove that
every model available through that client can construct and verify an analysis.

This protocol measures the latter claim with one fixed task.  Results are keyed
by the exact client, model, version, OpenHCS build, and date.  They must not be
generalized to other models or registered clients.

Authorities and scope
---------------------

``McpClientRegistrationTarget.__registry__`` owns the local registration
targets.  Validation tools and result checks may resolve a reported
``registration_target_id`` through that registry, but must not copy its target
identities into a second list or schema enum.

The MCP capability declarations own the available operations.  Pipeline,
compiled-plan, runtime, artifact, UI, and viewer projections remain the
scientific evidence authorities.  A client transcript is evidence of what the
model attempted; it is not a substitute for those typed OpenHCS results.

Fixed fixture
-------------

Create one canonical plate through the production synthetic-plate service.  Use
the deterministic request exercised by the installed MCP demo:

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Format and well
     - ImageXpress, ``A01``
   * - Grid and tile
     - one by one, 192 by 192 pixels
   * - Overlap and stage error
     - 10 percent, one pixel
   * - Acquired components
     - two wavelengths, one Z plane
   * - Generated biology
     - 12 cells, 0.95 shared-cell fraction
   * - Reproducibility
     - random seed 1, all source components included

Retain the complete generation request and a sorted, relative-path per-file
SHA-256 manifest.  Copy that canonical plate byte-for-byte into each client's
separate run directory.  Before starting the client, verify both the copied
hashes and exact relative file set; a hash manifest alone does not reject an
unexpected file left by an earlier run.  Do not regenerate the fixture
independently for each client: generated TIFF images are seed-stable, but the
ImageXpress HTD metadata includes a time-derived unique plate identifier.
Separate plate/output directories prevent stale outputs or viewer state from
crossing run boundaries.

Fixed prompt
------------

Replace only ``{plate_path}``, ``{output_root}``, and ``{viewer_port}`` with
run-specific values:

.. code-block:: text

   The folder {plate_path} is a synthetic two-channel ImageXpress assay.
   Channel 1 is a nuclear stain and channel 2 is a secondary cellular marker.

   Using only the OpenHCS MCP and the connected OpenHCS desktop, build a compact
   pipeline that segments and counts channel-1 nuclei, quantifies marker-positive
   cells in channel 2, saves reviewable images, ROIs, and measurements under
   {output_root}, and opens the results in Napari on port {viewer_port}. Start by
   inspecting the folder and confirming its axes.
   Use registered OpenHCS functions or presets, not repository/source inspection
   or a new custom function. Leave the finished workflow visible in the desktop
   UI. Keep the ZMQ Server Manager and its granular execution progress visible
   while the pipeline runs. Validate and compile before running, verify the
   structured outputs and settled Napari viewer state, and return the editable
   generated Python plus a concise evidence summary. Distinguish the interactive
   in-memory Live Results projection from its typed materialized disk locations,
   and do not add a duplicate exporter when the declared outputs are already
   persistent. If a validation or execution step fails, diagnose and repair it
   through MCP.

   This prompt authorizes writes under the stated output directory, changes to
   this isolated OpenHCS desktop session, pipeline execution, and launching
   Napari. Do not use shell commands or inspect repository files.

Isolation and intervention accounting
-------------------------------------

Start the client in an empty run directory, with the OpenHCS source checkout
outside its granted workspace.  Pin and record the full client version, model
identifier, OpenHCS version, MCP transport and surface profile, and start/end
times.  Record the exact client approval/sandbox policy and invocation flags;
pre-authorized bypass mode must be visible in the published evidence rather
than presented as the client's default behavior.  Serialize runs on the host,
prove the recorded viewer port is free before launch, and close the isolated
desktop, runtime, and viewer before the next client starts.

Disable repository, shell, browser, and filesystem tools when the client
supports doing so.  Otherwise retain its structured event trace and reject the
run if a non-MCP tool inspects or mutates the workflow.  The fixed prompt is the
initial task authorization.  Record a standard client safety-confirmation click
as an approval event, not as scientific guidance.  Count every later user
message, correction, manual UI action, or externally supplied hint as a domain
intervention.  Count MCP failures and model-initiated repair attempts
separately.

Acceptance gates
----------------

A run passes only when all of these checks have direct evidence:

1. Typed plate inspection identifies ImageXpress, ``A01``, one site, channels
   1 and 2, one Z plane, and one timepoint.
2. The client returns a complete editable pipeline document, and OpenHCS
   validation and compilation succeed before execution.
3. The artifact plan and terminal execution result establish that reviewable
   images, ROI data, and measurements were produced under the run output root,
   without a redundant export step when the callable already declares persistent
   typed outputs.
4. Napari reaches a settled state with nonzero mounted payloads and no missing
   or duplicate coordinates.
5. The trace contains no repository/source inspection or non-MCP workflow
   tools.
6. No domain intervention occurs after the fixed prompt.

A technically passing result is additionally labeled ``unattended`` when no
approval or domain intervention occurred, ``approval-gated`` when the only
human actions were standard client safety confirmations, or ``domain-assisted``
when the user supplied further task guidance.  A capability or environment
unavailable before the task starts is ``blocked`` rather than failed.  Any
other incomplete run is ``failed``.  Only the first label supports an
``unattended`` claim; both of the first two support a claim that no person
manually constructed the pipeline.

Machine-readable evidence
-------------------------

Store each run as one JSON object with ``schema_version`` equal to
``openhcs.agent-workflow-validation.v1``.  The object has these required
groups:

``run``
   Immutable run identity, UTC timestamps, wall time, technical verdict
   (``passed``, ``failed``, or ``blocked``), operation mode (``unattended``,
   ``approval-gated``, or ``domain-assisted``), and any failure or blocker.

``client``
   Client name/version, exact model and provider, nullable
   ``registration_target_id``, host platform, and authentication mode without
   credentials, plus the exact approval/sandbox policy and invocation flags.

``openhcs``
   Version and install/source identity, MCP transport and profile, exposed tool
   and resource counts, and server health.

``fixture``
   Complete generation request, plate/output roots, and sorted file-hash
   manifest.

``protocol``
   Prompt and hash, granted roots, authorized side effects, prohibited tool
   classes, and acceptance-gate version.

``trace``
   Paths and hashes for the complete client event log and final response;
   ordered MCP calls and result statuses; non-MCP calls; model repair attempts;
   human interventions; and approval events.

``evidence``
   Typed plate inspection, pipeline validation, compilation/artifact plan,
   execution terminal state, output inventory, viewer settlement, and editable
   pipeline-source path and hash.

``acceptance``
   One boolean and evidence reference for each numbered gate.

When ``registration_target_id`` is not null, validate it by querying
``McpClientRegistrationTarget.__registry__``.  The empirical result format must
remain open to unregistered development clients and must never become another
client registry.

Reporting results
-----------------

Publish the complete record and uncut event/video evidence together.  Edited
media may point to the uncut recording but must not imply actions absent from
it.  A compatibility logo should continue to mean installer registration; only
an exact client/model/version row may claim a validated workflow result.

The first retained baseline record and its linked transcript, event stream,
generated pipeline, edited overview, and uncut video live under
``website/assets/agent/``.  Later client/model rows must run this same protocol
rather than being inferred from that baseline.
