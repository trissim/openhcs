Testing OMERO integration
=========================

Test the narrowest owning boundary and reserve a live OMERO server for behavior
that cannot be represented by a fake backend.

Unit tests
----------

Use virtual source references, fake metadata, or a test PolyStore backend for
source-binding and compiler tests. Assert typed source/artifact plans rather
than matching ``/omero/...`` strings in generic code.

Test generic persistence in PolyStore at its declaration owners: text members
must own extension, MIME type, and parser behaviour; table tests must drive the
reported service-readiness states; and image batches must prove plane shape,
dtype, padding, and Z/C/T order without a live server.

Address tests must exercise ``OMEROWellAddress`` and ``OMEROPlaneAddress``
directly, including multi-letter rows, Windows-style virtual paths, result
suffixes, sparse site identities, and rejection of string-keyed coordinate
bags. ``OMEROPlaneFilenameTemplate`` tests additionally prove that symbolic
pattern fields round-trip without weakening concrete plane-address validation.
OpenHCS tests only the nominal projection from PolyStore components into
``FilenameParseResult``.

Integration tests
-----------------

The repository integration harness can start or connect to the configured
OMERO stack, upload a synthetic plate, register ``OMEROLocalBackend``, and run
the normal compile-before-execute flow. Live tests require the OMERO Python
client and deployment dependencies and should be selected explicitly. Once
selected, an unavailable stack is a test failure rather than a skip. Readiness
requires both a gateway connection and PolyStore's declared table service; a
responsive Blitz endpoint alone is not sufficient.
Start Docker before selecting the live variant. The instance manager may start
the packaged Compose services and waits boundedly for an already-starting
daemon, but it does not start Docker Desktop or a host service manager.

Keep credentials in the test environment, never in pipeline declarations or
fixtures committed to the repository. Always close gateway connections and
stop only server instances owned by the test harness.

Include a native plate without OpenHCS parser annotations and a derived plate
whose persisted image name carries a non-contiguous site identity. Listing and
loading must preserve that site rather than replacing it with WellSample order.

Exercise both direct and ZMQ execution against the same live plate. Verify the
created images and tables, consolidated summaries, connection cleanup, and a
cold pull and start of the pinned OMERO.web image. Do not open a browser from
the test harness; the URL is evidence for a human caller, not an integration-
test side effect. On CI failure, preserve the OMERO component diagnostics and
the table service log so an independently starting ``Tables-0`` process can be
distinguished from a generic container failure.

The CI OMERO lane builds the recursively recorded first-party submodule wheels
and runs independently of dependency publication readiness. It therefore tests
the current storage and transport snapshots before their public package
releases, while the separate installer-facing lane proves the published
dependency set.

Deployment tests for packaging, credentials, and server lifecycle belong in
OpenHCS; generic storage behavior belongs in PolyStore. Panel behavior belongs
in ``omero_openhcs``. The panel is not enabled by the default deployment and is
not compatible with the current execution submission contract until its
documented panel-to-server integration gate passes. See
:doc:`../guides/omero_integration`.
