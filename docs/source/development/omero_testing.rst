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

Integration tests
-----------------

The repository integration harness can start or connect to the configured
OMERO stack, upload a synthetic plate, register ``OMEROLocalBackend``, and run
the normal compile-before-execute flow. Live tests require the OMERO Python
client and deployment dependencies and should be selected explicitly.

Keep credentials in the test environment, never in pipeline declarations or
fixtures committed to the repository. Always close gateway connections and
stop only server instances owned by the test harness.

Exercise both direct and ZMQ execution against the same live plate. Verify the
created images and tables, consolidated summaries, connection cleanup, and a
cold build of the pinned OMERO.web image. Do not open a browser from the test
harness; the URL is evidence for a human caller, not an integration-test side
effect.

Deployment tests for packaging, credentials, and server lifecycle belong in
``omero_openhcs``; generic storage behavior belongs in PolyStore. The web
package is not compatible with the current execution submission contract until
its documented panel-to-server integration gate passes. See
:doc:`../guides/omero_integration`.
