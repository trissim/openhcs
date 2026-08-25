OMERO integration transition
============================

PolyStore owns generic OMERO storage mechanics. OpenHCS packages the deployment
bundle under ``openhcs/omero``; its bundled ``omero_openhcs`` plugin owns the
OMERO.web application integration. PolyStore declarations also own OMERO text
format parsing, table-service readiness, and image-plane batch construction;
OpenHCS does not mirror those mechanics.

The packaged deployment pins PostgreSQL, OMERO.server, and OMERO.web. This makes
a cold deployment resolve the same stack as a cached deployment. Its default
OMERO.web service is the upstream viewer; it does not activate the alpha
``omero_openhcs`` panel before that panel passes its current execution-contract
integration gate.

The same Compose declaration owns the local host, published ports, user, and
password. ``OMEROInstanceManager`` projects those values into typed connection
settings and accepts explicit overrides for remote instances; it does not keep
a second set of local defaults. Docker daemon lifecycle remains an operator or
host-platform responsibility rather than OpenHCS platform-string dispatch.

OMERO server components become available independently. A responsive Blitz
gateway or web application therefore does not by itself establish storage
readiness. The OpenHCS instance lifecycle accepts a connection only after
PolyStore's table-service declaration reports that OMERO.tables is enabled; the
same declaration owns the bounded retry policy used by table creation.

See :doc:`external_foundations` for the package boundary and
:doc:`../guides/omero_integration` for the supported integration path.
