# OMERO-OpenHCS web prototype

`omero_openhcs` is the application/deployment boundary for an OMERO.web right
panel that submits OpenHCS work to an execution server. PolyStore owns generic
OMERO storage and ROI mechanics; OpenHCS owns source binding, compilation, and
execution.

## Compatibility status

This package is an alpha prototype, not a supported OpenHCS entry point. Its
Django views and bundled JavaScript still construct the legacy control payload
directly. The current OpenHCS server requires the complete execution signature,
including both global and plate pipeline configuration source. The panel's
bundled pipeline examples also use removed OpenHCS imports.

Do not deploy this package against the current server until an integration test
proves panel submission, status polling, cancellation, and authentication
against the same OpenHCS/ZMQRuntime versions. Use the OpenHCS desktop workflow
for current processing.

## Package boundary

- `omero_openhcs/` owns the Django app, web templates, deployment configuration,
  credentials, and browser-to-server integration.
- PolyStore owns `OMEROLocalBackend`, source references, virtual workspaces, and
  ROI persistence.
- OpenHCS owns pipeline declarations, source bindings, compilation, execution,
  and runtime projection.

The plugin must use OpenHCS's typed execution submission or an explicitly
versioned protocol adapter. It must not duplicate the wire schema or embed a
second catalog of valid pipeline symbols.

See [INSTALL.md](INSTALL.md) for the development-only setup and compatibility
gate.
