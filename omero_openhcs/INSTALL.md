# OMERO-OpenHCS development setup

The web plugin is currently an alpha prototype and is not compatible with the
current OpenHCS execution submission contract. These steps install it for
development; they do not make it production-ready.

## Dependencies

From the OpenHCS repository, install OMERO client dependencies using the
repository-owned helper or requirements file, then install the Django plugin:

```bash
python scripts/install_omero_deps.py
python -m pip install -e "./omero_openhcs"
```

The plugin package itself declares `omero-web` and `pyzmq`. OpenHCS's `[omero]`
extra declares `omero-py`; ZeroC Ice availability remains platform- and
Python-version-specific. `requirements-omero.txt` and
`scripts/install_omero_deps.py` are the local authorities for that setup.

## Register the development app

```bash
omero config append omero.web.apps '"omero_openhcs"'
omero config append omero.web.ui.right_plugins \
  '["OpenHCS", "omero_openhcs/webclient_plugins/right_plugin.js.html", "openhcs_panel"]'
omero web restart
```

The execution server launcher still exists:

```bash
python -m openhcs.runtime.zmq_execution_server_launcher \
  --port 7777 \
  --persistent
```

Starting both processes is not a compatibility test. The current panel omits
required execution fields and its bundled examples use removed imports.

## Required compatibility gate

Before this package is advertised as usable, an automated integration test must
verify all of the following against the current server:

1. panel and URL registration under OMERO.web;
2. authenticated plate selection and authorization;
3. construction of the current typed OpenHCS execution submission;
4. submission with global and plate pipeline configuration source;
5. status polling and cancellation through the current control protocol;
6. OMERO source projection through the configured PolyStore backend;
7. result materialization and connection cleanup.

Do not copy a wire-field dictionary or pipeline-symbol table into this plugin.
Use the nominal OpenHCS/ZMQRuntime owner or a versioned adapter and keep example
pipeline code sourced from current OpenHCS documentation.

## Troubleshooting the development install

```bash
omero config get omero.web.apps
omero config get omero.web.ui.right_plugins
omero web logs
```

`OPENHCS_EXECUTION_HOST` and `OPENHCS_EXECUTION_PORT` select the server address
used by the current prototype. A reachable socket does not imply protocol
compatibility.
