# OpenHCS OMERO deployment bundle

This directory contains the single packaged Compose declaration used by
`OMEROInstanceManager` and the live OMERO integration suite. It starts pinned
PostgreSQL, OMERO.server, and OMERO.web releases. OpenHCS accepts a connection
only after PolyStore reports that the OMERO table service is ready.
The Compose connection extension owns the local ports and credentials consumed
by `OMEROInstanceManager`. Docker itself must already be running.

Start or connect through the runtime owner:

```python
from openhcs.runtime.omero_instance_manager import OMEROInstanceManager

with OMEROInstanceManager() as manager:
    connection = manager.conn
```

For supported workflows and tests, see the
[OMERO integration guide](../../docs/source/guides/omero_integration.rst) and
[OMERO testing guide](../../docs/source/development/omero_testing.rst).

## OMERO.web plugin status

`plugin/` owns an alpha OMERO.web panel prototype. The default Compose stack
does not install or expose it because its submission path has not passed the
current OpenHCS execution-contract integration gate. Its development setup and
required gate are documented in [plugin/INSTALL.md](plugin/INSTALL.md).
