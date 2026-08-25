.. _testing_guide:

Testing guide
=============

OpenHCS separates fast unit tests, CPU integration matrices, optional viewer
tests, and live OMERO integration. The command-line options are registered in
``tests/pytest_integration_options.py`` and composed into the integration matrix
by ``tests/integration/conftest.py``.

Default local discovery
-----------------------

``pytest.ini`` discovers the following integration variants by default:

- backends: ``disk`` and ``zarr``;
- microscopes: ``ImageXpress``, ``OperaPhenix``, and ``OpenHCS``;
- dimensions: ``3d``;
- execution: ``multiprocessing``;
- transport: ``direct`` and ``zmq``;
- viewers: ``none``, ``napari``, ``fiji``, and ``napari+fiji``.

VS Code adds ``OMERO`` to the microscope discovery list. Collection does not
install optional dependencies or guarantee that a live service is available.

Focused commands
----------------

Run unit tests in CPU-only mode:

.. code-block:: console

   OPENHCS_CPU_ONLY=1 python -m pytest tests/unit -q

Run one integration boundary explicitly:

.. code-block:: console

   OPENHCS_CPU_ONLY=1 python -m pytest tests/integration \
     --it-backends disk \
     --it-microscopes ImageXpress \
     --it-dims 3d \
     --it-exec-mode multiprocessing \
     --it-zmq-mode zmq \
     --it-visualizers none

Use comma-separated values or ``all`` only when the option implementation
supports them. Viewer variants require the corresponding ``napari``/``fiji``
dependencies and may open detached processes; select them deliberately.

CI coverage
-----------

``.github/workflows/integration-tests.yml`` is the authority for the current
matrix. It covers Python and operating-system boundaries, disk/zarr with
ImageXpress and OperaPhenix, submodule and published-dependency installation,
and wheel integration. A dedicated source job runs ``tests/pyqt_gui`` with
offscreen Qt against the exact pinned pyqt-reactive wheel. Dedicated Linux jobs
run OMERO on supported Python versions with an explicit ZeroC Ice wheel.

The foundational unit/core job and maintained PyQt GUI job emit fail-closed
coverage artifacts. The dependent Combined Coverage Artifact workflow combines
them with the instrumented Python-boundary, backend/microscope, OMERO, and
installed-wheel jobs after the complete matrix succeeds. Its report therefore
describes the union of those instrumented suites; installer and live-viewer
jobs remain separate behavioral gates.

The Windows installer smoke also exercises a staged update while both the old
environment entry point and the stable GUI launcher are held without delete
sharing. It then releases the launcher and verifies that its deferred desktop
projection can be refreshed.

The installed-desktop smoke on Windows and both macOS architectures also runs
the packaged restart worker against a real short-lived parent process and
requires its detached restart command to execute from the installed
environment.

The native installer jobs deliberately provide an unreachable pip
configuration file and unreachable primary and extra index overrides. Windows
installation and reinstall, macOS installation, and the staged-update worker
must complete without contacting that injected index. This proves that a
workstation's package-index settings cannot redirect a managed desktop
installation.

On ``main``, the Python quality job checks the cumulative change set since the
most recent successful Integration Tests head. A failed Python change remains
in scope on later pushes until it is corrected and the complete workflow
succeeds. Pull requests use their base commit as the corresponding boundary.

The standard execution matrix uses ``--it-visualizers none``. A separate Xvfb
job initializes and closes Fiji through PolyStore's declared managed ImageJ
runtime. It opens a generated ImageXpress plate through the real Bio-Formats
reader in three fresh processes, then runs one manifest-owned Official30 case
through Fiji-only and Fiji+Napari viewer paths. The bounded reader/viewer smoke
complements the wider headless matrix; it does not make every backend and
microscope combination a live-viewer test.

OMERO
-----

OMERO tests require Docker, the OMERO client stack, and supported ZeroC Ice
wheels. The integration helper may start the configured local OMERO stack, but
developers should treat that as an external-state mutation and inspect the test
configuration before running it.

.. code-block:: console

   python -m pytest tests/integration \
     --it-backends disk \
     --it-microscopes OMERO \
     --it-dims 3d \
     --it-exec-mode multiprocessing \
     --it-visualizers none

Keep credentials in the environment and close connections owned by a test. See
:doc:`../development/omero_testing` for the owner-boundary strategy.

Failure diagnosis
-----------------

Start with the smallest failing variant and add ``-v --tb=short -s``. For ZMQ
failures, begin with the client, server, or viewer log path reported by the
failing launch. With the default ``LoggingConfig`` and no explicit override,
GUI logs are written beneath ``get_openhcs_data_dir() / "logs"``: this is
``$XDG_DATA_HOME/openhcs/logs`` when ``XDG_DATA_HOME`` is set and otherwise
``~/.local/share/openhcs/logs``. For viewer failures, first prove the viewer can
start independently. For OMERO failures, distinguish dependency installation,
server readiness, authentication, source projection, and result
materialization.
