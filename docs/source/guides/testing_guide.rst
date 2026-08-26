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

The source, GUI, OMERO, parity, viewer, and wheel-candidate jobs build the
recursively recorded first-party submodule snapshots and run independently of
dependency publication. The native installer matrix does the same before
running the real Windows and dual-architecture macOS bootstrap installers and
staged updater. A separate release-readiness job requires exact dependency
floors, matching release tags, and PyPI-visible wheels before PyPI-style
installation can run. An unpublished first-party change therefore keeps
release readiness red without hiding current-source or native-installer test
results.

The foundational unit/core job and maintained PyQt GUI job emit fail-closed
coverage artifacts. The dependent Combined Coverage Artifact workflow combines
them with the instrumented Python-boundary, backend/microscope, OMERO, and
installed-wheel jobs after the complete matrix succeeds. Its report therefore
describes the union of those instrumented suites; installer and live-viewer
jobs remain separate behavioral gates.

The Documentation workflow also runs the context-aware ``actionlint`` checker
against every GitHub Actions definition from a checksum-verified release
binary. This catches invalid expressions and contexts that a YAML parser cannot
recognize, even when the workflow containing the defect cannot schedule jobs.
A repository-wide regression separately keeps expression-bearing ``run``
blocks below GitHub's hosted parser ceiling.

The Windows installer smoke also exercises a staged update while both the old
environment entry point and the stable GUI launcher are held without delete
sharing. It then releases the launcher and verifies that its deferred desktop
projection can be refreshed.

The installed-desktop smoke on Windows and both macOS architectures also runs
the packaged restart worker against a real short-lived parent process and
requires its detached restart command to execute from the installed
environment. It constructs the installed OpenHCS application as well, requires
the main window to reach its painted-ready boundary, and drives the live UI
through a packaged desktop MCP session. The probe discovers the main and Plate
Manager windows, invokes the declared code action, verifies its new window, and
then closes the GUI. Both execution and authenticated UI-bridge endpoints are
allocated through the configured transport declaration; cleanup targets only
those endpoints and removes the bridge descriptor.

The Linux wheel job runs this operation before its installed integration suite.
A complementary desktop candidate matrix runs it on Windows and both macOS
architectures using wheels built from the recursively recorded first-party
submodules. These jobs run independently of dependency publication, providing
cross-platform evidence for an unreleased dependency graph before the PyPI
lanes become eligible.

The daily Published Release Canary installs the latest stable desktop package
from PyPI on Linux, Windows, and macOS and repeats that live GUI/MCP probe. It is
the post-release signal for a dependency update that leaves package resolution
valid but breaks application startup or desktop-agent operation.

The native installer jobs build one metadata-discovered wheelhouse from the
recorded first-party submodules, then deliberately provide an unreachable pip
configuration file and unreachable primary and extra index overrides. Windows
installation and reinstall, macOS installation, and the staged-update worker
must resolve the prepared candidates from that wheelhouse without contacting
the injected index. This proves both the unreleased dependency graph and that a
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
configuration before running it. Once an OMERO variant is selected explicitly,
failure to reach the complete gateway-and-table-service readiness contract is a
test failure rather than a skip.

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
materialization. The CI failure step emits OMERO component diagnostics and the
``Tables-0`` log before the container logs so independent table-service startup
is visible. A cold packaged deployment may recover that component only after
PolyStore reports its managed repository; an external deployment remains under
its operator's lifecycle control.
