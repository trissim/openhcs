Test a branch in a separate Git worktree
========================================

Use a Git worktree when you need to test one branch without changing the
checkout used for another task. Give each OpenHCS worktree its own virtual
environment and submodule checkouts so imports and first-party dependency
revisions cannot leak between branches.

Create the worktree
-------------------

From an existing OpenHCS checkout, fetch the branch and add a sibling worktree:

.. code-block:: console

   git fetch origin
   git worktree add ../openhcs-feature feature/my-feature-branch
   git worktree list

If the branch does not yet exist, create it from the intended base explicitly:

.. code-block:: console

   git worktree add -b feature/my-feature-branch ../openhcs-feature origin/main

Initialise the worktree
-----------------------

Each worktree has its own checked-out files and submodule working trees. Set up
the feature worktree from inside that directory:

.. code-block:: console

   cd ../openhcs-feature
   git submodule sync --recursive
   git submodule update --init --recursive
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

Install the first-party submodules in the order documented in
:doc:`repository_setup`, then install OpenHCS:

.. code-block:: console

   python -m pip install -e external/ObjectState
   python -m pip install -e external/python-introspect
   python -m pip install -e external/metaclass-registry
   python -m pip install -e external/arraybridge
   python -m pip install -e external/pycodify
   python -m pip install -e external/PolyStore
   python -m pip install -e external/pyqt-reactive
   python -m pip install -e external/zmqruntime
   python -m pip install -e ".[dev,gui]"

Do not share an editable virtual environment between worktrees. The most recent
editable install would decide which checkout supplies ``openhcs`` and its
first-party packages, making a branch comparison ambiguous.

Verify test provenance
----------------------

Before interpreting a result, confirm that Python imports the intended
worktree and that its recursive submodules match the recorded revisions:

.. code-block:: console

   python -c "import openhcs; print(openhcs.__file__)"
   git status --short --branch
   git submodule status --recursive

The printed module path must be inside ``openhcs-feature``. A leading ``+`` in
``git submodule status`` means a submodule is checked out at a different commit
from the superproject declaration; resolve that mismatch before testing.

Run focused tests
-----------------

Run the smallest suite that exercises the change first:

.. code-block:: console

   OPENHCS_CPU_ONLY=1 python -m pytest tests/unit/path/to/test_owner.py -q

When the change crosses compilation or execution boundaries, run a bounded
integration configuration before widening the matrix:

.. code-block:: console

   OPENHCS_CPU_ONLY=1 python -m pytest tests/integration/test_main.py \
       --it-backends disk \
       --it-microscopes ImageXpress \
       --it-dims 3d \
       --it-exec-mode multiprocessing \
       --it-zmq-mode direct \
       --it-visualizers none \
       --it-sequential none \
       -xvs

The option registry in ``tests/pytest_integration_options.py`` is the authority
for accepted integration values. Use ``pytest --help`` rather than copying the
complete option set into another guide.

Compare branches
----------------

Use separate output files and environments for each worktree. Record the exact
commit and recursive submodule revisions beside each result:

.. code-block:: console

   git rev-parse HEAD
   git submodule status --recursive
   OPENHCS_CPU_ONLY=1 python -m pytest tests/unit -q > feature-results.txt 2>&1

Repeat from the comparison worktree, then compare the result files with
``diff``. Tracked fixtures are checked out independently in each worktree.
External datasets may be shared only when both runs intentionally reference the
same immutable location; keep generated outputs in worktree-specific or
run-specific directories.

Remove the worktree
-------------------

First inspect the worktree for uncommitted or untracked files:

.. code-block:: console

   git -C ../openhcs-feature status --short

After preserving anything needed, remove the registered worktree from the main
checkout:

.. code-block:: console

   git worktree remove ../openhcs-feature
   git worktree prune
   git worktree list

Do not delete Git worktree lock files manually. If an intentionally locked
worktree should become removable, use ``git worktree unlock
../openhcs-feature`` and inspect its status again.

See :doc:`pipeline_debugging_guide` for selecting runtime evidence,
:doc:`omero_testing` for live OMERO boundaries, and
:doc:`../guides/testing_guide` for the wider test strategy.
