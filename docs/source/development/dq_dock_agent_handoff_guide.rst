DQ-Dock Agent Handoff Guide
===========================

This document is a crash course for agents who need to debug the DQ-Dock
pipeline without re-deriving the whole architecture from scratch.

It is written as a handoff guide, not as end-user documentation.

Purpose
-------

Use this guide when you need to answer questions like:

- Why did a benchmark fail or time out?
- Is a bad RMSD coming from sampling, pruning, local optimization, or final
  ranking?
- What does "known conformer" actually mean in this codebase?
- Which scoring context is active at each stage?
- Which files and functions matter first for ``1hk4`` debugging?

Read This First
---------------

- Use ``.venv-rdkit/bin/python`` for real docking and benchmark runs. Plain
  ``python`` in this repo may not have RDKit.
- Wrap long benchmark runs in ``timeout`` and redirect output to a log file.
- Always compute both:

  - raw docking RMSD (absolute Cartesian RMSD against the native ligand), and
  - Kabsch RMSD (shape-equivalence RMSD after optimal rigid alignment).

- If raw RMSD is large but Kabsch RMSD is tiny, you do not have a conformer
  failure. You have a rigid-pose family / orientation-family problem.

Fast Mental Model
-----------------

For the certified benchmark path, the flow is:

1. ``dq_dock_engine/benchmark/benchmark_pdb.py``
   - prepare receptor / pocket / ligand files
   - compute charges if needed
   - build a certified blind docking request
2. ``dq_dock_engine/docking/pipeline.py``
   - prepare certified pocket / binding site
   - derive a nominal pipeline request and route
   - generate rigid SE(3) seed poses
   - run certified pruning and exact rescoring
   - run formal local refinement
   - run rigid SE(3) certified refinement / certification
   - run final ranking and native-gap certification
3. ``dq_dock_engine/docking/scoring_context.py``
   - choose between base physics, full rich chemistry, ranking context, and
     optimization context
4. ``dq_dock_engine/docking/formal_optimizer.py``
   - local finite action-family search over rigid pose updates
5. ``dq_dock_engine/docking/se3_refinement.py``
   - certified / observed rigid GD certificate layer

The most important practical split is:

- ``optimization_context()``: smooth base-physics objective used by the local
  optimizer
- ``ranking_context()`` / full context: richer score used for final pose ranking

That split is deliberate, but it is also a common source of confusion and bugs.

Key Mode / Flag Meanings
------------------------

These three knobs are easy to confuse.

``--use-crystal-ligand-geometry``
   Use the crystallographic ligand *internal geometry* as the ligand template.
   This does not seed the native rigid pose. It only sets bond lengths / angles /
   torsions in the starting ligand frame.

``--disable-conformer-search``
   Disable torsion search after rigid placement. This makes the run rigid with
   respect to ligand conformation.

``--reuse-initial-conformer``
   Only matters when conformer search is enabled. It keeps the starting
   conformer as an explicit candidate inside conformer search.

Useful combinations:

- known conformer: ``--use-crystal-ligand-geometry --disable-conformer-search``
- unknown conformer: omit both of those and keep conformer search enabled

Important Files
---------------

Start with these files before doing broad repo search.

``dq_dock_engine/benchmark/benchmark_pdb.py``
   Benchmark entry points, CLI flags, benchmark-mode request construction.

``dq_dock_engine/docking/pipeline.py``
   Main docking orchestration. This is the first file to inspect when a run is
   correct in principle but wrong in practice.

``dq_dock_engine/docking/scoring_context.py``
   Decides which score family is active in pruning, optimization, and final
   ranking.

``dq_dock_engine/docking/scoring.py``
   Exact and softened scoring kernels, including softening error bounds.

``dq_dock_engine/docking/formal_optimizer.py``
   Rigid local optimization on certified finite action families.

``dq_dock_engine/docking/formal_actions.py``
   Translation / rotation action families and adaptive step logic.

``dq_dock_engine/docking/se3_refinement.py``
   Final rigid SE(3) certificate layer.

``docs/source/development/docking_pipeline_audit_1hk4.rst``
   Historical ``1hk4`` audit. Useful context, but some numbers are now stale.

Current ``1hk4`` Findings
-------------------------

As of the current debugging pass, the most important confirmed facts are:

1. Known-conformer sampling is already close.

   With ``n_poses = 16384`` on ``1hk4`` in known-conformer mode, the certified
   seed family already contains an oracle pose at about ``0.390 A`` raw RMSD.
   So sampling is not fundamentally broken.

   The tiny probe family is a different story: with the current fixed
   ``SEED_BUDGET_PROBE_POSES = 16``, the probe-stage oracle best raw RMSD is only
   about ``5.23 A``. So the current probe phase is far too sparse to discover the
   near-native rigid basin on ``1hk4``.

   Two concrete sampling bugs were also present and are now fixed:

   - binding-site sampling used the enclosing cube of the certified binding-site
     ball, so off-pocket pose centers inside the cube but outside the certified
     radius were being sampled
   - global pose generation over-produced translation cells and then truncated the
     translation-quaternion product in flattened order, which biased coverage
     toward an initial slab of the box instead of the intended spatial budget

2. The bad ``1hk4`` outputs are mostly rigid-family failures, not conformer failures.

   For the bad top-ranked pose families, raw RMSD is around ``10-14 A`` while
   Kabsch RMSD is near zero. That means the code often finds a rigidly
   equivalent-looking family but not the native Cartesian pose.

3. Base-physics scoring ranks the wrong family on unoptimized seeds.

   On the same ``16384``-pose known-conformer run:

   - the oracle raw-RMSD seed ranks far from the top under base exact scoring
   - the top-ranked base-physics seed is around ``12.74 A`` raw RMSD

4. Full rich scoring on unoptimized seeds also still ranks the wrong family.

   Rich exact scoring improves energy discrimination, but on the tested
   ``16384``-pose seed set it still prefers a wrong rigid family. So this is not
   solved simply by swapping the score family at the seed stage.

5. Chunking was required for tractability.

   Large certified seed budgets must not be materialized as one giant monolithic
   score/prune tensor. The runtime now chunks certified pruning work instead of
   forcing the whole batch through one device allocation.

6. Rigid seed budgets must ignore torsions.

   The rigid global seed count is an SE(3) budget, not a joint rigid+torsion
   budget. Inflating the rigid seed count by rotatable bonds caused severe
   blowups for flexible ligands.

7. Softening-error deltas are tricky.

   For base physics, the softened-vs-exact delta taken directly from a pose batch
   can become enormous because a single severe clash can dominate the batchwise
   max error bound. When that happens, certified pruning can degenerate and keep
   nearly everything.

8. Formal local refinement was not obviously helping ``1hk4``.

   On the current debug path, refining the oracle seed and the base-top-ranked
   seed did not move them. This means the current ``1hk4`` failure is not just a
   final-ranking issue; the local improvement stage also needs scrutiny.

   After deeper auditing, that broad statement can be sharpened:

   - softened local coarse scoring can freeze the singleton selector into the
     noop action via a huge ambiguity band
   - exact local-family scoring with the lattice half-step does improve the
     oracle seed materially on ``1hk4``
   - the current formal optimizer behavior also depends on batch composition,
     because the exact receptor subset is chosen once for the whole batch

   Latest audited runtime state:

   - the optimizer now uses a theorem-aligned cell-scale root rotation step
     instead of ``pi/2``
   - local refinement is constrained to stay inside the certified binding-site
     sphere
   - a theorem-backed rigid future-improvement bound is now used to cut the
     local-refinement set before the expensive refinement loop
   - on ``1hk4`` known-conformer mode with ``n_poses_override=16384``, this cuts
     local refinement from ``16384`` poses to ``17`` while still returning a
     best pose around ``0.23 A`` raw RMSD

9. One concrete local-optimization bug to watch closely:

   candidate rigid actions must rotate around each pose centroid, not around the
   global origin. If rotation is applied to already-placed world coordinates
   around the origin, local search proposals become physically wrong and can
   collapse to no-op behavior.

11. The formal local rotation step was oversized.

    The old local optimizer root rotation step was effectively ``pi/2``. On
    ``1hk4`` that let local refinement jump into wrong rigid families. The safer
    geometric derivation is to match rotation-induced pointwise displacement to
    the translation cell scale:

    ``base_rotation_step_rad = base_translation_step / ligand_radius``.

    In current audits, this preserves near-native winners much better than the
    old oversized rotation root step.

12. Known-pocket local refinement should respect the certified binding site.

    The local action family is now constrained to keep candidate pose centers
    inside the certified binding-site sphere. This is not a heuristic penalty;
    it is enforcing the known-pocket search domain.

10. The current probe-phase reduction is blocked by an observed-certificate gap,
    not by the rigid seed runtime bridge theorem.

    Even on near-native ``1hk4`` seeds, the observed SE(3) refinement certificate
    can still be ``None`` because the local SE(3) Hessian is not positive-definite
    enough to produce the spectral certificate needed by the current observed
    certification bridge. This means the runtime falls back to the very
    conservative zero-step seed-budget derivation.

11. ``n_poses_override`` exists again on the benchmark CLI.

    This is explicitly for controlled debugging runs while the observed
    Hessian/spectral certificate path for probe-time seed-budget reduction is
    still incomplete. It should not be treated as a conceptual user control for
    certified mode.

How To Diagnose A Failure Quickly
---------------------------------

Use this order. It avoids wasting time.

1. Check whether the sampled family already contains a good pose.

   - Generate the initial pose batch.
   - Compute raw docking RMSD for all seeds.
   - Record ``oracle_best_raw``.

   If ``oracle_best_raw`` is still bad, the problem is sampling / seed-budget
   instantiation.

2. If sampling is good enough, compare ranking families.

   On the same seed set, compute ranks for:

   - base exact score (``optimization_context``)
   - full rich exact score
   - flip-disambiguation score

   If all of them rank the oracle seed poorly, then local optimization must do
   more work; pruning alone will not save the run.

3. If scoring still ranks the oracle poorly, isolate local optimization.

   Pick:

   - the oracle-best seed by raw RMSD, and
   - the top-ranked seed by the active pre-optimization score

   Refine them individually and compare before/after raw RMSD.

   Also compare two local-optimization settings:

   - oversized root rotation step (old path), and
   - cell-scale root rotation step ``base_step / ligand_radius``.

   On ``1hk4``, the latter materially changes which rigid family wins after local
   refinement.

4. Only after that should you inspect final certification / ranking.

   The final certification layer is often blamed too early. On ``1hk4`` the real
   problems usually happen before that.

Practical Reproduction Commands
-------------------------------

Known-conformer benchmark:

.. code-block:: bash

   timeout 420 ".venv-rdkit/bin/python" -m dq_dock_engine.benchmark.benchmark_pdb \
     --pdb_ids 1hk4 \
     --competitors none \
     --workers 1 \
     --results_dir benchmark_1hk4_known_debug \
     --target-rmsd 1.0 \
     --confidence 0.99 \
     --use-crystal-ligand-geometry \
     --disable-conformer-search \
     --attempt-timeout-seconds 360 \
     --max-retries 1 > /tmp/1hk4_known_debug.log 2>&1

Controlled debug run with explicit rigid seed count:

.. code-block:: bash

   timeout 600 ".venv-rdkit/bin/python" -m dq_dock_engine.benchmark.benchmark_pdb \
     --pdb_ids 1hk4 \
     --competitors none \
     --workers 1 \
     --results_dir benchmark_1hk4_known_debug \
     --target-rmsd 1.0 \
     --confidence 0.99 \
     --use-crystal-ligand-geometry \
     --disable-conformer-search \
     --n-poses-override 16384 \
     --attempt-timeout-seconds 540 \
     --max-retries 1 > /tmp/1hk4_known_debug.log 2>&1

Focused regression tests that are cheap and relevant while iterating:

.. code-block:: bash

   python -m pytest -o addopts='' \
     dq_dock_engine/tests/test_seed_budget.py \
     dq_dock_engine/tests/test_certified_pruning_topk.py

What To Print In A Pipeline Dissection Script
---------------------------------------------

If you write an ad hoc Python debug snippet, print at least these:

- ``n_poses``
- ``oracle_best_raw``
- ``oracle_best_kabsch``
- top-10 score/rmsd rows for:

  - base exact
  - rich exact
  - flip-disambiguation

- survivor count after pruning
- survivor-best raw RMSD
- final returned pose raw RMSD and Kabsch RMSD
- theorem handles on the returned best pose

This is the minimum set that lets a new agent tell whether the failure is in
sampling, pruning, local search, or final ranking.

Base Physics vs Rich Chemistry
------------------------------

This is the conceptual split that matters most.

``optimization_context()``
   Strips extended-rich channels, water bridges, and receptor-flex adjustments.
   It keeps the smooth base-physics objective used by the formal local optimizer.

``score_exact_batch()`` on the full context
   Uses the richer chemistry model, including extended-rich channels.

``ranking_context()``
   Uses the strongest currently trusted final-ranking context, with a fallback to
   the optimization context if orientation-disambiguation signals are inactive.

Do not casually collapse these into one concept while debugging. Many apparent
"scoring bugs" are really context-selection bugs.

Current High-Value Debug Targets
--------------------------------

If you are taking over this investigation, the next places to inspect are:

1. ``dq_dock_engine/docking/formal_optimizer.py``

   Confirm that local rigid actions are centered correctly and that per-pose
   support-expansion state does not bleed across unrelated poses in the same
   batch.

2. ``dq_dock_engine/docking/formal_actions.py``

   Audit ``compute_adaptive_translation_step()`` on ``1hk4``. The current
   softened-Lipschitz scaling can inflate a roughly ``1 A`` lattice spacing into
   a translation step on the order of ``10+ A``, which is suspicious for local
   refinement.

   On the audited ``1hk4`` path, the scaled step was about ``15.86 A`` while the
   unscaled half-cell step was about ``0.50 A``. The unscaled exact local search
   improved the oracle seed; the scaled softened path did not.

3. ``dq_dock_engine/docking/formal_sampling.py``

   Keep the implementation aligned with the actual geometry proved by the binding
   site theorems. For certified binding sites, that means sampling the certified
   ball rather than its enclosing cube, and respecting the intended translation
   budget instead of truncating a larger Cartesian product in flattened order.

4. ``dq_dock_engine/docking/pipeline.py``

   Reconcile the pruning context with the optimization context. The runtime wants
   a manageable retain set but must not discard the basin that can refine into
   the native pose.

   Also audit the two-phase seed-budget plumbing: the current fallback to the
   zero-step budget is mathematically conservative but practically enormous on
   ``1hk4``.

   The runtime now contains a new rigid local-improvement filter before formal
   local refinement. If you touch this code, preserve the invariants:

   - the bound is derived from dyadic local-action path length, not a fixed cap
   - the bound is with respect to the optimization objective used by the formal
     local optimizer
   - local SE(3) refinement must not be allowed to leave the certified binding
     site in known-pocket mode

5. ``dq_dock_engine/docking/scoring_context.py``

   Be explicit about when pruning should use:

   - batchwise softened-vs-exact bounds
   - analytic rich-chemistry bounds
   - final ranking context

6. ``1hk4`` scoring decomposition

   On the current audited path, the wrong off-pocket rigid family is favored even
   by base physics, and the strongest extra rich contribution comes from the
   pi-cation channel. Audit component-wise contributions before making broad
   scoring changes.

7. ``1hk4`` unknown conformer mode

   Do not spend serious time on unknown conformer until known conformer is below
   target. Otherwise you will mix rigid-family bugs with conformer-search bugs.

Current Goal State
------------------

The immediate target is simple:

- known-conformer ``1hk4`` should get below ``0.5 A`` raw RMSD reliably

At the moment, the strongest granular evidence is:

- the rigid seed family can already place a near-native pose
- after the sampling fixes, the best known-conformer seed at ``16384`` is already
  below ``0.5 A`` raw RMSD
- the fixed tiny probe family cannot
- softened local singleton refinement is currently too conservative / ambiguous
- exact local refinement with lattice-scale steps can move the oracle seed below
  ``0.5 A`` when audited in isolation
- exact local refinement becomes much better behaved once the root rotation step
  is tied to the translational cell scale instead of starting from ``pi/2``
- the benchmark path now returns ``1hk4`` known-conformer at about ``0.23 A`` raw
  RMSD when run with ``--n-poses-override 16384``
- the current large derived seed budget persists because the observed SE(3)
  certification bridge is still failing on ``1hk4``

Only after that should the debugging effort widen back out to unknown conformer,
seed-budget tightening, or broader benchmark quality.
