DQ-Dock Architecture And Debugging Guide
========================================

This guide explains the practical runtime architecture of DQ-Dock and the exact
debugging patterns that have been useful for auditing ``1hk4`` and related
certified-docking failures.

It is written for engineers and agents who need to debug the system in the
terminal without re-learning the whole codebase first.

Scope
-----

This document focuses on:

- the certified pocket-guided pipeline
- how the major runtime stages fit together
- what each stage is supposed to guarantee
- how to isolate failures to one stage at a time
- the concrete shell / inline-Python debugging patterns that have already been
  useful in practice

Contextless Agent Startup
-------------------------

If you are dropped into this repository with no prior context, do this first.

1. Use the right Python.

   Real docking / benchmark work in this repository should normally run under the
   project virtualenv with RDKit available:

   .. code-block:: bash

      source .venv/bin/activate

   If RDKit is missing, nothing downstream is trustworthy.

2. Read these files before searching broadly.

   - ``dq_dock_engine/docking/pipeline.py``
   - ``dq_dock_engine/docking/scoring_context.py``
   - ``dq_dock_engine/docking/scoring.py``
   - ``dq_dock_engine/docking/formal_optimizer.py``
   - ``dq_dock_engine/docking/conformer_search.py``
   - ``docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/HandleAliases.lean``
   - ``docs/source/development/dq_dock_agent_handoff_guide.rst``

3. Treat Lean as the source of truth for what is allowed.

   Runtime bugs are the only bugs. If the runtime wants a stronger or cheaper
   certificate, the work is:

   - find an existing theorem and instantiate it correctly, or
   - prove the missing theorem first, then wire the runtime to cite it

4. Do not trust apparent runtime wins until you know which theorem supports them.

   In this codebase, a speedup is only legitimate if one of the following is
   true:

   - it is a pure implementation optimization that preserves the exact same
     certified predicate
   - it switches the runtime to a stronger already-proved theorem
   - it adds a new proved theorem and then uses it

5. Start with profiling only after the theorem surface is understood.

   If the theorem side is clearly incomplete, do not benchmark endlessly. Prove
   and wire the missing theorem first, then profile the remaining bottlenecks.

Big Picture
-----------

For certified pocket-guided docking, the runtime pipeline is:

1. benchmark / request construction
2. certified pocket preparation
3. rigid seed-family generation
4. certified pruning + exact rescoring
5. formal local refinement
6. certified SE(3) refinement / probe certification
7. final exact ranking / winner certification

The core lesson from ``1hk4`` debugging is that you should treat each stage as a
separate theorem-backed subsystem. Do not jump straight to the final RMSD.

Main Runtime Files
------------------

``dq_dock_engine/benchmark/benchmark_pdb.py``
   Benchmark CLI, dataset loading, real-structure request construction.

``dq_dock_engine/docking/pipeline.py``
   Main certified docking orchestration. This is where most cross-stage bugs show
   up and where many theorem-backed runtime budgets are stitched together.

``dq_dock_engine/docking/scoring_context.py``
   Builds the exact/coarse score families and selects the context used for
   pruning, local optimization, and final ranking.

``dq_dock_engine/docking/scoring.py``
   Exact and coarse kernels for LJ, Ewald electrostatics, directional H-bonds,
   pi terms, metal coordination, and related runtime score families.

``dq_dock_engine/docking/formal_sampling.py``
   Deterministic certified rigid seed families for box and binding-site sampling.

``dq_dock_engine/docking/formal_optimizer.py``
   Formal local rigid optimizer over finite action families.

``dq_dock_engine/docking/formal_actions.py``
   Action-family step geometry, dyadic shell growth, and translation / rotation
   step derivation.

``dq_dock_engine/docking/se3_refinement.py``
   Observed and certified SE(3) refinement certificates.

``dq_dock_engine/docking/certified_runtime_plans.py``
   Runtime proof-plan dataclasses. If you need to understand what the pipeline is
   *claiming* to have proved, start here.

``dq_dock_engine/docking/formal_handles.py``
   The runtime-to-Lean provenance map. If a runtime path is not citing a handle
   group here, assume it is not yet theorem-owned.

``docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/*.lean``
   The actual proof surface for tractability and certification. Runtime budgets
   should be traceable back to one of these files.

Core Runtime Objects
--------------------

The key runtime objects that an agent should recognize immediately are:

``PipelineDockingRequest``
   The complete theorem-relevant runtime request. This is the object whose fields
   determine whether a path is rigid-only, conformer-active, pocket-guided,
   receptor-flexible, rich-chemistry, etc.

``CertifiedScoringContext``
   Encodes which certified score family is active for a stage and which theorem
   obligations go with it.

``CertifiedPruningDeltaBudget``
   Authoritative proof object for pruning slack. If pruning looks too weak, this
   object is the first thing to inspect.

``PoseSpecificImprovementBudgetFamily``
   Authoritative conformer / local-improvement budget family. This is the object
   that usually decides whether a pose can be skipped before an expensive stage.

``RefinementCertificate``
   Rigid refinement proof object. Missing refinement certificates usually block
   returned-pose RMSD guarantees.

``CertifiedReturnedPoseProofPlan``
   The final proof-plan summary for returned-pose certification. This is where the
   runtime states whether it has:

   - target-RMSD certification,
   - only energy-gap certification, or
   - a downgrade.

Certification Layers
--------------------

DQ-Dock has multiple proof layers. They are related, but they are not the same.

``Gap proof``
   Proves an energy ordering statement, usually that the returned or native pose
   wins with a certified margin under the final exact score family.

``Returned-pose certification``
   Proves a statement about the returned pose contract. Depending on the runtime
   evidence, this can be:

   - target RMSD certification,
   - certified ambiguity-set target RMSD,
   - certified energy-gap only,
   - or a downgrade.

``Pruning certification``
   Proves that survivor pruning is safe despite coarse scoring / omitted channels.

``Improvement-budget certification``
   Proves that a skipped local or conformer optimization step cannot recover more
   than the certified bound.

An agent should never conflate these. A benchmark can have a proved native gap
while still lacking a returned-pose RMSD certificate.

Theorem Family Map
------------------

These theorem groups matter most for current DQ-Dock runtime work.

Rigid pruning and lower bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BCPO*`` / ``BCRC*``
   Blind conformer pipeline optimality/runtime certificates. These drive canonical
   retain logic, omitted-channel reasoning, and pose-specific improvement bounds.

``CSC*``
   Conformer support coverage. These justify active torsion subsets, sparse
   supports, and conformer-cover energy-gap reasoning.

Rigid local refinement
~~~~~~~~~~~~~~~~~~~~~~

``FLO*``
   Formal local optimizer theorems.

``ERC*``
   Energy/RMSD convergence certificates for rigid refinement.

Conformer runtime budgets
~~~~~~~~~~~~~~~~~~~~~~~~~

``LSA*``
   Ligand strain approximation and strain-aware dominance.

``CS*``
   Conformer search theorems and branch-and-bound certificates.

``GAP*``
   Generic lower-bound gap theorems used by search and pruning logic.

Nonbonded score control
~~~~~~~~~~~~~~~~~~~~~~~

``LJ*``
   Exact and softened Lennard-Jones approximation / Lipschitz / lower-bound
   theorems.

``CB*``
   Real-space Ewald / Coulomb control and electrostatic envelopes.

If you change a runtime budget and cannot name the handle family that justifies
it, stop and prove the missing theorem first.

No-Heuristics Rule
------------------

DQ-Dock debugging has one hard rule:

- do not silently replace a missing proof obligation with a heuristic

Valid runtime changes are limited to:

- exact implementation optimizations with identical certified semantics
- switching to a stronger existing theorem-backed runtime witness
- proving a new theorem and then using it

Invalid examples:

- reusing a basin witness from a different conformer geometry
- substituting an energy-gap bound where an RMSD theorem still requires basin
  curvature
- trimming search spaces without a proof that the omitted region is bounded or
  irrelevant

Current Runtime Bottleneck Hierarchy
------------------------------------

For hard flexible ligands such as ``1gni``, the main costs are currently:

1. certified rigid pruning on the large sampled family
2. formal local refinement on the retained rigid candidates
3. conformer improvement budgets that remain too large for high-active-torsion
   final poses

The current theorem grind has already reduced many fake bottlenecks. What remains
now is mostly *real certified slack*, not accidental overhead.

Profiling Strategy That Matches The Proof Workflow
--------------------------------------------------

Use profiling in this order:

1. prove / instantiate the obvious missing theorem
2. only then measure the remaining runtime
3. add phase-level profiling markers around theorem-owned stages
4. do not benchmark every small code edit while the theorem surface is still
   clearly incomplete

Good profiling questions:

- which theorem-owned stage still dominates runtime?
- is the cost coming from setup / compilation overhead or from a genuinely large
  certified family?
- is the runtime spending time on a stage that should have been skipped by an
  already-proved bound?

Bad profiling questions:

- how do I make this fast if the current theorem still allows a huge search
  region?
- can I just trim this candidate family manually?

Current High-Value Theorem Targets
----------------------------------

If you need to resume theorem work with minimal context, these are the current
best targets.

1. Local nonbonded headroom theorems

   We already tightened torsion strain and part of LJ. The remaining large
   conformer budgets likely still require sharper theorem-backed headroom for:

   - exact LJ in the attractive tail,
   - real-space Ewald / screened Coulomb in local families,
   - and family-specific omitted-channel bounds.

2. Family-specific receptor omission

   Use omitted-channel theorems to prove that dropping far receptor atoms for a
   given conformer family only changes the score by a certified additive budget.
   This should reduce conformer-search runtime without cheating.

3. Tighter conformer improvement aggregation

   Avoid overcounting the same moving geometry through multiple torsion-local
   bounds. The generic bounded-channel-sum theorems suggest sharper instantiations
   than the naive per-bond sum.

4. Winner-band restriction before expensive stages

   If the current theorem surface supports it, restrict expensive refinement /
   conformer stages to the certified winner band rather than the full retained
   rigid set.

Conceptual Splits That Matter
-----------------------------

There are three score-family concepts that are easy to conflate:

``optimization_context()``
   Exact score used by the formal local optimizer.

``pruning_context()``
   Exact/coarse score family used for theorem-backed survivor pruning.

``ranking_context()``
   Final exact score used for pose ranking after optimization/refinement.

Many ``1hk4`` failures came from stage mismatch rather than a single bad score.
Always ask which context is active before debugging a ranking or pruning issue.

Benchmark Modes That Matter
---------------------------

``--use-crystal-ligand-geometry``
   Use crystal internal ligand geometry as the starting ligand template.

``--disable-conformer-search``
   Disable torsion search after rigid placement.

``--reuse-initial-conformer``
   Only relevant when conformer search is enabled; preserves the starting
   conformer inside conformer search.

``--n-poses-override``
   Debug-only override for the derived rigid seed budget. This is useful for
   controlled stage audits while the seed-budget reduction path is still being
   improved.

Known-conformer debugging mode is:

.. code-block:: bash

   --use-crystal-ligand-geometry --disable-conformer-search

Architecture By Stage
---------------------

Request Construction
~~~~~~~~~~~~~~~~~~~~

The benchmark layer builds a certified blind docking request with:

- receptor coordinates and radii
- ligand coordinates, radii, and chemistry annotations
- charge model
- target RMSD and confidence
- certified scoring family and chemistry mode

Failure modes here are mostly configuration mismatches, wrong file inputs, or
using the wrong Python environment.

Certified Pocket Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pocket preparation converts the detected or benchmark pocket into a certified
binding-site object. The binding-site center and radius are later used by:

- pocket-guided rigid sampling
- local-refinement binding-site confinement
- debugging sanity checks for off-pocket failures

If a supposedly good pose ends up well outside the certified binding-site
sphere, either sampling or refinement escaped the intended domain.

Rigid Seed Family Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rigid sampler in ``formal_sampling.py`` creates a deterministic family of:

- translation centers inside the certified binding-site sphere
- a fixed quaternion dictionary

Recent important fixes here were:

- sampling the certified sphere rather than its enclosing cube
- removing flattened truncation bias from the translation-quaternion product

When debugging sampling, the first question is always:

"Does the seed family already contain a near-native pose?"

Certified Pruning And Exact Rescoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pruning stage does two jobs:

- compute a certified retain mask
- produce exact survivor scores for the next stage

There are two distinct tractability concerns here:

1. coarse pruning delta may be too loose
2. even after exact rescoring, too many poses may still be eligible for local
   refinement

The current runtime now also uses a theorem-backed rigid future-improvement
bound to cut the local-refinement set before refinement actually runs.

Formal Local Refinement
~~~~~~~~~~~~~~~~~~~~~~~

The formal local optimizer searches finite rigid action families with dyadic step
shrinking. Important invariants:

- rotations must act about each pose centroid, not the world origin
- support expansion must be tracked per pose, not globally across the batch
- candidate actions must stay inside the certified binding-site sphere in
  known-pocket mode
- the root rotation step should match the translational cell scale, not jump to
  ``pi/2``

This stage is where many wrong-family wins were introduced before the recent
fixes.

SE(3) Refinement And Probe Certification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SE(3) refinement layer has two different roles:

- final rigid refinement/certification of already-selected poses
- probe-time basin certification for seed-budget reduction

The hard current gap is that probe certification still depends on a lower-
curvature certificate. On ``1hk4``, the observed spectral certificate often
exists now, but it is still too weak to reduce the seed budget materially.

Final Ranking
~~~~~~~~~~~~~

Final ranking is separate from local optimization. A pose may be good under the
optimizer objective and still lose under final exact ranking, or vice versa.

That is why the debug pattern is always:

- sample audit
- score-family audit
- local-refinement audit
- only then final-ranking audit

Terminal Debugging Patterns
---------------------------

The rest of this guide lists the concrete debugging patterns that have already
been useful. These are not abstract suggestions; they are the exact kinds of
terminal audits that have been effective on ``1hk4``.

1. Timeout-Wrapped Benchmark Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use these when you need realistic end-to-end behavior without leaving the shell
busy forever.

Pattern:

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

Use this pattern for:

- verifying a fix did not break the real benchmark path
- capturing benchmark output safely in a log file
- comparing known-conformer and unknown-conformer modes

2. Stage-Only Inline Python Harnesses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use small inline scripts to stop the pipeline after a specific stage and inspect
the intermediate objects directly.

Typical structure:

.. code-block:: python

   blind_request = CertifiedBlindDockingRequest(...)
   prep = _prepare_certified_blind_docking(...)
   pipeline_request = derive_request(...)
   request = nominalize_pipeline_request(...)
   route = derive_pipeline_route(request)
   request = route.prepare_request(request)
   pose_batch = route.generate_pose_batch(request)

Use this pattern when you need to audit:

- sampled pose families
- prepared boxes / binding-site geometry
- route-dependent score contexts
- exact survivor counts before later stages mutate them

3. Oracle Seed Audit
~~~~~~~~~~~~~~~~~~~~

Compute raw docking RMSD for every sampled seed and record the best one.

Pattern:

- generate the rigid pose batch
- apply poses to ligand coordinates
- compute ``compute_docking_rmsd_batched(...)`` over the full batch

Use this to answer:

- is sampling fundamentally missing the basin?
- or is a later stage discarding / mis-ranking it?

4. Raw RMSD vs Kabsch RMSD Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always compute both:

- raw docking RMSD
- Kabsch RMSD

If raw RMSD is large but Kabsch RMSD is tiny, you have a rigid-family failure,
not a conformer failure.

This was a decisive debugging pattern for ``1hk4``.

5. Score-Family Ranking Audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the exact same seed set, compute ranks under:

- ``optimization_context().score_exact_batch(...)``
- full ``score_exact_batch(...)``
- ``score_flip_disambiguation_batch(...)``

Use this to determine whether the bad winner originates from:

- base physics
- rich chemistry
- orientation-disambiguation logic

6. Certified Pruning Audit
~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``_certified_pruning_pass(...)`` directly and print:

- pruning delta
- survivor count
- best raw RMSD among survivors
- oracle rank under the coarse/pruning score

This isolates whether pruning is too loose, too aggressive, or simply targeting
the wrong score family.

7. Local-Refinement A/B Audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refine a small number of hand-picked poses individually.

Useful comparisons:

- oracle seed vs top-ranked seed
- oversized root rotation vs cell-scale root rotation
- softened local coarse vs exact local coarse
- pocket-constrained vs unconstrained local refinement

This pattern is the fastest way to determine whether local refinement is helping,
hurting, or doing nothing.

8. Batch-Dependence Audit
~~~~~~~~~~~~~~~~~~~~~~~~~

Run local refinement on a pose alone, then refine it again in a batch with an
unrelated pose. If the result changes, you have batch-coupled runtime state.

This pattern exposed the earlier exact-receptor-subset and support-expansion
coupling problems.

9. Hessian / Spectral Audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute Hessian eigenvalues at:

- the native pose
- the best sampled seed
- the locally refined winner
- multiple scoring contexts (optimization, ranking, full)

Use this to answer whether the observed probe certificate is failing because the
pose is actually outside a convex basin, or because the runtime is using the
wrong energy function for certification.

10. Component Score Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare native vs wrong winning pose under individual channels:

- LJ
- screened Coulomb / Ewald
- contact
- directional H-bond donor/acceptor channels
- metal coordination
- each extended interaction term

Use this when a wrong family still wins after sampling and local refinement.

11. Pocket vs Full-Receptor Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build the same pose pair against:

- the pocket receptor
- the full receptor

Use this to determine whether a wrong winner is an artifact of pocket truncation
or an issue that persists even under the full receptor.

12. Probe-Family Search Audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the probe certification logic across several probe family sizes and compare:

- whether a certificate exists
- the resulting derived seed budget
- the ``q``, ``mu_coord``, and ``M_coord`` values in the certificate

This is the fastest way to inspect the current seed-budget reduction failure.

13. Resolution / Family Geometry Audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Print, for a chosen ``n_poses``:

- quaternion count
- lattice resolution
- number of unique translations
- binding-site radius
- translation cell width

Use this to connect observed sampling quality back to the actual deterministic
family geometry.

14. Controlled Seed Override Audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--n-poses-override`` only to answer questions like:

- what RMSD becomes reachable at ``n = 4096`` vs ``16384``?
- is the derived seed budget too large because of a proof/certificate gap or a
  runtime bug?

This is a debugging tool, not a conceptual certified-mode control.

Debugging Checklist
-------------------

When a certified docking result is bad or slow, use this exact order:

1. confirm the environment and benchmark mode
2. audit sampled oracle RMSD
3. compare raw RMSD vs Kabsch RMSD
4. audit score-family ranks on the seed set
5. audit certified pruning delta and survivor counts
6. isolate local refinement with one-pose A/B tests
7. inspect Hessian/spectral behavior only after the earlier stages are understood
8. compare pocket vs full receptor if a wrong out-of-pocket family still wins

Current ``1hk4`` Runtime State
------------------------------

As of the current debugging pass:

- known-conformer ``1hk4`` succeeds at about ``0.23 A`` raw RMSD when run with
  ``--n-poses-override 16384``
- certified local refinement is cut from ``16384`` poses to ``17`` by a
  theorem-backed rigid future-improvement bound
- the observed probe certificate can now exist on ``1hk4``, but it is still too
  weak to reduce the seed budget materially
- coarse certified pruning is still the biggest remaining tractability gap

Companion Documents
-------------------

- ``docs/source/development/dq_dock_agent_handoff_guide.rst``
  - faster status-oriented handoff for agents
- ``docs/source/development/docking_pipeline_audit_1hk4.rst``
  - older focused ``1hk4`` audit context
- ``plan/theorem-gap-implementation-roadmap.md``
  - roadmap tying the remaining runtime gaps back to Lean theorem families
