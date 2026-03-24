1hk4 Docking Pipeline Audit
==========================

This note records a terminal-level deconstruction of the full-chemistry docking
path on ``1hk4`` and audits the runtime against the current Lean ground truth.

Scope
-----

- benchmark target: ``1hk4``
- chemistry: full feature mode
  - ``ChargeMethod.GASTEIGER``
  - ``ExactChemistryMode.EXTENDED_RICH``
  - ``CertifiedScoringFamily.LJ_REALSPACE_EWALD``
- known conformer sanity check first
- objective: remove runtime artifacts that diverge from the theorem-backed model

Lean Ground Truth Used
----------------------

The runtime should mirror these proof families:

- ``SeedBudgetDerivation.lean``
  - geometric CDF inversion for seed budgets
  - monotonicity of conservative basin underestimates
  - composed two-phase guarantee
- ``BlindConformerPipelineOptimality.lean``
  - canonical retain set is the theorem-backed pruning object
- ``BlindConformerPipelineRefinements.lean``
  - retain-set size is emergent from certified lower bounds
  - minimal adequate seed budgets are optimal once adequacy is instantiated
- ``BlindConformerRuntimeCertificates.lean``
  - end-to-end canonical pruning plus canonical refinement budget
- ``EnergyRMSDConvergence.lean``
  - local spectral / curvature certificates
  - canonical adequate refinement budget
  - logarithmic sufficient iteration bound

Pipeline Map
------------

The known-conformer benchmark path goes through:

1. ``benchmark_pdb.py``
   - prepare receptor and ligand structures
   - build exact-chemistry charge / receptor annotations
   - construct a certified blind docking request
2. ``pipeline.py``
   - derive route
   - generate rigid SE(3) seed poses
   - certified pruning pass
   - exact rescoring of canonical survivors
   - formal coordinate-space local search
   - SE(3) certified refinement / certification layer
3. ``scoring_context.py`` and ``scoring.py``
   - exact physics score
   - exact rich chemistry score
   - water bridge and receptor-flex adjustments
4. ``se3_refinement.py``
   - local spectral certificate
   - observed / certified GD refinement certificate

What Was Wrong
--------------

Two concrete runtime divergences were identified.

1. Fixed survivor capacity

The certified pruning route still imposed a fixed survivor cap. This is not part
of the Lean model: the canonical retain set size is emergent from the certified
lower-bound threshold. Large seed budgets therefore failed for runtime reasons,
not theoremic ones.

2. Certified refinement consumed stale pose vectors

Formal round refinement operates on coordinates. The subsequent SE(3) certified
refinement step was still being fed the pre-formal translation/quaternion pair,
not the formally refined pose. This broke the intended proof composition:

- formal rounds selected and improved a basin in coordinate space,
- but the SE(3) layer was certifying a different state.

Observed Failure Signature
--------------------------

Before the runtime fixes, the full benchmark could produce absurd known-conformer
results, including extremely large RMSD values and zero-ish energies caused by the
refinement layer effectively certifying the wrong pose state.

Proof-Aligned Runtime Fixes
---------------------------

The following fixes are theorem-aligned plumbing, not heuristic tuning:

1. Remove the fixed survivor cap from the certified pruning route.
   The runtime now keeps the full certified retain set.

2. Recover the rigid SE(3) pose corresponding to the formally refined coordinates
   via the deterministic Kabsch solution, then hand that pose to the SE(3)
   refinement/certification layer.

Known-Conformer Diagnostics
---------------------------

Using the deconstructed terminal path on ``1hk4`` with full chemistry,
``target_rmsd = 1.0`` and ``confidence = 0.99``:

.. list-table:: Verified operating points
   :header-rows: 1

   * - ``n_poses``
     - total time (s)
     - survivors
     - actual RMSD vs native (A)
     - exact score
   * - ``16384``
     - ``25.58``
     - ``323``
     - ``1.3583``
     - ``-10.2447``
   * - ``32768``
     - ``21.63``
     - ``548``
     - ``1.1005``
     - ``-10.5587``
   * - ``49152``
     - ``25.07``
     - ``952``
     - ``0.8862``
     - ``-11.3463``
   * - ``65536``
     - ``27.09``
     - ``1216``
     - ``0.9804``
     - ``-11.3887``

Interpretation
--------------

- The practical ``< 1 A`` target is already achievable on ``1hk4`` within
  roughly ``25`` seconds using full chemistry and known conformer.
- The measured RMSD listed above is the actual RMSD against the native ligand,
  not a surrogate proxy.
- The main remaining tractability issue is not physics correctness but seed-budget
  instantiation and large-budget batching.

Remaining Proof/Runtime Gaps
----------------------------

1. Probe phase instantiation

The current probe-phase implementation still needs to be brought fully in line
with the newer seed-budget proofs. The correct direction is:

- conservative two-phase certification only,
- no ad hoc probe inflation,
- no runtime-only constants justified by habit rather than proof.

2. Large-budget tractability

Very large theorem-backed seed budgets should be streamed or chunked rather than
materialized as a single giant in-memory pose batch. This is a runtime execution
issue, not a theorem issue.

3. Unknown conformer

The known-conformer path is now close to the desired regime. Unknown-conformer
performance should be debugged only after the two-phase proof-backed seed-budget
instantiation is tightened, so that failures are not caused by a runtime budgeting
artifact.
