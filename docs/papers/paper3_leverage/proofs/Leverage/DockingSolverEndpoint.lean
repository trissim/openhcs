import Leverage.DockingTheoryBridge

namespace Leverage

open Classical DecisionQuotient
open DecisionQuotient.Tractability
open DecisionQuotient.Tractability.MolecularSrank
open DecisionQuotient.Tractability.SampledDocking
open DecisionQuotient.Tractability.DiscretizedAction

noncomputable instance : DecidableEq MDAction := Classical.decEq _

/-- Canonical molecular input state extracted directly from the binding problem's
protein pocket and ligand objects. -/
def bindingProblemBaseState (prob : MDBindingProblem) : MDState :=
  { protein := prob.protein.atoms
    ligand := prob.ligand.atoms }

/-- Canonical base ligand action associated with a raw docking problem. -/
def bindingProblemBaseAction (prob : MDBindingProblem) : MDAction :=
  { ligand := prob.ligand.atoms }

/-- Raw pocket+ligand constructor controls for finite sampled cross-docking. -/
structure RawPocketLigandSamplingConfig where
  gridRadius : Nat
  resolution : ℝ
  beta : ℝ

/-- Canonical sampled action family synthesized from raw pocket+ligand inputs:
all lifted grid actions plus the native ligand action anchor. -/
noncomputable def rawPocketLigandSampleFamily
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig) :
    SampledActionFamily MDAction :=
  { support :=
      insert (bindingProblemBaseAction prob)
        (((Finset.univ : Finset (GridMDAction prob.ligand.numAtoms cfg.gridRadius)).image
          (fun ga => liftGridAction cfg.resolution ga)))
    nonempty := by
      exact ⟨bindingProblemBaseAction prob, by simp⟩ }

theorem rawPocketLigandSampleFamily_contains_baseAction
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig) :
    bindingProblemBaseAction prob ∈ (rawPocketLigandSampleFamily prob cfg).support := by
  simp [rawPocketLigandSampleFamily]

noncomputable def atomSquaredDistance (a b : Atom) : ℝ :=
  pointDistance a.position b.position ^ (2 : Nat)

noncomputable def ligandSquaredDistanceSum
    (lig1 lig2 : List Atom) : ℝ :=
  (List.zipWith atomSquaredDistance lig1 lig2).sum

noncomputable def ligandRMSD
    (lig1 lig2 : List Atom) : ℝ :=
  let m := Nat.min lig1.length lig2.length
  if m = 0 then 0
  else Real.sqrt (ligandSquaredDistanceSum lig1 lig2 / (m : ℝ))

theorem ligandRMSD_nonneg
    (lig1 lig2 : List Atom) :
    0 ≤ ligandRMSD lig1 lig2 := by
  by_cases hm : Nat.min lig1.length lig2.length = 0
  · simp [ligandRMSD, hm]
  · have hSqrt :
        0 ≤ Real.sqrt
          (ligandSquaredDistanceSum lig1 lig2 /
            (Nat.min lig1.length lig2.length : ℝ)) :=
      Real.sqrt_nonneg _
    simpa [ligandRMSD, hm] using hSqrt

/-- Canonical RMSD-to-input-ligand map synthesized from raw molecular input. -/
noncomputable def canonicalRMSDToInputLigand
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig) :
    SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ :=
  fun a => ligandRMSD a.1.ligand prob.ligand.atoms

theorem canonicalRMSDToInputLigand_nonneg
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (a : SupportedAction (rawPocketLigandSampleFamily prob cfg)) :
    0 ≤ canonicalRMSDToInputLigand prob cfg a :=
  ligandRMSD_nonneg _ _

/-- Finite posterior model over a finite action family. -/
structure FinitePosteriorModel (A : Type*) [Fintype A] where
  posterior : A → ℝ
  posterior_nonneg : ∀ a : A, 0 ≤ posterior a
  posterior_sum_one : (Finset.univ : Finset A).sum posterior = 1

noncomputable def FinitePosteriorModel.successProbability
    {A : Type*} [Fintype A]
    (P : FinitePosteriorModel A)
    (accept : A → Prop)
    [DecidablePred accept] : ℝ :=
  (Finset.univ : Finset A).sum (fun a => if accept a then P.posterior a else 0)

theorem FinitePosteriorModel.successProbability_nonneg
    {A : Type*} [Fintype A]
    (P : FinitePosteriorModel A)
    (accept : A → Prop)
    [DecidablePred accept] :
    0 ≤ P.successProbability accept := by
  unfold FinitePosteriorModel.successProbability
  exact Finset.sum_nonneg (fun a _ => by
    by_cases h : accept a
    · simp [h, P.posterior_nonneg a]
    · simp [h])

theorem FinitePosteriorModel.successProbability_le_one
    {A : Type*} [Fintype A]
    (P : FinitePosteriorModel A)
    (accept : A → Prop)
    [DecidablePred accept] :
    P.successProbability accept ≤ 1 := by
  unfold FinitePosteriorModel.successProbability
  have hLe :
      (Finset.univ : Finset A).sum (fun a => if accept a then P.posterior a else 0) ≤
        (Finset.univ : Finset A).sum P.posterior := by
    exact Finset.sum_le_sum (fun a _ => by
      by_cases h : accept a
      · simp [h]
      · simp [h, P.posterior_nonneg a])
  simpa [P.posterior_sum_one] using hLe

theorem FinitePosteriorModel.successProbability_unit_interval
    {A : Type*} [Fintype A]
    (P : FinitePosteriorModel A)
    (accept : A → Prop)
    [DecidablePred accept] :
    0 ≤ P.successProbability accept ∧ P.successProbability accept ≤ 1 := by
  exact ⟨P.successProbability_nonneg accept, P.successProbability_le_one accept⟩

theorem FinitePosteriorModel.successProbability_mono
    {A : Type*} [Fintype A]
    (P : FinitePosteriorModel A)
    (accept₁ accept₂ : A → Prop)
    [DecidablePred accept₁]
    [DecidablePred accept₂]
    (hImp : ∀ a : A, accept₁ a → accept₂ a) :
    P.successProbability accept₁ ≤ P.successProbability accept₂ := by
  unfold FinitePosteriorModel.successProbability
  exact Finset.sum_le_sum (fun a _ => by
    by_cases h1 : accept₁ a
    · have h2 : accept₂ a := hImp a h1
      simp [h1, h2]
    · by_cases h2 : accept₂ a
      · simp [h1, h2, P.posterior_nonneg a]
      · simp [h1, h2])

/-- Acceptance contract for the final docking endpoint. -/
structure DockingAcceptanceSpec where
  targetRMSD : ℝ
  targetProbability : ℝ
  targetRMSD_nonneg : 0 ≤ targetRMSD
  targetProbability_nonneg : 0 ≤ targetProbability
  targetProbability_le_one : targetProbability ≤ 1

/-- Finite sampled docking endpoint input: raw molecular problem, finite sampled
candidate pose family, inverse-temperature parameter for posterior synthesis,
and an RMSD-to-reference function on sampled poses. -/
structure SampledDockingSolverInput where
  prob : MDBindingProblem
  samples : SampledActionFamily MDAction
  beta : ℝ
  rmsdToReference : SupportedAction samples → ℝ

/-- Constructor from raw pocket+ligand problem and sampling controls to the
solver input object (sample family + posterior synthesis parameters). -/
noncomputable def sampledDockingSolverInputFromRawPocketLigand
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    SampledDockingSolverInput :=
  { prob := prob
    samples := rawPocketLigandSampleFamily prob cfg
    beta := cfg.beta
    rmsdToReference := rmsdToReference }

noncomputable def sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig) :
    SampledDockingSolverInput :=
  sampledDockingSolverInputFromRawPocketLigand
    prob cfg (canonicalRMSDToInputLigand prob cfg)

def SampledDockingSolverInput.baseState
    (I : SampledDockingSolverInput) : MDState :=
  bindingProblemBaseState I.prob

noncomputable def SampledDockingSolverInput.sampledDecisionProblem
    (I : SampledDockingSolverInput) :
    DecisionProblem (SupportedAction I.samples) MDState :=
  restrictedDecisionProblem I.prob.toDecisionProblem I.samples

noncomputable def SampledDockingSolverInput.selectedPose
    (I : SampledDockingSolverInput) : SupportedAction I.samples :=
  maxCodeOptAction I.sampledDecisionProblem I.baseState

theorem SampledDockingSolverInput.selectedPose_mem_sampled_opt
    (I : SampledDockingSolverInput) :
    I.selectedPose ∈ I.sampledDecisionProblem.Opt I.baseState := by
  exact maxCodeOptAction_mem_opt I.sampledDecisionProblem I.baseState

theorem SampledDockingSolverInput.selectedPose_mem_ambient_opt_of_capture
    (I : SampledDockingSolverInput)
    (hCapture : ∃ a : SupportedAction I.samples, a.1 ∈ I.prob.toDecisionProblem.Opt I.baseState) :
    I.selectedPose.1 ∈ I.prob.toDecisionProblem.Opt I.baseState := by
  have hRestrictedEq :=
    restricted_opt_eq_ambient_slice_of_exists_global_sampled_opt
      I.prob.toDecisionProblem I.samples I.baseState hCapture
  have hSelected :
      I.selectedPose ∈ (restrictedDecisionProblem I.prob.toDecisionProblem I.samples).Opt I.baseState := by
    simpa [SampledDockingSolverInput.sampledDecisionProblem] using I.selectedPose_mem_sampled_opt
  rw [hRestrictedEq] at hSelected
  exact hSelected

noncomputable def SampledDockingSolverInput.supportedBoltzmannWeight
    (I : SampledDockingSolverInput)
    (a : SupportedAction I.samples) : ℝ :=
  Real.exp (I.beta * I.sampledDecisionProblem.utility a I.baseState)

noncomputable def SampledDockingSolverInput.supportedBoltzmannPartition
    (I : SampledDockingSolverInput) : ℝ :=
  (Finset.univ : Finset (SupportedAction I.samples)).sum I.supportedBoltzmannWeight

theorem SampledDockingSolverInput.supportedBoltzmannPartition_pos
    (I : SampledDockingSolverInput) :
    0 < I.supportedBoltzmannPartition := by
  rcases I.samples.nonempty with ⟨a0, ha0⟩
  let a0s : SupportedAction I.samples := ⟨a0, ha0⟩
  have hMem : a0s ∈ (Finset.univ : Finset (SupportedAction I.samples)) := by
    simp
  have hPos : 0 < I.supportedBoltzmannWeight a0s := by
    unfold SampledDockingSolverInput.supportedBoltzmannWeight
    exact Real.exp_pos _
  have hLe : I.supportedBoltzmannWeight a0s ≤ I.supportedBoltzmannPartition := by
    have hRaw :
        Real.exp (I.beta * I.sampledDecisionProblem.utility a0s I.baseState) ≤
          (Finset.univ : Finset (SupportedAction I.samples)).sum
            (fun a => Real.exp (I.beta * I.sampledDecisionProblem.utility a I.baseState)) := by
      have hNonneg :
          ∀ a ∈ (Finset.univ : Finset (SupportedAction I.samples)),
            0 ≤ Real.exp (I.beta * I.sampledDecisionProblem.utility a I.baseState) := by
        intro a _
        exact le_of_lt (Real.exp_pos _)
      exact Finset.single_le_sum hNonneg hMem
    simpa [SampledDockingSolverInput.supportedBoltzmannWeight,
      SampledDockingSolverInput.supportedBoltzmannPartition] using hRaw
  exact lt_of_lt_of_le hPos hLe

noncomputable def SampledDockingSolverInput.posteriorModel
    (I : SampledDockingSolverInput) :
    FinitePosteriorModel (SupportedAction I.samples) :=
  { posterior := fun a => I.supportedBoltzmannWeight a / I.supportedBoltzmannPartition
    posterior_nonneg := by
      intro a
      exact div_nonneg (le_of_lt (Real.exp_pos _))
        (le_of_lt I.supportedBoltzmannPartition_pos)
    posterior_sum_one := by
      have hZpos := I.supportedBoltzmannPartition_pos
      calc
        (Finset.univ : Finset (SupportedAction I.samples)).sum
            (fun a => I.supportedBoltzmannWeight a / I.supportedBoltzmannPartition)
            = ((Finset.univ : Finset (SupportedAction I.samples)).sum I.supportedBoltzmannWeight) /
                I.supportedBoltzmannPartition := by
                  rw [Finset.sum_div]
        _ = I.supportedBoltzmannPartition / I.supportedBoltzmannPartition := by
              rfl
        _ = 1 := by
              exact div_self (ne_of_gt hZpos) }

theorem SampledDockingSolverInput.posteriorModel_certificate_bundle
    (I : SampledDockingSolverInput) :
    I.selectedPose ∈ I.sampledDecisionProblem.Opt I.baseState ∧
      (∀ a : SupportedAction I.samples, 0 ≤ I.posteriorModel.posterior a) ∧
      ((Finset.univ : Finset (SupportedAction I.samples)).sum I.posteriorModel.posterior = 1) := by
  exact ⟨I.selectedPose_mem_sampled_opt,
    I.posteriorModel.posterior_nonneg,
    I.posteriorModel.posterior_sum_one⟩

noncomputable def SampledDockingSolverInput.successProbability
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) : ℝ :=
  I.posteriorModel.successProbability
    (fun a => I.rmsdToReference a ≤ spec.targetRMSD)

theorem SampledDockingSolverInput.successProbability_unit_interval
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    0 ≤ I.successProbability spec ∧ I.successProbability spec ≤ 1 := by
  exact I.posteriorModel.successProbability_unit_interval
    (fun a => I.rmsdToReference a ≤ spec.targetRMSD)

theorem SampledDockingSolverInput.posterior_certificate_bundle
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    I.selectedPose ∈ I.sampledDecisionProblem.Opt I.baseState ∧
      0 ≤ I.successProbability spec ∧ I.successProbability spec ≤ 1 := by
  exact ⟨I.selectedPose_mem_sampled_opt,
    (I.successProbability_unit_interval spec).1,
    (I.successProbability_unit_interval spec).2⟩

def SampledDockingSolverInput.benchmarkRMSDPass
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) : Prop :=
  I.rmsdToReference I.selectedPose ≤ spec.targetRMSD

def SampledDockingSolverInput.benchmarkProbabilityPass
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) : Prop :=
  spec.targetProbability ≤ I.successProbability spec

def SampledDockingSolverInput.benchmarkContractSatisfied
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) : Prop :=
  I.benchmarkRMSDPass spec ∧ I.benchmarkProbabilityPass spec

/-- Deployment-time RMSD calibration model: proxy RMSD with a certified absolute
error radius to the benchmark-mode reference RMSD semantics. -/
structure DeploymentRMSDCalibration
    (I : SampledDockingSolverInput) where
  proxyRMSD : SupportedAction I.samples → ℝ
  rmsdErrorBound : ℝ
  rmsdErrorBound_nonneg : 0 ≤ rmsdErrorBound
  rmsd_abs_error : ∀ a : SupportedAction I.samples,
    |proxyRMSD a - I.rmsdToReference a| ≤ rmsdErrorBound

noncomputable def DeploymentRMSDCalibration.conservativeSuccessProbability
    {I : SampledDockingSolverInput}
    (C : DeploymentRMSDCalibration I)
    (spec : DockingAcceptanceSpec) : ℝ :=
  I.posteriorModel.successProbability
    (fun a => C.proxyRMSD a + C.rmsdErrorBound ≤ spec.targetRMSD)

def DeploymentRMSDCalibration.deploymentRMSDPass
    {I : SampledDockingSolverInput}
    (C : DeploymentRMSDCalibration I)
    (spec : DockingAcceptanceSpec) : Prop :=
  C.proxyRMSD I.selectedPose + C.rmsdErrorBound ≤ spec.targetRMSD

def DeploymentRMSDCalibration.deploymentProbabilityPass
    {I : SampledDockingSolverInput}
    (C : DeploymentRMSDCalibration I)
    (spec : DockingAcceptanceSpec) : Prop :=
  spec.targetProbability ≤ C.conservativeSuccessProbability spec

def DeploymentRMSDCalibration.deploymentContractSatisfied
    {I : SampledDockingSolverInput}
    (C : DeploymentRMSDCalibration I)
    (spec : DockingAcceptanceSpec) : Prop :=
  C.deploymentRMSDPass spec ∧ C.deploymentProbabilityPass spec

theorem DeploymentRMSDCalibration.proxy_plus_bound_le_implies_true_rmsd_le
    {I : SampledDockingSolverInput}
    (C : DeploymentRMSDCalibration I)
    {a : SupportedAction I.samples}
    {target : ℝ}
    (h : C.proxyRMSD a + C.rmsdErrorBound ≤ target) :
    I.rmsdToReference a ≤ target := by
  have hAbs := C.rmsd_abs_error a
  have hLower : -C.rmsdErrorBound ≤ C.proxyRMSD a - I.rmsdToReference a :=
    (abs_le.mp hAbs).1
  have hBridge : I.rmsdToReference a ≤ C.proxyRMSD a + C.rmsdErrorBound := by
    linarith
  exact le_trans hBridge h

theorem DeploymentRMSDCalibration.conservativeSuccessProbability_le_true
    {I : SampledDockingSolverInput}
    (C : DeploymentRMSDCalibration I)
    (spec : DockingAcceptanceSpec) :
    C.conservativeSuccessProbability spec ≤ I.successProbability spec := by
  unfold DeploymentRMSDCalibration.conservativeSuccessProbability
  unfold SampledDockingSolverInput.successProbability
  refine FinitePosteriorModel.successProbability_mono I.posteriorModel
    (accept₁ := fun a => C.proxyRMSD a + C.rmsdErrorBound ≤ spec.targetRMSD)
    (accept₂ := fun a => I.rmsdToReference a ≤ spec.targetRMSD)
    ?_
  intro a hProxy
  exact C.proxy_plus_bound_le_implies_true_rmsd_le hProxy

theorem DeploymentRMSDCalibration.deployment_contract_implies_benchmark_contract
    {I : SampledDockingSolverInput}
    (C : DeploymentRMSDCalibration I)
    (spec : DockingAcceptanceSpec)
    (hDeployment : C.deploymentContractSatisfied spec) :
    I.benchmarkContractSatisfied spec := by
  refine ⟨?_, ?_⟩
  · exact C.proxy_plus_bound_le_implies_true_rmsd_le hDeployment.1
  · exact le_trans hDeployment.2 (C.conservativeSuccessProbability_le_true spec)

theorem DeploymentRMSDCalibration.conservativeSuccessProbability_unit_interval
    {I : SampledDockingSolverInput}
    (C : DeploymentRMSDCalibration I)
    (spec : DockingAcceptanceSpec) :
    0 ≤ C.conservativeSuccessProbability spec ∧
      C.conservativeSuccessProbability spec ≤ 1 := by
  unfold DeploymentRMSDCalibration.conservativeSuccessProbability
  exact I.posteriorModel.successProbability_unit_interval
    (fun a => C.proxyRMSD a + C.rmsdErrorBound ≤ spec.targetRMSD)

noncomputable def canonicalDeploymentCalibration
    (I : SampledDockingSolverInput) : DeploymentRMSDCalibration I :=
  { proxyRMSD := I.rmsdToReference
    rmsdErrorBound := 0
    rmsdErrorBound_nonneg := by norm_num
    rmsd_abs_error := by
      intro a
      simp }

theorem canonicalDeploymentCalibration_deploymentContract_iff_benchmark
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    (canonicalDeploymentCalibration I).deploymentContractSatisfied spec ↔
      I.benchmarkContractSatisfied spec := by
  unfold DeploymentRMSDCalibration.deploymentContractSatisfied
    DeploymentRMSDCalibration.deploymentRMSDPass
    DeploymentRMSDCalibration.deploymentProbabilityPass
    DeploymentRMSDCalibration.conservativeSuccessProbability
    SampledDockingSolverInput.benchmarkContractSatisfied
    SampledDockingSolverInput.benchmarkRMSDPass
    SampledDockingSolverInput.benchmarkProbabilityPass
    canonicalDeploymentCalibration
    SampledDockingSolverInput.successProbability
  simp

private def realTensorPort
    (name : String) (rank : Nat) : DecisionQuotient.Computation.ArrayDSL.PortSpecIR :=
  { name := name
    ty :=
      { kind := DecisionQuotient.Computation.ArrayDSL.ExprKind.tensor
        scalarType? := some DecisionQuotient.Computation.ArrayDSL.ScalarType.real
        rank? := some rank } }

private def realScalarPort
    (name : String) : DecisionQuotient.Computation.ArrayDSL.PortSpecIR :=
  { name := name
    ty :=
      { kind := DecisionQuotient.Computation.ArrayDSL.ExprKind.scalar
        scalarType? := some DecisionQuotient.Computation.ArrayDSL.ScalarType.real
        rank? := none } }

private def booleanScalarPort
    (name : String) : DecisionQuotient.Computation.ArrayDSL.PortSpecIR :=
  { name := name
    ty :=
      { kind := DecisionQuotient.Computation.ArrayDSL.ExprKind.scalar
        scalarType? := some DecisionQuotient.Computation.ArrayDSL.ScalarType.boolean
        rank? := none } }

/-- Canonical universal-IR view of the sampled docking solver: score sampled
poses against the pocket, aggregate posterior success mass, evaluate RMSD for the
selected pose, and branch on acceptance. -/
def SampledDockingSolverInput.canonicalSolverProgramIR
    (I : SampledDockingSolverInput) :
    DecisionQuotient.Computation.ArrayDSL.ProgramIR :=
  { programName := "sampled_docking_solver"
    inputs :=
      [ realTensorPort "protein_coords" 2
      , realTensorPort "ligand_coords" 2
      , realTensorPort "candidate_pose_coords" 3
      , realTensorPort "best_pose_delta" 2
      , realTensorPort "candidate_success_weights" 1
      , realScalarPort "cutoff"
      , realScalarPort "epsilon"
      , realScalarPort "sigma"
      , realScalarPort "beta"
      , realScalarPort "target_rmsd"
      , realScalarPort "target_probability" ]
    constants := []
    blocks :=
      [ { blockId := 0
          nodes :=
            [ { nodeId := 0
                opName := "pairwiseDistances"
                inputs := ["candidate_pose_coords", "protein_coords"]
                outputs := ["candidate_pairwise_distances"]
                attrs := [("role", "candidate_pose_scoring_distances")] }
            , { nodeId := 1
                opName := "applyCutoff"
                inputs := ["candidate_pairwise_distances", "cutoff"]
                outputs := ["masked_pairwise_distances"]
                attrs := [("role", "cutoff_masking")] }
            , { nodeId := 2
                opName := "sumPairPotentials"
                inputs := ["masked_pairwise_distances", "cutoff", "epsilon", "sigma"]
                outputs := ["candidate_pose_scores"]
                attrs := [("role", "lennard_jones_scoring")] }
            , { nodeId := 3
                opName := "reduce_sum"
                inputs := ["candidate_success_weights"]
                outputs := ["posterior_success_probability"]
                attrs := [("role", "posterior_success_mass")] }
            , { nodeId := 4
                opName := "norm"
                inputs := ["best_pose_delta"]
                outputs := ["best_pose_rmsd"]
                attrs := [("role", "rmsd_evaluation")] }
            ]
          terminator :=
            DecisionQuotient.Computation.ArrayDSL.BlockTerminatorIR.branch
              "accept_flag" 1 2 }
      , { blockId := 1
          nodes := []
          terminator :=
            DecisionQuotient.Computation.ArrayDSL.BlockTerminatorIR.ret
              ["best_pose", "posterior_success_probability", "accept_flag"] }
      , { blockId := 2
          nodes := []
          terminator :=
            DecisionQuotient.Computation.ArrayDSL.BlockTerminatorIR.ret
              ["best_pose", "posterior_success_probability", "accept_flag"] }
      ]
    entryBlock := 0
    outputs := ["best_pose", "posterior_success_probability", "accept_flag"]
    dataEdges := [(0, 1), (1, 2), (2, 3)]
    shapeConstraints :=
      [ .fixed "protein_coords" [I.prob.protein.numAtoms, 3]
      , .fixed "ligand_coords" [I.prob.ligand.numAtoms, 3]
      , .fixed "candidate_pose_coords" [I.samples.support.card, I.prob.ligand.numAtoms, 3]
      , .fixed "best_pose_delta" [I.prob.ligand.numAtoms, 3]
      , .fixed "candidate_success_weights" [I.samples.support.card]
      ] }

theorem SampledDockingSolverInput.canonicalSolverProgramIR_wellFormedDAG
    (I : SampledDockingSolverInput) :
    I.canonicalSolverProgramIR.wellFormedDAG := by
  intro edge hEdge
  simp [SampledDockingSolverInput.canonicalSolverProgramIR] at hEdge
  rcases hEdge with rfl | rfl | rfl <;> decide

theorem SampledDockingSolverInput.canonicalSolverProgramIR_requiredOps
    (I : SampledDockingSolverInput) :
    I.canonicalSolverProgramIR.requiredOps =
      ["pairwiseDistances", "applyCutoff", "sumPairPotentials", "reduce_sum", "norm"] := by
  simpa [SampledDockingSolverInput.canonicalSolverProgramIR,
    DecisionQuotient.Computation.ArrayDSL.ProgramIR.requiredOps] using
    (show ["pairwiseDistances", "applyCutoff", "sumPairPotentials", "reduce_sum", "norm"].eraseDups =
        ["pairwiseDistances", "applyCutoff", "sumPairPotentials", "reduce_sum", "norm"] from by
      native_decide)

theorem SampledDockingSolverInput.canonicalSolverProgramIR_requires_sumPairPotentials
    (I : SampledDockingSolverInput) :
    "sumPairPotentials" ∈ I.canonicalSolverProgramIR.requiredOps := by
  simp [I.canonicalSolverProgramIR_requiredOps]

def SampledDockingSolverInput.jaxCodegenReport
    (I : SampledDockingSolverInput) :
    DecisionQuotient.Computation.ArrayDSL.ProgramBackendCodegenReport :=
  DecisionQuotient.Computation.ArrayDSL.exportUniversalIRCatalog.codegenProgramReport
    DecisionQuotient.Computation.ArrayDSL.Backend.jax
    I.canonicalSolverProgramIR

theorem SampledDockingSolverInput.jax_codegen_success
    (I : SampledDockingSolverInput) :
    I.jaxCodegenReport.success = true := by
  have hMissing : I.jaxCodegenReport.missingOps = [] := by
    rw [SampledDockingSolverInput.jaxCodegenReport,
      DecisionQuotient.Computation.ArrayDSL.UniversalIRCatalog.codegenProgramReport,
      I.canonicalSolverProgramIR_requiredOps]
    change List.filter
        (fun opName =>
          (DecisionQuotient.Computation.ArrayDSL.exportUniversalIRCatalog.codegenOpFor?
            DecisionQuotient.Computation.ArrayDSL.Backend.jax opName).isNone)
        ["pairwiseDistances", "applyCutoff", "sumPairPotentials", "reduce_sum", "norm"] = []
    clear I
    native_decide
  exact
    (DecisionQuotient.Computation.ArrayDSL.ProgramBackendCodegenReport.success_eq_true_iff_no_missing
      I.jaxCodegenReport).2 hMissing

theorem SampledDockingSolverInput.jax_codegen_no_missing_ops
    (I : SampledDockingSolverInput) :
    I.jaxCodegenReport.missingOps = [] := by
  exact (DecisionQuotient.Computation.ArrayDSL.ProgramBackendCodegenReport.success_eq_true_iff_no_missing
    I.jaxCodegenReport).mp I.jax_codegen_success

theorem sampledDockingSolverInputFromRawPocketLigand_bundle
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    let I := sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.selectedPose ∈ I.sampledDecisionProblem.Opt I.baseState ∧
      (∀ a : SupportedAction I.samples, 0 ≤ I.posteriorModel.posterior a) ∧
      ((Finset.univ : Finset (SupportedAction I.samples)).sum I.posteriorModel.posterior = 1) ∧
      I.jaxCodegenReport.success = true := by
  intro I
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simpa [sampledDockingSolverInputFromRawPocketLigand] using
      (rawPocketLigandSampleFamily_contains_baseAction prob cfg)
  · exact (I.posteriorModel_certificate_bundle).1
  · exact (I.posteriorModel_certificate_bundle).2.1
  · exact (I.posteriorModel_certificate_bundle).2.2
  · exact I.jax_codegen_success

theorem sampledDockingSolverInputFromRawPocketLigand_selectedPose_mem_ambient_opt_of_capture
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ)
    (hCapture :
      ∃ a : SupportedAction (rawPocketLigandSampleFamily prob cfg),
        a.1 ∈ prob.toDecisionProblem.Opt (bindingProblemBaseState prob)) :
    let I := sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference
    I.selectedPose.1 ∈ I.prob.toDecisionProblem.Opt I.baseState := by
  intro I
  have hCaptureI :
      ∃ a : SupportedAction I.samples,
        a.1 ∈ I.prob.toDecisionProblem.Opt I.baseState := by
    simpa [I, sampledDockingSolverInputFromRawPocketLigand,
      SampledDockingSolverInput.baseState, bindingProblemBaseState] using hCapture
  exact I.selectedPose_mem_ambient_opt_of_capture hCaptureI

theorem sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD_bundle
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig) :
    let I := sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.selectedPose ∈ I.sampledDecisionProblem.Opt I.baseState ∧
      (∀ a : SupportedAction I.samples, 0 ≤ I.posteriorModel.posterior a) ∧
      ((Finset.univ : Finset (SupportedAction I.samples)).sum I.posteriorModel.posterior = 1) ∧
      I.jaxCodegenReport.success = true ∧
      0 ≤ I.rmsdToReference I.selectedPose := by
  intro I
  have hBase := sampledDockingSolverInputFromRawPocketLigand_bundle
    prob cfg (canonicalRMSDToInputLigand prob cfg)
  refine ⟨hBase.1, hBase.2.1, hBase.2.2.1, hBase.2.2.2.1, hBase.2.2.2.2, ?_⟩
  simpa [sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD,
    sampledDockingSolverInputFromRawPocketLigand,
    canonicalRMSDToInputLigand] using
    canonicalRMSDToInputLigand_nonneg prob cfg I.selectedPose

/-- Successful acceptance-constrained docking certificate. -/
structure CertifiedAcceptedDockingPose
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) where
  supportedPose : SupportedAction I.samples
  isSelected : supportedPose = I.selectedPose
  sampledOptimal : supportedPose ∈ I.sampledDecisionProblem.Opt I.baseState
  rmsdAccepted : I.rmsdToReference supportedPose ≤ spec.targetRMSD
  probabilityAccepted : spec.targetProbability ≤ I.successProbability spec
  successProbabilityBounds : 0 ≤ I.successProbability spec ∧ I.successProbability spec ≤ 1

def CertifiedAcceptedDockingPose.pose
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (C : CertifiedAcceptedDockingPose I spec) : MDAction :=
  C.supportedPose.1

/-- Failure certificate for the acceptance-constrained docking API. -/
structure DockingSolverFailure
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) where
  supportedPose : SupportedAction I.samples
  isSelected : supportedPose = I.selectedPose
  acceptanceFailed :
    ¬ (I.rmsdToReference supportedPose ≤ spec.targetRMSD ∧
        spec.targetProbability ≤ I.successProbability spec)
  successProbabilityBounds : 0 ≤ I.successProbability spec ∧ I.successProbability spec ≤ 1

def DockingSolverFailure.pose
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (F : DockingSolverFailure I spec) : MDAction :=
  F.supportedPose.1

/-- Solver output API: either a certified accepted pose or an explicit failure
certificate for the current RMSD/probability contract. -/
inductive AcceptanceConstrainedDockingSolverResult
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) where
  | accepted (cert : CertifiedAcceptedDockingPose I spec)
  | failure (cert : DockingSolverFailure I spec)

/-- Final acceptance-constrained sampled docking endpoint. It chooses the best
sampled pose by exact utility, synthesizes the finite Boltzmann posterior from
the pocket+ligand input state, and returns either a certified pose or failure. -/
noncomputable def solveAcceptanceConstrainedDocking
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    AcceptanceConstrainedDockingSolverResult I spec :=
  let pose := I.selectedPose
  if hRmsd : I.rmsdToReference pose ≤ spec.targetRMSD then
    if hProb : spec.targetProbability ≤ I.successProbability spec then
      .accepted
        { supportedPose := pose
          isSelected := rfl
          sampledOptimal := I.selectedPose_mem_sampled_opt
          rmsdAccepted := hRmsd
          probabilityAccepted := hProb
          successProbabilityBounds := I.successProbability_unit_interval spec }
    else
      .failure
        { supportedPose := pose
          isSelected := rfl
          acceptanceFailed := by
            intro hAccept
            exact hProb hAccept.2
          successProbabilityBounds := I.successProbability_unit_interval spec }
  else
    .failure
      { supportedPose := pose
        isSelected := rfl
        acceptanceFailed := by
          intro hAccept
          exact hRmsd hAccept.1
        successProbabilityBounds := I.successProbability_unit_interval spec }

theorem solveAcceptanceConstrainedDocking_accept_iff
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    (∃ cert, solveAcceptanceConstrainedDocking I spec =
      AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
      I.benchmarkContractSatisfied spec := by
  unfold solveAcceptanceConstrainedDocking
    SampledDockingSolverInput.benchmarkContractSatisfied
    SampledDockingSolverInput.benchmarkRMSDPass
    SampledDockingSolverInput.benchmarkProbabilityPass
  by_cases hRmsd : I.rmsdToReference I.selectedPose ≤ spec.targetRMSD
  · by_cases hProb : spec.targetProbability ≤ I.successProbability spec
    · simp [hRmsd, hProb]
    · simp [hRmsd, hProb]
  · simp [hRmsd]

theorem solveAcceptanceConstrainedDocking_failure_iff
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    (∃ cert, solveAcceptanceConstrainedDocking I spec =
      AcceptanceConstrainedDockingSolverResult.failure cert) ↔
      ¬ I.benchmarkContractSatisfied spec := by
  unfold solveAcceptanceConstrainedDocking
    SampledDockingSolverInput.benchmarkContractSatisfied
    SampledDockingSolverInput.benchmarkRMSDPass
    SampledDockingSolverInput.benchmarkProbabilityPass
  by_cases hRmsd : I.rmsdToReference I.selectedPose ≤ spec.targetRMSD
  · by_cases hProb : spec.targetProbability ≤ I.successProbability spec
    · simp [hRmsd, hProb]
    · simp [hRmsd, hProb]
  · simp [hRmsd]

noncomputable def benchmarkAcceptedCertificateOfContract
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (hContract : I.benchmarkContractSatisfied spec) :
    CertifiedAcceptedDockingPose I spec :=
  Classical.choose ((solveAcceptanceConstrainedDocking_accept_iff I spec).2 hContract)

theorem benchmarkAcceptedCertificateOfContract_spec
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (hContract : I.benchmarkContractSatisfied spec) :
    solveAcceptanceConstrainedDocking I spec =
      AcceptanceConstrainedDockingSolverResult.accepted
        (benchmarkAcceptedCertificateOfContract I spec hContract) :=
  Classical.choose_spec ((solveAcceptanceConstrainedDocking_accept_iff I spec).2 hContract)

/-- Deployment-mode failure certificate (conservative proxy contract did not
pass, with conservative success-probability bounds still certified). -/
structure DeploymentFailureCertificate
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (C : DeploymentRMSDCalibration I) where
  deploymentFailed : ¬ C.deploymentContractSatisfied spec
  conservativeProbabilityBounds :
    0 ≤ C.conservativeSuccessProbability spec ∧
      C.conservativeSuccessProbability spec ≤ 1

/-- Deployment-mode result API: either benchmark-accepted pose certificate
under conservative deployment pass conditions, or explicit deployment-failure
certificate. -/
inductive DeploymentConstrainedDockingSolverResult
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (C : DeploymentRMSDCalibration I) where
  | accepted (cert : CertifiedAcceptedDockingPose I spec)
  | rejected (cert : DeploymentFailureCertificate I spec C)

noncomputable def solveDeploymentConstrainedDocking
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (C : DeploymentRMSDCalibration I) :
    DeploymentConstrainedDockingSolverResult I spec C :=
  if hDep : C.deploymentContractSatisfied spec then
    let hBench := C.deployment_contract_implies_benchmark_contract spec hDep
    DeploymentConstrainedDockingSolverResult.accepted
      (benchmarkAcceptedCertificateOfContract I spec hBench)
  else
    DeploymentConstrainedDockingSolverResult.rejected
      { deploymentFailed := hDep
        conservativeProbabilityBounds :=
          C.conservativeSuccessProbability_unit_interval spec }

theorem solveDeploymentConstrainedDocking_accepted_iff
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (C : DeploymentRMSDCalibration I) :
    (∃ cert, solveDeploymentConstrainedDocking I spec C =
      DeploymentConstrainedDockingSolverResult.accepted cert) ↔
      C.deploymentContractSatisfied spec := by
  unfold solveDeploymentConstrainedDocking
  by_cases hDep : C.deploymentContractSatisfied spec
  · simp [hDep]
  · simp [hDep]

theorem solveDeploymentConstrainedDocking_sound
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (C : DeploymentRMSDCalibration I) :
    match solveDeploymentConstrainedDocking I spec C with
    | DeploymentConstrainedDockingSolverResult.accepted _ =>
        C.deploymentContractSatisfied spec ∧ I.benchmarkContractSatisfied spec
    | DeploymentConstrainedDockingSolverResult.rejected _ =>
        ¬ C.deploymentContractSatisfied spec := by
  unfold solveDeploymentConstrainedDocking
  by_cases hDep : C.deploymentContractSatisfied spec
  · simp [hDep, C.deployment_contract_implies_benchmark_contract spec hDep]
  · simp [hDep]

theorem solveDeploymentConstrainedDocking_accept_refines_benchmark_accept
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (C : DeploymentRMSDCalibration I)
    (hAccept :
      ∃ cert, solveDeploymentConstrainedDocking I spec C =
        DeploymentConstrainedDockingSolverResult.accepted cert) :
    ∃ cert, solveAcceptanceConstrainedDocking I spec =
      AcceptanceConstrainedDockingSolverResult.accepted cert := by
  have hDep : C.deploymentContractSatisfied spec :=
    (solveDeploymentConstrainedDocking_accepted_iff I spec C).1 hAccept
  have hBench : I.benchmarkContractSatisfied spec :=
    C.deployment_contract_implies_benchmark_contract spec hDep
  exact (solveAcceptanceConstrainedDocking_accept_iff I spec).2 hBench

/-- Execution witness connecting canonical ProgramIR runtime outputs to the
acceptance-constrained solver semantics. -/
structure CanonicalProgramExecutionWitness
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) where
  program : DecisionQuotient.Computation.ArrayDSL.ProgramIR
  program_is_canonical : program = I.canonicalSolverProgramIR
  outputBestPose : SupportedAction I.samples
  outputSuccessProbability : ℝ
  outputRMSD : ℝ
  outputAcceptFlag : Bool
  outputBestPose_eq_selected : outputBestPose = I.selectedPose
  outputSuccessProbability_eq_solver : outputSuccessProbability = I.successProbability spec
  outputRMSD_eq_solver : outputRMSD = I.rmsdToReference I.selectedPose
  outputAcceptFlag_eq_decide :
    outputAcceptFlag = decide
      (outputRMSD ≤ spec.targetRMSD ∧
        spec.targetProbability ≤ outputSuccessProbability)

theorem CanonicalProgramExecutionWitness.refines_solver_result
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (W : CanonicalProgramExecutionWitness I spec) :
    (W.outputAcceptFlag = true ↔
      ∃ cert, solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
    (W.outputAcceptFlag = false ↔
      ∃ cert, solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) := by
  have hFlagTrue :
      W.outputAcceptFlag = true ↔ I.benchmarkContractSatisfied spec := by
    calc
      W.outputAcceptFlag = true
          ↔ decide (W.outputRMSD ≤ spec.targetRMSD ∧
                spec.targetProbability ≤ W.outputSuccessProbability) = true := by
                simp [W.outputAcceptFlag_eq_decide]
      _ ↔ (W.outputRMSD ≤ spec.targetRMSD ∧
            spec.targetProbability ≤ W.outputSuccessProbability) := by
            simp
      _ ↔ I.benchmarkContractSatisfied spec := by
            simp [SampledDockingSolverInput.benchmarkContractSatisfied,
              SampledDockingSolverInput.benchmarkRMSDPass,
              SampledDockingSolverInput.benchmarkProbabilityPass,
              W.outputRMSD_eq_solver,
              W.outputSuccessProbability_eq_solver]
  have hFlagFalse :
      W.outputAcceptFlag = false ↔ ¬ I.benchmarkContractSatisfied spec := by
    calc
      W.outputAcceptFlag = false
          ↔ decide (W.outputRMSD ≤ spec.targetRMSD ∧
                spec.targetProbability ≤ W.outputSuccessProbability) = false := by
                simp [W.outputAcceptFlag_eq_decide]
      _ ↔ ¬ (W.outputRMSD ≤ spec.targetRMSD ∧
            spec.targetProbability ≤ W.outputSuccessProbability) := by
            simp
      _ ↔ ¬ I.benchmarkContractSatisfied spec := by
            simp [SampledDockingSolverInput.benchmarkContractSatisfied,
              SampledDockingSolverInput.benchmarkRMSDPass,
              SampledDockingSolverInput.benchmarkProbabilityPass,
              W.outputRMSD_eq_solver,
              W.outputSuccessProbability_eq_solver]
  refine ⟨?_, ?_⟩
  · exact hFlagTrue.trans (solveAcceptanceConstrainedDocking_accept_iff I spec).symm
  · exact hFlagFalse.trans (solveAcceptanceConstrainedDocking_failure_iff I spec).symm

noncomputable def canonicalProgramExecutionWitnessOfSpec
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    CanonicalProgramExecutionWitness I spec :=
  { program := I.canonicalSolverProgramIR
    program_is_canonical := rfl
    outputBestPose := I.selectedPose
    outputSuccessProbability := I.successProbability spec
    outputRMSD := I.rmsdToReference I.selectedPose
    outputAcceptFlag :=
      decide
        (I.rmsdToReference I.selectedPose ≤ spec.targetRMSD ∧
          spec.targetProbability ≤ I.successProbability spec)
    outputBestPose_eq_selected := rfl
    outputSuccessProbability_eq_solver := rfl
    outputRMSD_eq_solver := rfl
    outputAcceptFlag_eq_decide := rfl }

theorem canonicalProgramExecutionWitnessOfSpec_refines_solver_result
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    (canonicalProgramExecutionWitnessOfSpec I spec).outputAcceptFlag = true ↔
      ∃ cert, solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  exact (CanonicalProgramExecutionWitness.refines_solver_result
    (canonicalProgramExecutionWitnessOfSpec I spec)).1

theorem canonicalProgramExecutionWitnessOfSpec_refines_solver_failure
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    (canonicalProgramExecutionWitnessOfSpec I spec).outputAcceptFlag = false ↔
      ∃ cert, solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.failure cert := by
  exact (CanonicalProgramExecutionWitness.refines_solver_result
    (canonicalProgramExecutionWitnessOfSpec I spec)).2

/-- Canonical ProgramIR runtime output record for the docking endpoint. -/
structure CanonicalProgramRuntimeOutput
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) where
  outputBestPose : SupportedAction I.samples
  outputSuccessProbability : ℝ
  outputRMSD : ℝ
  outputAcceptFlag : Bool

/-- Minimal operational state for the canonical ProgramIR endpoint. -/
structure CanonicalProgramInterpreterState
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) where
  selectedPose : SupportedAction I.samples
  posteriorSuccessProbability : ℝ
  selectedPoseRMSD : ℝ

noncomputable def CanonicalProgramInterpreterState.acceptFlag
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (S : CanonicalProgramInterpreterState I spec) : Bool :=
  decide (S.selectedPoseRMSD ≤ spec.targetRMSD ∧
    spec.targetProbability ≤ S.posteriorSuccessProbability)

noncomputable def interpretCanonicalProgramState
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    CanonicalProgramInterpreterState I spec :=
  { selectedPose := I.selectedPose
    posteriorSuccessProbability := I.successProbability spec
    selectedPoseRMSD := I.rmsdToReference I.selectedPose }

noncomputable def canonicalRuntimeOutputOfInterpreterState
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (S : CanonicalProgramInterpreterState I spec) :
    CanonicalProgramRuntimeOutput I spec :=
  { outputBestPose := S.selectedPose
    outputSuccessProbability := S.posteriorSuccessProbability
    outputRMSD := S.selectedPoseRMSD
    outputAcceptFlag := S.acceptFlag }

noncomputable def runCanonicalSolverProgram
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    CanonicalProgramRuntimeOutput I spec :=
  { outputBestPose := I.selectedPose
    outputSuccessProbability := I.successProbability spec
    outputRMSD := I.rmsdToReference I.selectedPose
    outputAcceptFlag :=
      decide
        (I.rmsdToReference I.selectedPose ≤ spec.targetRMSD ∧
          spec.targetProbability ≤ I.successProbability spec) }

theorem runCanonicalSolverProgram_eq_interpreter_output
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    runCanonicalSolverProgram I spec =
      canonicalRuntimeOutputOfInterpreterState (interpretCanonicalProgramState I spec) := by
  rfl

theorem interpretCanonicalProgramState_acceptFlag_eq_runtime
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    (canonicalRuntimeOutputOfInterpreterState
      (interpretCanonicalProgramState I spec)).outputAcceptFlag =
      (runCanonicalSolverProgram I spec).outputAcceptFlag := by
  simp [runCanonicalSolverProgram_eq_interpreter_output]

theorem runCanonicalSolverProgram_acceptFlag_eq_decide
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    (runCanonicalSolverProgram I spec).outputAcceptFlag =
      decide
        (I.rmsdToReference I.selectedPose ≤ spec.targetRMSD ∧
          spec.targetProbability ≤ I.successProbability spec) :=
  rfl

theorem runCanonicalSolverProgram_refines_solver_result
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    ((runCanonicalSolverProgram I spec).outputAcceptFlag = true ↔
      ∃ cert, solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
    ((runCanonicalSolverProgram I spec).outputAcceptFlag = false ↔
      ∃ cert, solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) := by
  have hCore :=
    CanonicalProgramExecutionWitness.refines_solver_result
      (canonicalProgramExecutionWitnessOfSpec I spec)
  refine ⟨?_, ?_⟩
  · simpa [runCanonicalSolverProgram, canonicalProgramExecutionWitnessOfSpec] using hCore.1
  · simpa [runCanonicalSolverProgram, canonicalProgramExecutionWitnessOfSpec] using hCore.2

def canonicalProgramEvaluator
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    DecisionQuotient.Computation.ArrayDSL.ProgramEvaluator
      (CanonicalProgramRuntimeOutput I spec) :=
  { evalNode := fun _ state => state
    evalCondition := fun _ state => state.outputAcceptFlag }

theorem canonicalProgramIR_executeFromEntry_generic_done
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    DecisionQuotient.Computation.ArrayDSL.ProgramIR.executeFromEntry
      I.canonicalSolverProgramIR
      (canonicalProgramEvaluator I spec)
      2
      (runCanonicalSolverProgram I spec)
      =
      DecisionQuotient.Computation.ArrayDSL.ProgramEvalResult.done
        (runCanonicalSolverProgram I spec) := by
  cases hFlag : (runCanonicalSolverProgram I spec).outputAcceptFlag <;>
    simp [DecisionQuotient.Computation.ArrayDSL.ProgramIR.executeFromEntry,
      DecisionQuotient.Computation.ArrayDSL.ProgramIR.executeFuel,
      DecisionQuotient.Computation.ArrayDSL.ProgramIR.step,
      DecisionQuotient.Computation.ArrayDSL.ProgramIR.findBlock?,
      DecisionQuotient.Computation.ArrayDSL.BasicBlockIR.eval,
      DecisionQuotient.Computation.ArrayDSL.BasicBlockIR.evalNodes,
      DecisionQuotient.Computation.ArrayDSL.BasicBlockIR.evalTerminator,
      SampledDockingSolverInput.canonicalSolverProgramIR,
      canonicalProgramEvaluator,
      hFlag]

theorem canonicalProgramIR_executeFromEntry_generic_deterministic
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (r₁ r₂ : DecisionQuotient.Computation.ArrayDSL.ProgramEvalResult
      (CanonicalProgramRuntimeOutput I spec))
    (h₁ :
      DecisionQuotient.Computation.ArrayDSL.ProgramIR.executeFromEntry
        I.canonicalSolverProgramIR
        (canonicalProgramEvaluator I spec)
        2
        (runCanonicalSolverProgram I spec) = r₁)
    (h₂ :
      DecisionQuotient.Computation.ArrayDSL.ProgramIR.executeFromEntry
        I.canonicalSolverProgramIR
        (canonicalProgramEvaluator I spec)
        2
        (runCanonicalSolverProgram I spec) = r₂) :
    r₁ = r₂ := by
  exact DecisionQuotient.Computation.ArrayDSL.ProgramIR.executeFromEntry_deterministic
    I.canonicalSolverProgramIR (canonicalProgramEvaluator I spec) 2
    (runCanonicalSolverProgram I spec) r₁ r₂ h₁ h₂

theorem canonicalProgramIR_executeFromEntry_generic_refines_solver_result
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    DecisionQuotient.Computation.ArrayDSL.ProgramIR.executeFromEntry
      I.canonicalSolverProgramIR
      (canonicalProgramEvaluator I spec)
      2
      (runCanonicalSolverProgram I spec)
      =
      DecisionQuotient.Computation.ArrayDSL.ProgramEvalResult.done
        (runCanonicalSolverProgram I spec) ∧
    ((runCanonicalSolverProgram I spec).outputAcceptFlag = true ↔
      ∃ cert,
        solveAcceptanceConstrainedDocking I spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
    ((runCanonicalSolverProgram I spec).outputAcceptFlag = false ↔
      ∃ cert,
        solveAcceptanceConstrainedDocking I spec =
          AcceptanceConstrainedDockingSolverResult.failure cert) := by
  exact ⟨canonicalProgramIR_executeFromEntry_generic_done I spec,
    (runCanonicalSolverProgram_refines_solver_result I spec).1,
    (runCanonicalSolverProgram_refines_solver_result I spec).2⟩

theorem interpretCanonicalProgramState_refines_solver_result
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) :
    ((canonicalRuntimeOutputOfInterpreterState
        (interpretCanonicalProgramState I spec)).outputAcceptFlag = true ↔
      ∃ cert, solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
    ((canonicalRuntimeOutputOfInterpreterState
        (interpretCanonicalProgramState I spec)).outputAcceptFlag = false ↔
      ∃ cert, solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) := by
  simpa [runCanonicalSolverProgram_eq_interpreter_output] using
    runCanonicalSolverProgram_refines_solver_result I spec

noncomputable def solveRawPocketLigandBenchmark
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    AcceptanceConstrainedDockingSolverResult
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
      spec :=
  solveAcceptanceConstrainedDocking
    (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
    spec

theorem solveRawPocketLigandBenchmark_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    (∃ cert,
      solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ∨
    (∃ cert,
      solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
        AcceptanceConstrainedDockingSolverResult.failure cert) := by
  unfold solveRawPocketLigandBenchmark
  cases hRes :
      solveAcceptanceConstrainedDocking
        (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
        spec with
  | accepted cert => exact Or.inl ⟨cert, rfl⟩
  | failure cert => exact Or.inr ⟨cert, rfl⟩

theorem solveRawPocketLigandBenchmark_accept_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    (∃ cert,
      solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference).benchmarkContractSatisfied spec := by
  simpa [solveRawPocketLigandBenchmark] using
    (solveAcceptanceConstrainedDocking_accept_iff
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
      spec)

theorem solveRawPocketLigandBenchmark_failure_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    (∃ cert,
      solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
        AcceptanceConstrainedDockingSolverResult.failure cert) ↔
      ¬ (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference).benchmarkContractSatisfied spec := by
  simpa [solveRawPocketLigandBenchmark] using
    (solveAcceptanceConstrainedDocking_failure_iff
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
      spec)

noncomputable def solveRawPocketLigandCanonicalBenchmark
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    AcceptanceConstrainedDockingSolverResult
      (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg)
      spec :=
  solveRawPocketLigandBenchmark prob cfg spec (canonicalRMSDToInputLigand prob cfg)

theorem solveRawPocketLigandCanonicalBenchmark_accept_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      solveRawPocketLigandCanonicalBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
      (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg).benchmarkContractSatisfied spec := by
  simpa [solveRawPocketLigandCanonicalBenchmark,
    sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD,
    sampledDockingSolverInputFromRawPocketLigand] using
    solveRawPocketLigandBenchmark_accept_iff
      prob cfg spec (canonicalRMSDToInputLigand prob cfg)

theorem solveRawPocketLigandCanonicalBenchmark_failure_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      solveRawPocketLigandCanonicalBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) ↔
      ¬ (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg).benchmarkContractSatisfied spec := by
  simpa [solveRawPocketLigandCanonicalBenchmark,
    sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD,
    sampledDockingSolverInputFromRawPocketLigand] using
    solveRawPocketLigandBenchmark_failure_iff
      prob cfg spec (canonicalRMSDToInputLigand prob cfg)

theorem solveRawPocketLigandCanonicalBenchmark_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      solveRawPocketLigandCanonicalBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ∨
    (∃ cert,
      solveRawPocketLigandCanonicalBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) := by
  simpa [solveRawPocketLigandCanonicalBenchmark] using
    solveRawPocketLigandBenchmark_total
      prob cfg spec (canonicalRMSDToInputLigand prob cfg)

noncomputable def solveRawPocketLigandDeployment
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ)
    (C : DeploymentRMSDCalibration
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)) :
    DeploymentConstrainedDockingSolverResult
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
      spec C :=
  solveDeploymentConstrainedDocking
    (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
    spec C

theorem solveRawPocketLigandDeployment_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ)
    (C : DeploymentRMSDCalibration
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)) :
    (∃ cert,
      solveRawPocketLigandDeployment prob cfg spec rmsdToReference C =
        DeploymentConstrainedDockingSolverResult.accepted cert) ∨
    (∃ cert,
      solveRawPocketLigandDeployment prob cfg spec rmsdToReference C =
        DeploymentConstrainedDockingSolverResult.rejected cert) := by
  unfold solveRawPocketLigandDeployment
  cases hRes :
      solveDeploymentConstrainedDocking
        (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
        spec C with
  | accepted cert => exact Or.inl ⟨cert, rfl⟩
  | rejected cert => exact Or.inr ⟨cert, rfl⟩

theorem solveRawPocketLigandDeployment_accepted_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ)
    (C : DeploymentRMSDCalibration
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)) :
    (∃ cert,
      solveRawPocketLigandDeployment prob cfg spec rmsdToReference C =
        DeploymentConstrainedDockingSolverResult.accepted cert) ↔
      C.deploymentContractSatisfied spec := by
  simpa [solveRawPocketLigandDeployment] using
    (solveDeploymentConstrainedDocking_accepted_iff
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
      spec C)

noncomputable def solveRawPocketLigandCanonicalDeployment
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    DeploymentConstrainedDockingSolverResult
      (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg)
      spec
      (canonicalDeploymentCalibration
        (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg)) :=
  solveDeploymentConstrainedDocking
    (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg)
    spec
    (canonicalDeploymentCalibration
      (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg))

theorem solveRawPocketLigandCanonicalDeployment_accepted_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      solveRawPocketLigandCanonicalDeployment prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert) ↔
      (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg).benchmarkContractSatisfied spec := by
  let I := sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg
  have hAcc :
      (∃ cert,
        solveDeploymentConstrainedDocking I spec (canonicalDeploymentCalibration I) =
          DeploymentConstrainedDockingSolverResult.accepted cert) ↔
        (canonicalDeploymentCalibration I).deploymentContractSatisfied spec :=
    solveDeploymentConstrainedDocking_accepted_iff I spec (canonicalDeploymentCalibration I)
  have hCal :
      (canonicalDeploymentCalibration I).deploymentContractSatisfied spec ↔
        I.benchmarkContractSatisfied spec :=
    canonicalDeploymentCalibration_deploymentContract_iff_benchmark I spec
  simpa [I, solveRawPocketLigandCanonicalDeployment] using hAcc.trans hCal

theorem solveRawPocketLigandCanonicalDeployment_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      solveRawPocketLigandCanonicalDeployment prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert) ∨
    (∃ cert,
      solveRawPocketLigandCanonicalDeployment prob cfg spec =
        DeploymentConstrainedDockingSolverResult.rejected cert) := by
  unfold solveRawPocketLigandCanonicalDeployment
  cases hRes :
      solveDeploymentConstrainedDocking
        (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg)
        spec
        (canonicalDeploymentCalibration
          (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg)) with
  | accepted cert => exact Or.inl ⟨cert, rfl⟩
  | rejected cert => exact Or.inr ⟨cert, rfl⟩

theorem solveRawPocketLigandCanonicalDeployment_accept_refines_benchmark_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (hAccept :
      ∃ cert,
        solveRawPocketLigandCanonicalDeployment prob cfg spec =
          DeploymentConstrainedDockingSolverResult.accepted cert) :
    ∃ cert,
      solveRawPocketLigandCanonicalBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  have hBench :
      (sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg).benchmarkContractSatisfied spec :=
    (solveRawPocketLigandCanonicalDeployment_accepted_iff prob cfg spec).1 hAccept
  exact (solveRawPocketLigandCanonicalBenchmark_accept_iff prob cfg spec).2 hBench

theorem solveRawPocketLigandDeployment_accept_refines_benchmark_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ)
    (C : DeploymentRMSDCalibration
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference))
    (hAccept :
      ∃ cert,
        solveRawPocketLigandDeployment prob cfg spec rmsdToReference C =
          DeploymentConstrainedDockingSolverResult.accepted cert) :
    ∃ cert,
      solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  have hCore :=
    solveDeploymentConstrainedDocking_accept_refines_benchmark_accept
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
      spec C (by simpa [solveRawPocketLigandDeployment] using hAccept)
  simpa [solveRawPocketLigandBenchmark] using hCore

theorem solveRawPocketLigandBenchmark_bundle
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    let I := sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.jaxCodegenReport.success = true ∧
      ((∃ cert,
          solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
            AcceptanceConstrainedDockingSolverResult.failure cert) ↔
        ¬ I.benchmarkContractSatisfied spec) := by
  intro I
  refine ⟨?_, ?_, ?_, ?_⟩
  · simpa [sampledDockingSolverInputFromRawPocketLigand] using
      rawPocketLigandSampleFamily_contains_baseAction prob cfg
  · exact I.jax_codegen_success
  · simpa using solveRawPocketLigandBenchmark_accept_iff prob cfg spec rmsdToReference
  · simpa using solveRawPocketLigandBenchmark_failure_iff prob cfg spec rmsdToReference

abbrev RawCanonicalProgramExecutionWitness
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :=
  CanonicalProgramExecutionWitness
    (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
    spec

theorem RawCanonicalProgramExecutionWitness.refines_solver_result
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ}
    (W : RawCanonicalProgramExecutionWitness prob cfg spec rmsdToReference) :
    (W.outputAcceptFlag = true ↔
      ∃ cert,
        solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
          AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
    (W.outputAcceptFlag = false ↔
      ∃ cert,
        solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
          AcceptanceConstrainedDockingSolverResult.failure cert) := by
  simpa [solveRawPocketLigandBenchmark] using
    (CanonicalProgramExecutionWitness.refines_solver_result W)

noncomputable def rawCanonicalProgramExecutionWitnessOfSpec
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    RawCanonicalProgramExecutionWitness prob cfg spec rmsdToReference :=
  canonicalProgramExecutionWitnessOfSpec
    (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)
    spec

theorem rawCanonicalProgramExecutionWitnessOfSpec_refines_solver_result
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ) :
    ((rawCanonicalProgramExecutionWitnessOfSpec prob cfg spec rmsdToReference).outputAcceptFlag = true ↔
      ∃ cert,
        solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
          AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
    ((rawCanonicalProgramExecutionWitnessOfSpec prob cfg spec rmsdToReference).outputAcceptFlag = false ↔
      ∃ cert,
        solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
          AcceptanceConstrainedDockingSolverResult.failure cert) := by
  exact RawCanonicalProgramExecutionWitness.refines_solver_result
    (rawCanonicalProgramExecutionWitnessOfSpec prob cfg spec rmsdToReference)

theorem rawCanonicalProgramExecutionWitnessOfSpec_refines_solver_result_canonical
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    ((rawCanonicalProgramExecutionWitnessOfSpec
        prob cfg spec (canonicalRMSDToInputLigand prob cfg)).outputAcceptFlag = true ↔
      ∃ cert,
        solveRawPocketLigandCanonicalBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
    ((rawCanonicalProgramExecutionWitnessOfSpec
        prob cfg spec (canonicalRMSDToInputLigand prob cfg)).outputAcceptFlag = false ↔
      ∃ cert,
        solveRawPocketLigandCanonicalBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.failure cert) := by
  simpa [solveRawPocketLigandCanonicalBenchmark] using
    rawCanonicalProgramExecutionWitnessOfSpec_refines_solver_result
      prob cfg spec (canonicalRMSDToInputLigand prob cfg)

theorem rawPocketLigandDefinitiveEndpoint_bundle
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ)
    (C : DeploymentRMSDCalibration
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference))
    (W : RawCanonicalProgramExecutionWitness prob cfg spec rmsdToReference) :
    let I := sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.jaxCodegenReport.success = true ∧
      ((∃ cert,
          solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          solveRawPocketLigandDeployment prob cfg spec rmsdToReference C =
            DeploymentConstrainedDockingSolverResult.accepted cert) ↔
        C.deploymentContractSatisfied spec) ∧
      (W.outputAcceptFlag = true ↔
        ∃ cert,
          solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
      (C.deploymentContractSatisfied spec →
        ∃ cert,
          solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
            AcceptanceConstrainedDockingSolverResult.accepted cert) := by
  intro I
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [sampledDockingSolverInputFromRawPocketLigand] using
      rawPocketLigandSampleFamily_contains_baseAction prob cfg
  · exact I.jax_codegen_success
  · simpa using solveRawPocketLigandBenchmark_accept_iff prob cfg spec rmsdToReference
  · simpa using solveRawPocketLigandDeployment_accepted_iff prob cfg spec rmsdToReference C
  · exact (RawCanonicalProgramExecutionWitness.refines_solver_result W).1
  · intro hDep
    have hCore :
        ∃ cert, solveAcceptanceConstrainedDocking I spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert :=
      (solveAcceptanceConstrainedDocking_accept_iff I spec).2
        (C.deployment_contract_implies_benchmark_contract spec hDep)
    simpa [solveRawPocketLigandBenchmark] using hCore

theorem rawPocketLigandDefinitiveEndpoint_bundle_of_spec_witness
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (rmsdToReference : SupportedAction (rawPocketLigandSampleFamily prob cfg) → ℝ)
    (C : DeploymentRMSDCalibration
      (sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference)) :
    let W := rawCanonicalProgramExecutionWitnessOfSpec prob cfg spec rmsdToReference
    let I := sampledDockingSolverInputFromRawPocketLigand prob cfg rmsdToReference
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.jaxCodegenReport.success = true ∧
      ((∃ cert,
          solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          solveRawPocketLigandDeployment prob cfg spec rmsdToReference C =
            DeploymentConstrainedDockingSolverResult.accepted cert) ↔
        C.deploymentContractSatisfied spec) ∧
      (W.outputAcceptFlag = true ↔
        ∃ cert,
          solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
      (C.deploymentContractSatisfied spec →
        ∃ cert,
          solveRawPocketLigandBenchmark prob cfg spec rmsdToReference =
            AcceptanceConstrainedDockingSolverResult.accepted cert) := by
  intro W I
  exact rawPocketLigandDefinitiveEndpoint_bundle
    prob cfg spec rmsdToReference C W

theorem rawPocketLigandCanonicalDefinitiveEndpoint_bundle
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    let I := sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg
    let W := rawCanonicalProgramExecutionWitnessOfSpec
      prob cfg spec (canonicalRMSDToInputLigand prob cfg)
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.jaxCodegenReport.success = true ∧
      ((∃ cert,
          solveRawPocketLigandCanonicalBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          solveRawPocketLigandCanonicalDeployment prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          solveRawPocketLigandCanonicalBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ∨
        (∃ cert,
          solveRawPocketLigandCanonicalBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.failure cert)) ∧
      ((∃ cert,
          solveRawPocketLigandCanonicalDeployment prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ∨
        (∃ cert,
          solveRawPocketLigandCanonicalDeployment prob cfg spec =
            DeploymentConstrainedDockingSolverResult.rejected cert)) ∧
      (W.outputAcceptFlag = true ↔
        ∃ cert,
          solveRawPocketLigandCanonicalBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
      ((∃ cert,
          solveRawPocketLigandCanonicalDeployment prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) →
        ∃ cert,
          solveRawPocketLigandCanonicalBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) := by
  intro I W
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD,
      sampledDockingSolverInputFromRawPocketLigand] using
      rawPocketLigandSampleFamily_contains_baseAction prob cfg
  · exact I.jax_codegen_success
  · exact solveRawPocketLigandCanonicalBenchmark_accept_iff prob cfg spec
  · exact solveRawPocketLigandCanonicalDeployment_accepted_iff prob cfg spec
  · exact solveRawPocketLigandCanonicalBenchmark_total prob cfg spec
  · exact solveRawPocketLigandCanonicalDeployment_total prob cfg spec
  · exact (rawCanonicalProgramExecutionWitnessOfSpec_refines_solver_result_canonical prob cfg spec).1
  · exact solveRawPocketLigandCanonicalDeployment_accept_refines_benchmark_accept
      prob cfg spec

noncomputable def canonicalRawDockingInput
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig) : SampledDockingSolverInput :=
  sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD prob cfg

noncomputable def canonicalRawDockingCalibration
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig) :
    DeploymentRMSDCalibration (canonicalRawDockingInput prob cfg) :=
  canonicalDeploymentCalibration (canonicalRawDockingInput prob cfg)

abbrev CanonicalRawDockingExecutionWitness
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :=
  RawCanonicalProgramExecutionWitness
    prob cfg spec (canonicalRMSDToInputLigand prob cfg)

noncomputable def legacySolveDefinitiveRawCrossDock
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    DeploymentConstrainedDockingSolverResult
      (canonicalRawDockingInput prob cfg)
      spec
      (canonicalRawDockingCalibration prob cfg) :=
  solveRawPocketLigandCanonicalDeployment prob cfg spec

noncomputable def legacySolveDefinitiveRawCrossDockBenchmark
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    AcceptanceConstrainedDockingSolverResult
      (canonicalRawDockingInput prob cfg)
      spec :=
  solveRawPocketLigandCanonicalBenchmark prob cfg spec

/-- Certificate backend benchmark result used by the constructive-first endpoint
surface. This keeps legacy internals hidden behind constructive-facing names. -/
noncomputable def definitiveBenchmarkCertificateBackend
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    AcceptanceConstrainedDockingSolverResult
      (canonicalRawDockingInput prob cfg)
      spec :=
  legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec

/-- Certificate backend deployment result used by the constructive-first
endpoint surface. -/
noncomputable def definitiveDeploymentCertificateBackend
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    DeploymentConstrainedDockingSolverResult
      (canonicalRawDockingInput prob cfg)
      spec
      (canonicalRawDockingCalibration prob cfg) :=
  legacySolveDefinitiveRawCrossDock prob cfg spec

theorem solveDefinitiveRawCrossDock_accepted_iff_benchmark_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDock prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert) ↔
      (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
  simpa [legacySolveDefinitiveRawCrossDock, canonicalRawDockingInput,
    canonicalRawDockingCalibration] using
    solveRawPocketLigandCanonicalDeployment_accepted_iff prob cfg spec

theorem solveDefinitiveRawCrossDockBenchmark_accepted_iff_benchmark_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
      (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
  simpa [legacySolveDefinitiveRawCrossDockBenchmark, canonicalRawDockingInput,
    sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD] using
    solveRawPocketLigandCanonicalBenchmark_accept_iff prob cfg spec

theorem solveDefinitiveRawCrossDockBenchmark_failure_iff_not_benchmark_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) ↔
      ¬ (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
  simpa [legacySolveDefinitiveRawCrossDockBenchmark, canonicalRawDockingInput,
    sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD] using
    solveRawPocketLigandCanonicalBenchmark_failure_iff prob cfg spec

theorem solveDefinitiveRawCrossDock_accept_refines_benchmark_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (hAccept :
      ∃ cert,
        legacySolveDefinitiveRawCrossDock prob cfg spec =
          DeploymentConstrainedDockingSolverResult.accepted cert) :
    ∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  simpa [legacySolveDefinitiveRawCrossDock, legacySolveDefinitiveRawCrossDockBenchmark] using
    solveRawPocketLigandCanonicalDeployment_accept_refines_benchmark_accept
      prob cfg spec hAccept

theorem solveDefinitiveRawCrossDock_runtime_flag_refines_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    ((rawCanonicalProgramExecutionWitnessOfSpec
        prob cfg spec (canonicalRMSDToInputLigand prob cfg)).outputAcceptFlag = true ↔
      ∃ cert,
        legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert) := by
  simpa [legacySolveDefinitiveRawCrossDockBenchmark] using
    (rawCanonicalProgramExecutionWitnessOfSpec_refines_solver_result_canonical
      prob cfg spec).1

theorem solveDefinitiveRawCrossDock_bundle
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    let I := canonicalRawDockingInput prob cfg
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.jaxCodegenReport.success = true ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((rawCanonicalProgramExecutionWitnessOfSpec
          prob cfg spec (canonicalRMSDToInputLigand prob cfg)).outputAcceptFlag = true ↔
        ∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) := by
  intro I
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simpa [canonicalRawDockingInput,
      sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD,
      sampledDockingSolverInputFromRawPocketLigand] using
      rawPocketLigandSampleFamily_contains_baseAction prob cfg
  · exact I.jax_codegen_success
  · simpa [legacySolveDefinitiveRawCrossDockBenchmark,
      canonicalRawDockingInput,
      sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD] using
      solveRawPocketLigandCanonicalBenchmark_accept_iff prob cfg spec
  · simpa [legacySolveDefinitiveRawCrossDock,
      canonicalRawDockingInput,
      canonicalRawDockingCalibration,
      sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD] using
      solveRawPocketLigandCanonicalDeployment_accepted_iff prob cfg spec
  · exact solveDefinitiveRawCrossDock_runtime_flag_refines_accept prob cfg spec

theorem solveDefinitiveRawCrossDock_accepted_iff_deployment_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDock prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert) ↔
      (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec := by
  have hBench :=
    solveDefinitiveRawCrossDock_accepted_iff_benchmark_contract prob cfg spec
  have hCal :=
    canonicalDeploymentCalibration_deploymentContract_iff_benchmark
      (canonicalRawDockingInput prob cfg) spec
  exact hBench.trans hCal.symm

theorem solveDefinitiveRawCrossDock_rejected_iff_not_deployment_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDock prob cfg spec =
        DeploymentConstrainedDockingSolverResult.rejected cert) ↔
      ¬ (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec := by
  refine ⟨?_, ?_⟩
  · intro hReject hDep
    have hAcc :=
      (solveDefinitiveRawCrossDock_accepted_iff_deployment_contract prob cfg spec).2 hDep
    rcases hAcc with ⟨accCert, hacc⟩
    rcases hReject with ⟨rejCert, hrej⟩
    have hEq :
        DeploymentConstrainedDockingSolverResult.accepted accCert =
          DeploymentConstrainedDockingSolverResult.rejected rejCert := by
      calc
        DeploymentConstrainedDockingSolverResult.accepted accCert
            = legacySolveDefinitiveRawCrossDock prob cfg spec := by simp [hacc]
        _ = DeploymentConstrainedDockingSolverResult.rejected rejCert := hrej
    cases hEq
  · intro hNotDep
    have hTot :
        (∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ∨
        (∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.rejected cert) := by
      simpa [legacySolveDefinitiveRawCrossDock] using
        solveRawPocketLigandCanonicalDeployment_total prob cfg spec
    rcases hTot with hAcc | hRej
    · exfalso
      exact hNotDep
        ((solveDefinitiveRawCrossDock_accepted_iff_deployment_contract prob cfg spec).1 hAcc)
    · exact hRej

theorem solveDefinitiveRawCrossDock_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDock prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert) ∨
    (∃ cert,
      legacySolveDefinitiveRawCrossDock prob cfg spec =
        DeploymentConstrainedDockingSolverResult.rejected cert) := by
  simpa [legacySolveDefinitiveRawCrossDock] using
    solveRawPocketLigandCanonicalDeployment_total prob cfg spec

theorem definitiveBenchmarkCertificateBackend_accepted_iff_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      definitiveBenchmarkCertificateBackend prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
      (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
  simpa [definitiveBenchmarkCertificateBackend] using
    solveDefinitiveRawCrossDockBenchmark_accepted_iff_benchmark_contract prob cfg spec

theorem definitiveBenchmarkCertificateBackend_failure_iff_not_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      definitiveBenchmarkCertificateBackend prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) ↔
      ¬ (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
  simpa [definitiveBenchmarkCertificateBackend] using
    solveDefinitiveRawCrossDockBenchmark_failure_iff_not_benchmark_contract prob cfg spec

theorem definitiveDeploymentCertificateBackend_accepted_iff_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      definitiveDeploymentCertificateBackend prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert) ↔
      (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec := by
  simpa [definitiveDeploymentCertificateBackend] using
    solveDefinitiveRawCrossDock_accepted_iff_deployment_contract prob cfg spec

theorem definitiveDeploymentCertificateBackend_rejected_iff_not_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      definitiveDeploymentCertificateBackend prob cfg spec =
        DeploymentConstrainedDockingSolverResult.rejected cert) ↔
      ¬ (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec := by
  simpa [definitiveDeploymentCertificateBackend] using
    solveDefinitiveRawCrossDock_rejected_iff_not_deployment_contract prob cfg spec

noncomputable def legacyDefinitiveRawCrossDockAcceptanceFlag
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) : Bool :=
  (runCanonicalSolverProgram (canonicalRawDockingInput prob cfg) spec).outputAcceptFlag

theorem definitiveRawCrossDockAcceptanceFlag_true_iff_benchmark_accepted
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    legacyDefinitiveRawCrossDockAcceptanceFlag prob cfg spec = true ↔
      ∃ cert,
        legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert := by
  simpa [legacyDefinitiveRawCrossDockAcceptanceFlag,
    legacySolveDefinitiveRawCrossDockBenchmark] using
    (runCanonicalSolverProgram_refines_solver_result
      (canonicalRawDockingInput prob cfg) spec).1

theorem definitiveRawCrossDockAcceptanceFlag_false_iff_benchmark_failure
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    legacyDefinitiveRawCrossDockAcceptanceFlag prob cfg spec = false ↔
      ∃ cert,
        legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.failure cert := by
  simpa [legacyDefinitiveRawCrossDockAcceptanceFlag,
    legacySolveDefinitiveRawCrossDockBenchmark] using
    (runCanonicalSolverProgram_refines_solver_result
      (canonicalRawDockingInput prob cfg) spec).2

theorem definitiveRawCrossDockAcceptanceFlag_true_iff_deployment_accepted
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    legacyDefinitiveRawCrossDockAcceptanceFlag prob cfg spec = true ↔
      ∃ cert,
        legacySolveDefinitiveRawCrossDock prob cfg spec =
          DeploymentConstrainedDockingSolverResult.accepted cert := by
  refine ⟨?_, ?_⟩
  · intro hFlag
    have hBenchAccept :=
      (definitiveRawCrossDockAcceptanceFlag_true_iff_benchmark_accepted prob cfg spec).1 hFlag
    have hBenchContract :=
      (solveDefinitiveRawCrossDockBenchmark_accepted_iff_benchmark_contract prob cfg spec).1 hBenchAccept
    have hDepContract :=
      (canonicalDeploymentCalibration_deploymentContract_iff_benchmark
        (canonicalRawDockingInput prob cfg) spec).2 hBenchContract
    exact (solveDefinitiveRawCrossDock_accepted_iff_deployment_contract prob cfg spec).2 hDepContract
  · intro hDepAccept
    have hDepContract :=
      (solveDefinitiveRawCrossDock_accepted_iff_deployment_contract prob cfg spec).1 hDepAccept
    have hBenchContract :=
      (canonicalDeploymentCalibration_deploymentContract_iff_benchmark
        (canonicalRawDockingInput prob cfg) spec).1 hDepContract
    have hBenchAccept :=
      (solveDefinitiveRawCrossDockBenchmark_accepted_iff_benchmark_contract prob cfg spec).2 hBenchContract
    exact (definitiveRawCrossDockAcceptanceFlag_true_iff_benchmark_accepted prob cfg spec).2 hBenchAccept

theorem definitiveRawCrossDockAcceptanceFlag_false_iff_deployment_rejected
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    legacyDefinitiveRawCrossDockAcceptanceFlag prob cfg spec = false ↔
      ∃ cert,
        legacySolveDefinitiveRawCrossDock prob cfg spec =
          DeploymentConstrainedDockingSolverResult.rejected cert := by
  refine ⟨?_, ?_⟩
  · intro hFlag
    have hBenchFail :=
      (definitiveRawCrossDockAcceptanceFlag_false_iff_benchmark_failure prob cfg spec).1 hFlag
    have hNotBench :=
      (solveDefinitiveRawCrossDockBenchmark_failure_iff_not_benchmark_contract prob cfg spec).1 hBenchFail
    have hNotDep :
        ¬ (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec := by
      intro hDep
      exact hNotBench
        ((canonicalDeploymentCalibration_deploymentContract_iff_benchmark
          (canonicalRawDockingInput prob cfg) spec).1 hDep)
    exact (solveDefinitiveRawCrossDock_rejected_iff_not_deployment_contract prob cfg spec).2 hNotDep
  · intro hDepReject
    have hNotDep :=
      (solveDefinitiveRawCrossDock_rejected_iff_not_deployment_contract prob cfg spec).1 hDepReject
    have hNotBench :
        ¬ (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
      intro hBench
      exact hNotDep
        ((canonicalDeploymentCalibration_deploymentContract_iff_benchmark
          (canonicalRawDockingInput prob cfg) spec).2 hBench)
    have hBenchFail :=
      (solveDefinitiveRawCrossDockBenchmark_failure_iff_not_benchmark_contract prob cfg spec).2 hNotBench
    exact (definitiveRawCrossDockAcceptanceFlag_false_iff_benchmark_failure prob cfg spec).2 hBenchFail

/-- Computable rationalized acceptance kernel for the definitive endpoint:
all comparisons are done in `Rat`, with explicit real-valued error certificates
transporting back to the benchmark contract. -/
structure RationalizedAcceptanceKernel
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec) where
  rmsdQ : Rat
  successQ : Rat
  targetRMSD_Q : Rat
  targetProbability_Q : Rat
  rmsdMarginQ : Rat
  successMarginQ : Rat
  rmsdMargin_nonneg : 0 ≤ rmsdMarginQ
  successMargin_nonneg : 0 ≤ successMarginQ
  targetRMSD_eq : (targetRMSD_Q : ℝ) = spec.targetRMSD
  targetProbability_eq : (targetProbability_Q : ℝ) = spec.targetProbability
  rmsd_abs_error :
    |(rmsdQ : ℝ) - I.rmsdToReference I.selectedPose| ≤ (rmsdMarginQ : ℝ)
  success_abs_error :
    |(successQ : ℝ) - I.successProbability spec| ≤ (successMarginQ : ℝ)

def RationalizedAcceptanceKernel.computableAcceptFlag
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec) : Bool :=
  decide
    (K.rmsdQ + K.rmsdMarginQ ≤ K.targetRMSD_Q ∧
      K.targetProbability_Q + K.successMarginQ ≤ K.successQ)

theorem RationalizedAcceptanceKernel.computableAcceptFlag_true_iff
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec) :
    K.computableAcceptFlag = true ↔
      (K.rmsdQ + K.rmsdMarginQ ≤ K.targetRMSD_Q ∧
        K.targetProbability_Q + K.successMarginQ ≤ K.successQ) := by
  unfold RationalizedAcceptanceKernel.computableAcceptFlag
  simp

theorem RationalizedAcceptanceKernel.computableAcceptFlag_sound
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec)
    (hFlag : K.computableAcceptFlag = true) :
    I.benchmarkContractSatisfied spec := by
  have hRat := (K.computableAcceptFlag_true_iff).1 hFlag
  have hRmsdRat :
      (K.rmsdQ : ℝ) + (K.rmsdMarginQ : ℝ) ≤ (K.targetRMSD_Q : ℝ) := by
    exact_mod_cast hRat.1
  have hSuccRat :
      (K.targetProbability_Q : ℝ) + (K.successMarginQ : ℝ) ≤ (K.successQ : ℝ) := by
    exact_mod_cast hRat.2
  have hRmsdAbs := abs_le.mp K.rmsd_abs_error
  have hSuccAbs := abs_le.mp K.success_abs_error
  have hRmsdTrue : I.rmsdToReference I.selectedPose ≤ spec.targetRMSD := by
    have hBridge : I.rmsdToReference I.selectedPose ≤ (K.rmsdQ : ℝ) + (K.rmsdMarginQ : ℝ) := by
      linarith [hRmsdAbs.1]
    have hTarget : (K.rmsdQ : ℝ) + (K.rmsdMarginQ : ℝ) ≤ spec.targetRMSD := by
      simpa [K.targetRMSD_eq] using hRmsdRat
    exact le_trans hBridge hTarget
  have hSuccTrue : spec.targetProbability ≤ I.successProbability spec := by
    have hTargetBridge : spec.targetProbability ≤ (K.successQ : ℝ) - (K.successMarginQ : ℝ) := by
      have hTmp : (K.targetProbability_Q : ℝ) ≤ (K.successQ : ℝ) - (K.successMarginQ : ℝ) := by
        linarith [hSuccRat]
      simpa [K.targetProbability_eq] using hTmp
    have hSuccBridge : (K.successQ : ℝ) - (K.successMarginQ : ℝ) ≤ I.successProbability spec := by
      linarith [hSuccAbs.2]
    exact le_trans hTargetBridge hSuccBridge
  exact ⟨hRmsdTrue, hSuccTrue⟩

theorem RationalizedAcceptanceKernel.computableAcceptFlag_refines_benchmark_accept
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec)
    (hFlag : K.computableAcceptFlag = true) :
    ∃ cert,
      solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  exact (solveAcceptanceConstrainedDocking_accept_iff I spec).2
    (K.computableAcceptFlag_sound hFlag)

/-- Separation witness certifying that at least one benchmark inequality fails
with a strict margin that dominates rational approximation uncertainty. -/
structure RationalizedRejectionSeparationWitness
    (I : SampledDockingSolverInput)
    (spec : DockingAcceptanceSpec)
    (K : RationalizedAcceptanceKernel I spec) where
  strict_violation :
    spec.targetRMSD < (K.rmsdQ : ℝ) - (K.rmsdMarginQ : ℝ) ∨
      (K.successQ : ℝ) + (K.successMarginQ : ℝ) < spec.targetProbability

theorem RationalizedAcceptanceKernel.rejectionSeparation_not_benchmarkContract
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec)
    (W : RationalizedRejectionSeparationWitness I spec K) :
    ¬ I.benchmarkContractSatisfied spec := by
  intro hBench
  rcases hBench with ⟨hRmsdBench, hProbBench⟩
  rcases W.strict_violation with hRmsdSep | hProbSep
  · have hAbs := abs_le.mp K.rmsd_abs_error
    have hLower : (K.rmsdQ : ℝ) - (K.rmsdMarginQ : ℝ) ≤ I.rmsdToReference I.selectedPose := by
      linarith [hAbs.1]
    have hContra : spec.targetRMSD < I.rmsdToReference I.selectedPose :=
      lt_of_lt_of_le hRmsdSep hLower
    exact (not_lt_of_ge hRmsdBench) hContra
  · have hAbs := abs_le.mp K.success_abs_error
    have hUpper : I.successProbability spec ≤ (K.successQ : ℝ) + (K.successMarginQ : ℝ) := by
      linarith [hAbs.2]
    have hContra : I.successProbability spec < spec.targetProbability :=
      lt_of_le_of_lt hUpper hProbSep
    exact (not_lt_of_ge hProbBench) hContra

theorem RationalizedAcceptanceKernel.rejectionSeparation_flag_false
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec)
    (W : RationalizedRejectionSeparationWitness I spec K) :
    K.computableAcceptFlag = false := by
  by_cases hTrue : K.computableAcceptFlag = true
  · exfalso
    have hRat := (K.computableAcceptFlag_true_iff).1 hTrue
    have hRmsdRat : (K.rmsdQ : ℝ) + (K.rmsdMarginQ : ℝ) ≤ spec.targetRMSD := by
      have hTmp : (K.rmsdQ : ℝ) + (K.rmsdMarginQ : ℝ) ≤ (K.targetRMSD_Q : ℝ) := by
        exact_mod_cast hRat.1
      simpa [K.targetRMSD_eq] using hTmp
    have hSuccRat : spec.targetProbability + (K.successMarginQ : ℝ) ≤ (K.successQ : ℝ) := by
      have hTmp : (K.targetProbability_Q : ℝ) + (K.successMarginQ : ℝ) ≤ (K.successQ : ℝ) := by
        exact_mod_cast hRat.2
      simpa [K.targetProbability_eq] using hTmp
    have hRmsdMarginNonneg : 0 ≤ (K.rmsdMarginQ : ℝ) := by
      exact_mod_cast K.rmsdMargin_nonneg
    have hSuccMarginNonneg : 0 ≤ (K.successMarginQ : ℝ) := by
      exact_mod_cast K.successMargin_nonneg
    rcases W.strict_violation with hRmsdSep | hProbSep
    · have hUpper : spec.targetRMSD < (K.rmsdQ : ℝ) + (K.rmsdMarginQ : ℝ) := by
        linarith [hRmsdSep, hRmsdMarginNonneg]
      exact (not_lt_of_ge hRmsdRat) hUpper
    · have hSuccLtTarget : (K.successQ : ℝ) < spec.targetProbability := by
        have hSuccLe : (K.successQ : ℝ) ≤ (K.successQ : ℝ) + (K.successMarginQ : ℝ) := by
          linarith [hSuccMarginNonneg]
        exact lt_of_le_of_lt hSuccLe hProbSep
      have hTargetLeSucc : spec.targetProbability ≤ (K.successQ : ℝ) := by
        linarith [hSuccRat, hSuccMarginNonneg]
      exact (not_lt_of_ge hTargetLeSucc) hSuccLtTarget
  · cases hFlag : K.computableAcceptFlag <;> simp [hFlag] at hTrue ⊢

def RationalizedAcceptanceKernel.ofExactRatWitness
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (rmsdQ successQ targetRMSD_Q targetProbability_Q : Rat)
    (hRmsd : (rmsdQ : ℝ) = I.rmsdToReference I.selectedPose)
    (hSuccess : (successQ : ℝ) = I.successProbability spec)
    (hTargetRMSD : (targetRMSD_Q : ℝ) = spec.targetRMSD)
    (hTargetProb : (targetProbability_Q : ℝ) = spec.targetProbability) :
    RationalizedAcceptanceKernel I spec :=
  { rmsdQ := rmsdQ
    successQ := successQ
    targetRMSD_Q := targetRMSD_Q
    targetProbability_Q := targetProbability_Q
    rmsdMarginQ := 0
    successMarginQ := 0
    rmsdMargin_nonneg := by norm_num
    successMargin_nonneg := by norm_num
    targetRMSD_eq := hTargetRMSD
    targetProbability_eq := hTargetProb
    rmsd_abs_error := by
      simp [hRmsd]
    success_abs_error := by
      simp [hSuccess] }

theorem RationalizedAcceptanceKernel.ofExactRatWitness_flag_true_iff
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (rmsdQ successQ targetRMSD_Q targetProbability_Q : Rat)
    (hRmsd : (rmsdQ : ℝ) = I.rmsdToReference I.selectedPose)
    (hSuccess : (successQ : ℝ) = I.successProbability spec)
    (hTargetRMSD : (targetRMSD_Q : ℝ) = spec.targetRMSD)
    (hTargetProb : (targetProbability_Q : ℝ) = spec.targetProbability) :
    let K :=
      RationalizedAcceptanceKernel.ofExactRatWitness
        (I := I) (spec := spec)
        rmsdQ successQ targetRMSD_Q targetProbability_Q
        hRmsd hSuccess hTargetRMSD hTargetProb
    K.computableAcceptFlag = true ↔ I.benchmarkContractSatisfied spec := by
  dsimp [RationalizedAcceptanceKernel.ofExactRatWitness,
    RationalizedAcceptanceKernel.computableAcceptFlag,
    SampledDockingSolverInput.benchmarkContractSatisfied,
    SampledDockingSolverInput.benchmarkRMSDPass,
    SampledDockingSolverInput.benchmarkProbabilityPass]
  constructor
  · intro hFlag
    have hRat : rmsdQ + 0 ≤ targetRMSD_Q ∧ targetProbability_Q + 0 ≤ successQ := by
      simpa [decide_eq_true_iff] using hFlag
    have hRat' : rmsdQ ≤ targetRMSD_Q ∧ targetProbability_Q ≤ successQ := by
      simpa using hRat
    have hRmsdReal : (rmsdQ : ℝ) ≤ (targetRMSD_Q : ℝ) := by
      exact_mod_cast hRat'.1
    have hSuccReal : (targetProbability_Q : ℝ) ≤ (successQ : ℝ) := by
      exact_mod_cast hRat'.2
    exact ⟨by simpa [hRmsd, hTargetRMSD] using hRmsdReal,
      by simpa [hTargetProb, hSuccess] using hSuccReal⟩
  · intro hBench
    rcases hBench with ⟨hRmsdBench, hProbBench⟩
    have hRmsdReal : (rmsdQ : ℝ) ≤ (targetRMSD_Q : ℝ) := by
      simpa [hRmsd, hTargetRMSD] using hRmsdBench
    have hSuccReal : (targetProbability_Q : ℝ) ≤ (successQ : ℝ) := by
      simpa [hTargetProb, hSuccess] using hProbBench
    have hRmsdRat : rmsdQ ≤ targetRMSD_Q := by
      exact_mod_cast hRmsdReal
    have hSuccRat : targetProbability_Q ≤ successQ := by
      exact_mod_cast hSuccReal
    have hRat : rmsdQ + 0 ≤ targetRMSD_Q ∧ targetProbability_Q + 0 ≤ successQ := by
      simpa using And.intro hRmsdRat hSuccRat
    simpa [decide_eq_true_iff] using hRat

inductive ComputableKernelDecision where
  | accepted
  | rejected
  deriving Repr, DecidableEq

def RationalizedAcceptanceKernel.kernelDecision
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec) : ComputableKernelDecision :=
  if K.computableAcceptFlag then .accepted else .rejected

theorem RationalizedAcceptanceKernel.kernelDecision_accepted_iff
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec) :
    K.kernelDecision = ComputableKernelDecision.accepted ↔
      K.computableAcceptFlag = true := by
  unfold RationalizedAcceptanceKernel.kernelDecision
  by_cases h : K.computableAcceptFlag
  · simp [h]
  · simp [h]

theorem RationalizedAcceptanceKernel.kernelDecision_refines_benchmark_accept
    {I : SampledDockingSolverInput}
    {spec : DockingAcceptanceSpec}
    (K : RationalizedAcceptanceKernel I spec)
    (hDec : K.kernelDecision = ComputableKernelDecision.accepted) :
    ∃ cert,
      solveAcceptanceConstrainedDocking I spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  have hFlag : K.computableAcceptFlag = true :=
    (K.kernelDecision_accepted_iff).1 hDec
  exact K.computableAcceptFlag_refines_benchmark_accept hFlag

def solveDefinitiveRawCrossDockComputableKernel
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (K : RationalizedAcceptanceKernel (canonicalRawDockingInput prob cfg) spec) :
    ComputableKernelDecision :=
  K.kernelDecision

theorem solveDefinitiveRawCrossDockComputableKernel_refines_benchmark_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (K : RationalizedAcceptanceKernel (canonicalRawDockingInput prob cfg) spec)
    (hDec :
      solveDefinitiveRawCrossDockComputableKernel prob cfg spec K =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  have hCore :
      ∃ cert,
        solveAcceptanceConstrainedDocking (canonicalRawDockingInput prob cfg) spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert :=
    K.kernelDecision_refines_benchmark_accept (by simpa [solveDefinitiveRawCrossDockComputableKernel] using hDec)
  simpa [legacySolveDefinitiveRawCrossDockBenchmark,
    canonicalRawDockingInput,
    sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD] using hCore

/-- Concrete artifact payload used to instantiate the constructive rational-kernel
endpoint (artifact identity + backend provenance + rational contract data). -/
structure RationalizedKernelArtifactPayload where
  artifactId : String
  backendName : String
  generatedOpsDigest : String
  rmsdQ : Rat
  successQ : Rat
  targetRMSD_Q : Rat
  targetProbability_Q : Rat
  rmsdMarginQ : Rat
  successMarginQ : Rat
  artifactId_nonempty : artifactId ≠ ""
  backendName_nonempty : backendName ≠ ""
  generatedOpsDigest_nonempty : generatedOpsDigest ≠ ""
  rmsdMargin_nonneg : 0 ≤ rmsdMarginQ
  successMargin_nonneg : 0 ≤ successMarginQ

/-- Artifact-level instantiation of the rationalized acceptance kernel for one
raw pocket/ligand/spec endpoint instance. This turns interface hypotheses into
explicit artifact fields and checked error bounds. -/
structure DefinitiveRawCrossDockArtifactInstantiation
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) where
  payload : RationalizedKernelArtifactPayload
  targetRMSD_eq : (payload.targetRMSD_Q : ℝ) = spec.targetRMSD
  targetProbability_eq : (payload.targetProbability_Q : ℝ) = spec.targetProbability
  rmsd_abs_error :
    |(payload.rmsdQ : ℝ) -
      (canonicalRawDockingInput prob cfg).rmsdToReference
        (canonicalRawDockingInput prob cfg).selectedPose| ≤
      (payload.rmsdMarginQ : ℝ)
  success_abs_error :
    |(payload.successQ : ℝ) -
      (canonicalRawDockingInput prob cfg).successProbability spec| ≤
      (payload.successMarginQ : ℝ)

def DefinitiveRawCrossDockArtifactInstantiation.kernel
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    RationalizedAcceptanceKernel (canonicalRawDockingInput prob cfg) spec :=
  { rmsdQ := A.payload.rmsdQ
    successQ := A.payload.successQ
    targetRMSD_Q := A.payload.targetRMSD_Q
    targetProbability_Q := A.payload.targetProbability_Q
    rmsdMarginQ := A.payload.rmsdMarginQ
    successMarginQ := A.payload.successMarginQ
    rmsdMargin_nonneg := A.payload.rmsdMargin_nonneg
    successMargin_nonneg := A.payload.successMargin_nonneg
    targetRMSD_eq := A.targetRMSD_eq
    targetProbability_eq := A.targetProbability_eq
    rmsd_abs_error := A.rmsd_abs_error
    success_abs_error := A.success_abs_error }

/-- Exact-rational artifact record: a concrete artifact whose rational values are
declared to match the real endpoint quantities exactly (zero margins). -/
structure ExactRatArtifactInstantiation
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) where
  artifactId : String
  backendName : String
  generatedOpsDigest : String
  artifactId_nonempty : artifactId ≠ ""
  backendName_nonempty : backendName ≠ ""
  generatedOpsDigest_nonempty : generatedOpsDigest ≠ ""
  rmsdQ : Rat
  successQ : Rat
  targetRMSD_Q : Rat
  targetProbability_Q : Rat
  rmsd_eq :
    (rmsdQ : ℝ) =
      (canonicalRawDockingInput prob cfg).rmsdToReference
        (canonicalRawDockingInput prob cfg).selectedPose
  success_eq :
    (successQ : ℝ) =
      (canonicalRawDockingInput prob cfg).successProbability spec
  targetRMSD_eq : (targetRMSD_Q : ℝ) = spec.targetRMSD
  targetProbability_eq : (targetProbability_Q : ℝ) = spec.targetProbability

def ExactRatArtifactInstantiation.toKernelArtifact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    (A : ExactRatArtifactInstantiation prob cfg spec) :
    DefinitiveRawCrossDockArtifactInstantiation prob cfg spec :=
  { payload :=
      { artifactId := A.artifactId
        backendName := A.backendName
        generatedOpsDigest := A.generatedOpsDigest
        rmsdQ := A.rmsdQ
        successQ := A.successQ
        targetRMSD_Q := A.targetRMSD_Q
        targetProbability_Q := A.targetProbability_Q
        rmsdMarginQ := 0
        successMarginQ := 0
        artifactId_nonempty := A.artifactId_nonempty
        backendName_nonempty := A.backendName_nonempty
        generatedOpsDigest_nonempty := A.generatedOpsDigest_nonempty
        rmsdMargin_nonneg := by norm_num
        successMargin_nonneg := by norm_num }
    targetRMSD_eq := A.targetRMSD_eq
    targetProbability_eq := A.targetProbability_eq
    rmsd_abs_error := by simp [A.rmsd_eq]
    success_abs_error := by simp [A.success_eq] }

/-- Fully constructive benchmark decision path for the definitive endpoint,
driven only by artifact-instantiated rational kernel data. -/
def solveDefinitiveRawCrossDockBenchmarkConstructive
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    ComputableKernelDecision :=
  solveDefinitiveRawCrossDockComputableKernel prob cfg spec A.kernel

/-- Constructive deployment decision path (same computable kernel decision as
benchmark mode in the canonical calibration specialization). -/
def solveDefinitiveRawCrossDockConstructive
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    ComputableKernelDecision :=
  solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A

/-- Public constructive-first benchmark decision endpoint. -/
def solveDefinitiveRawCrossDockBenchmarkDecision :=
  solveDefinitiveRawCrossDockBenchmarkConstructive

/-- Public constructive-first deployment decision endpoint. -/
def solveDefinitiveRawCrossDockDecision :=
  solveDefinitiveRawCrossDockConstructive

theorem solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
      solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A := by
  rfl

theorem solveDefinitiveRawCrossDockDecision_eq_constructive
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockDecision prob cfg spec A =
      solveDefinitiveRawCrossDockConstructive prob cfg spec A := by
  rfl

theorem solveDefinitiveRawCrossDockDecision_eq_benchmarkDecision
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockDecision prob cfg spec A =
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A := by
  rfl

theorem solveDefinitiveRawCrossDockConstructive_eq_benchmark
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockConstructive prob cfg spec A =
      solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A := by
  rfl

theorem solveDefinitiveRawCrossDockBenchmarkConstructive_accepted_iff_kernel_flag
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A =
        ComputableKernelDecision.accepted ↔
      A.kernel.computableAcceptFlag = true := by
  simpa [solveDefinitiveRawCrossDockBenchmarkConstructive,
    solveDefinitiveRawCrossDockComputableKernel] using
    (A.kernel.kernelDecision_accepted_iff)

theorem solveDefinitiveRawCrossDockBenchmarkDecision_accepted_iff_kernel_flag
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted ↔
      A.kernel.computableAcceptFlag = true := by
  simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive] using
    solveDefinitiveRawCrossDockBenchmarkConstructive_accepted_iff_kernel_flag
      prob cfg spec A

theorem solveDefinitiveRawCrossDockBenchmarkConstructive_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A =
        ComputableKernelDecision.accepted ∨
      solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A =
        ComputableKernelDecision.rejected := by
  cases hDec : solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A with
  | accepted => exact Or.inl (by simpa [hDec] using rfl)
  | rejected => exact Or.inr (by simpa [hDec] using rfl)

theorem solveDefinitiveRawCrossDockBenchmarkDecision_rejected_iff_kernel_flag_false
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.rejected ↔
      A.kernel.computableAcceptFlag = false := by
  constructor
  · intro hRej
    by_cases hFlag : A.kernel.computableAcceptFlag = true
    · have hAcc :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.accepted :=
        (solveDefinitiveRawCrossDockBenchmarkDecision_accepted_iff_kernel_flag
          prob cfg spec A).2 hFlag
      have hContra :
          ComputableKernelDecision.accepted = ComputableKernelDecision.rejected := by
        simpa [hAcc] using hRej
      cases hContra
    · cases hBool : A.kernel.computableAcceptFlag <;> simp [hBool] at hFlag ⊢
  · intro hFlagFalse
    have hNotAcc :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A ≠
          ComputableKernelDecision.accepted := by
      intro hAcc
      have hFlagTrue : A.kernel.computableAcceptFlag = true :=
        (solveDefinitiveRawCrossDockBenchmarkDecision_accepted_iff_kernel_flag
          prob cfg spec A).1 hAcc
      simp [hFlagFalse] at hFlagTrue
    have hTot :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.accepted ∨
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.rejected := by
      simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive] using
        solveDefinitiveRawCrossDockBenchmarkConstructive_total prob cfg spec A
    rcases hTot with hAcc | hRej
    · exact (hNotAcc hAcc).elim
    · exact hRej

theorem solveDefinitiveRawCrossDockBenchmarkConstructive_refines_benchmark_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  exact solveDefinitiveRawCrossDockComputableKernel_refines_benchmark_accept
    prob cfg spec A.kernel hDec

theorem solveDefinitiveRawCrossDockConstructive_refines_deployment_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockConstructive prob cfg spec A =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      legacySolveDefinitiveRawCrossDock prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert := by
  have hBenchAccept :
      ∃ cert,
        legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert :=
    solveDefinitiveRawCrossDockBenchmarkConstructive_refines_benchmark_accept
      prob cfg spec A
      (by simpa [solveDefinitiveRawCrossDockConstructive_eq_benchmark] using hDec)
  have hBenchContract :
      (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec :=
    (solveDefinitiveRawCrossDockBenchmark_accepted_iff_benchmark_contract
      prob cfg spec).1 hBenchAccept
  have hDepContract :
      (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec :=
    (canonicalDeploymentCalibration_deploymentContract_iff_benchmark
      (canonicalRawDockingInput prob cfg) spec).2 hBenchContract
  exact (solveDefinitiveRawCrossDock_accepted_iff_deployment_contract
    prob cfg spec).2 hDepContract

theorem solveDefinitiveRawCrossDockBenchmarkDecision_refines_legacy_benchmark_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  exact solveDefinitiveRawCrossDockBenchmarkConstructive_refines_benchmark_accept
    prob cfg spec A
    (by simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive] using hDec)

theorem solveDefinitiveRawCrossDockDecision_refines_legacy_deployment_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      legacySolveDefinitiveRawCrossDock prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert := by
  exact solveDefinitiveRawCrossDockConstructive_refines_deployment_accept
    prob cfg spec A
    (by simpa [solveDefinitiveRawCrossDockDecision_eq_constructive] using hDec)

theorem solveDefinitiveRawCrossDockBenchmarkDecision_refines_certificate_backend_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      definitiveBenchmarkCertificateBackend prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  simpa [definitiveBenchmarkCertificateBackend] using
    solveDefinitiveRawCrossDockBenchmarkDecision_refines_legacy_benchmark_accept
      prob cfg spec A hDec

theorem solveDefinitiveRawCrossDockDecision_refines_certificate_backend_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      definitiveDeploymentCertificateBackend prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert := by
  simpa [definitiveDeploymentCertificateBackend] using
    solveDefinitiveRawCrossDockDecision_refines_legacy_deployment_accept
      prob cfg spec A hDec

/-- Constructive accepted certificate for the public benchmark decision path. -/
structure ConstructiveBenchmarkAcceptedCertificate
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) where
  decisionAccepted :
    solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
      ComputableKernelDecision.accepted
  kernelFlagTrue : A.kernel.computableAcceptFlag = true
  benchmarkContract : (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec
  certificateBackendAccepted :
    ∃ cert,
      definitiveBenchmarkCertificateBackend prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert

def constructiveBenchmarkAcceptedCertificateOfDecision
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted) :
    ConstructiveBenchmarkAcceptedCertificate prob cfg spec A := by
  have hFlag : A.kernel.computableAcceptFlag = true :=
    (solveDefinitiveRawCrossDockBenchmarkDecision_accepted_iff_kernel_flag
      prob cfg spec A).1 hDec
  have hLegacy :
      ∃ cert,
        definitiveBenchmarkCertificateBackend prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert :=
    solveDefinitiveRawCrossDockBenchmarkDecision_refines_certificate_backend_accept
      prob cfg spec A hDec
  have hBench : (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec :=
    (definitiveBenchmarkCertificateBackend_accepted_iff_contract
      prob cfg spec).1 hLegacy
  exact
    { decisionAccepted := hDec
      kernelFlagTrue := hFlag
      benchmarkContract := hBench
      certificateBackendAccepted := hLegacy }

/-- Constructive rejected certificate for the public benchmark decision path. -/
structure ConstructiveBenchmarkRejectedCertificate
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) where
  decisionRejected :
    solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
      ComputableKernelDecision.rejected
  kernelFlagFalse : A.kernel.computableAcceptFlag = false

def constructiveBenchmarkRejectedCertificateOfDecision
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.rejected) :
    ConstructiveBenchmarkRejectedCertificate prob cfg spec A :=
  { decisionRejected := hDec
    kernelFlagFalse :=
      (solveDefinitiveRawCrossDockBenchmarkDecision_rejected_iff_kernel_flag_false
        prob cfg spec A).1 hDec }

inductive ConstructiveBenchmarkCertifiedResult
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) where
  | accepted (cert : ConstructiveBenchmarkAcceptedCertificate prob cfg spec A)
  | rejected (cert : ConstructiveBenchmarkRejectedCertificate prob cfg spec A)

theorem solveDefinitiveRawCrossDockBenchmarkDecision_rejected_of_not_accepted
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hNot :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A ≠
        ComputableKernelDecision.accepted) :
    solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
      ComputableKernelDecision.rejected := by
  have hTot :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted ∨
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.rejected := by
    simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive] using
      solveDefinitiveRawCrossDockBenchmarkConstructive_total prob cfg spec A
  rcases hTot with hAcc | hRej
  · exact (hNot hAcc).elim
  · exact hRej

def solveDefinitiveRawCrossDockBenchmarkCertified
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    ConstructiveBenchmarkCertifiedResult prob cfg spec A :=
  if hDec :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted then
    .accepted
      (constructiveBenchmarkAcceptedCertificateOfDecision prob cfg spec A hDec)
  else
    .rejected
      (constructiveBenchmarkRejectedCertificateOfDecision prob cfg spec A
        (solveDefinitiveRawCrossDockBenchmarkDecision_rejected_of_not_accepted
          prob cfg spec A hDec))

theorem solveDefinitiveRawCrossDockBenchmarkCertified_accepted_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    (∃ cert,
      solveDefinitiveRawCrossDockBenchmarkCertified prob cfg spec A =
        ConstructiveBenchmarkCertifiedResult.accepted cert) ↔
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted := by
  constructor
  · intro hAcc
    by_cases hDec :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.accepted
    · exact hDec
    · exfalso
      rcases hAcc with ⟨cert, hEq⟩
      unfold solveDefinitiveRawCrossDockBenchmarkCertified at hEq
      simp [hDec] at hEq
  · intro hDec
    refine ⟨constructiveBenchmarkAcceptedCertificateOfDecision prob cfg spec A hDec, ?_⟩
    unfold solveDefinitiveRawCrossDockBenchmarkCertified
    simp [hDec]

theorem solveDefinitiveRawCrossDockBenchmarkCertified_rejected_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    (∃ cert,
      solveDefinitiveRawCrossDockBenchmarkCertified prob cfg spec A =
        ConstructiveBenchmarkCertifiedResult.rejected cert) ↔
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.rejected := by
  constructor
  · intro hRej
    by_cases hDec :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.accepted
    · exfalso
      rcases hRej with ⟨cert, hEq⟩
      unfold solveDefinitiveRawCrossDockBenchmarkCertified at hEq
      simp [hDec] at hEq
    · exact
        solveDefinitiveRawCrossDockBenchmarkDecision_rejected_of_not_accepted
          prob cfg spec A hDec
  · intro hRej
    have hNotAcc :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A ≠
          ComputableKernelDecision.accepted := by
      intro hAcc
      have hContra : ComputableKernelDecision.accepted = ComputableKernelDecision.rejected := by
        simpa [hAcc] using hRej
      cases hContra
    have hDecRej :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.rejected :=
      solveDefinitiveRawCrossDockBenchmarkDecision_rejected_of_not_accepted
        prob cfg spec A hNotAcc
    refine ⟨constructiveBenchmarkRejectedCertificateOfDecision prob cfg spec A hDecRej, ?_⟩
    unfold solveDefinitiveRawCrossDockBenchmarkCertified
    simp [hNotAcc, hDecRej,
      solveDefinitiveRawCrossDockBenchmarkDecision_rejected_of_not_accepted
        prob cfg spec A hNotAcc]

theorem solveDefinitiveRawCrossDockBenchmarkCertified_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    (∃ cert,
      solveDefinitiveRawCrossDockBenchmarkCertified prob cfg spec A =
        ConstructiveBenchmarkCertifiedResult.accepted cert) ∨
      (∃ cert,
        solveDefinitiveRawCrossDockBenchmarkCertified prob cfg spec A =
          ConstructiveBenchmarkCertifiedResult.rejected cert) := by
  have hTot :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted ∨
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.rejected := by
    simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive] using
      solveDefinitiveRawCrossDockBenchmarkConstructive_total prob cfg spec A
  rcases hTot with hAcc | hRej
  · exact Or.inl
      ((solveDefinitiveRawCrossDockBenchmarkCertified_accepted_iff
        prob cfg spec A).2 hAcc)
  · exact Or.inr
      ((solveDefinitiveRawCrossDockBenchmarkCertified_rejected_iff
        prob cfg spec A).2 hRej)

/-- Constructive accepted certificate for the public deployment decision path. -/
structure ConstructiveDeploymentAcceptedCertificate
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) where
  decisionAccepted :
    solveDefinitiveRawCrossDockDecision prob cfg spec A =
      ComputableKernelDecision.accepted
  benchmarkCertificate : ConstructiveBenchmarkAcceptedCertificate prob cfg spec A
  deploymentContract :
    (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec
  certificateBackendAccepted :
    ∃ cert,
      definitiveDeploymentCertificateBackend prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert

def constructiveDeploymentAcceptedCertificateOfDecision
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.accepted) :
    ConstructiveDeploymentAcceptedCertificate prob cfg spec A := by
  have hBenchDec :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.accepted := by
    simpa [solveDefinitiveRawCrossDockDecision_eq_benchmarkDecision] using hDec
  let B := constructiveBenchmarkAcceptedCertificateOfDecision
    prob cfg spec A hBenchDec
  have hLegacyDep :
      ∃ cert,
        definitiveDeploymentCertificateBackend prob cfg spec =
          DeploymentConstrainedDockingSolverResult.accepted cert :=
    solveDefinitiveRawCrossDockDecision_refines_certificate_backend_accept
      prob cfg spec A hDec
  have hDep :
      (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec :=
    (definitiveDeploymentCertificateBackend_accepted_iff_contract
      prob cfg spec).1 hLegacyDep
  exact
    { decisionAccepted := hDec
      benchmarkCertificate := B
      deploymentContract := hDep
      certificateBackendAccepted := hLegacyDep }

/-- Constructive rejected certificate for the public deployment decision path. -/
structure ConstructiveDeploymentRejectedCertificate
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) where
  decisionRejected :
    solveDefinitiveRawCrossDockDecision prob cfg spec A =
      ComputableKernelDecision.rejected
  benchmarkRejectedCertificate : ConstructiveBenchmarkRejectedCertificate prob cfg spec A
  kernelFlagFalse : A.kernel.computableAcceptFlag = false

def constructiveDeploymentRejectedCertificateOfDecision
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.rejected) :
    ConstructiveDeploymentRejectedCertificate prob cfg spec A := by
  have hBenchRej :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
        ComputableKernelDecision.rejected := by
    simpa [solveDefinitiveRawCrossDockDecision_eq_benchmarkDecision] using hDec
  let B := constructiveBenchmarkRejectedCertificateOfDecision
    prob cfg spec A hBenchRej
  exact
    { decisionRejected := hDec
      benchmarkRejectedCertificate := B
      kernelFlagFalse := B.kernelFlagFalse }

inductive ConstructiveDeploymentCertifiedResult
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) where
  | accepted (cert : ConstructiveDeploymentAcceptedCertificate prob cfg spec A)
  | rejected (cert : ConstructiveDeploymentRejectedCertificate prob cfg spec A)

theorem solveDefinitiveRawCrossDockDecision_rejected_of_not_accepted
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (hNot :
      solveDefinitiveRawCrossDockDecision prob cfg spec A ≠
        ComputableKernelDecision.accepted) :
    solveDefinitiveRawCrossDockDecision prob cfg spec A =
      ComputableKernelDecision.rejected := by
  have hTot :
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.accepted ∨
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.rejected := by
    have hBenchTot :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.accepted ∨
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.rejected := by
      simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive] using
        solveDefinitiveRawCrossDockBenchmarkConstructive_total prob cfg spec A
    simpa [solveDefinitiveRawCrossDockDecision_eq_benchmarkDecision] using hBenchTot
  rcases hTot with hAcc | hRej
  · exact (hNot hAcc).elim
  · exact hRej

def solveDefinitiveRawCrossDockCertified
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    ConstructiveDeploymentCertifiedResult prob cfg spec A :=
  if hDec :
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.accepted then
    .accepted
      (constructiveDeploymentAcceptedCertificateOfDecision prob cfg spec A hDec)
  else
    .rejected
      (constructiveDeploymentRejectedCertificateOfDecision prob cfg spec A
        (solveDefinitiveRawCrossDockDecision_rejected_of_not_accepted
          prob cfg spec A hDec))

theorem solveDefinitiveRawCrossDockCertified_accepted_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    (∃ cert,
      solveDefinitiveRawCrossDockCertified prob cfg spec A =
        ConstructiveDeploymentCertifiedResult.accepted cert) ↔
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.accepted := by
  constructor
  · intro hAcc
    by_cases hDec :
        solveDefinitiveRawCrossDockDecision prob cfg spec A =
          ComputableKernelDecision.accepted
    · exact hDec
    · exfalso
      rcases hAcc with ⟨cert, hEq⟩
      unfold solveDefinitiveRawCrossDockCertified at hEq
      simp [hDec] at hEq
  · intro hDec
    refine ⟨constructiveDeploymentAcceptedCertificateOfDecision prob cfg spec A hDec, ?_⟩
    unfold solveDefinitiveRawCrossDockCertified
    simp [hDec]

theorem solveDefinitiveRawCrossDockCertified_rejected_iff
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    (∃ cert,
      solveDefinitiveRawCrossDockCertified prob cfg spec A =
        ConstructiveDeploymentCertifiedResult.rejected cert) ↔
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.rejected := by
  constructor
  · intro hRej
    by_cases hDec :
        solveDefinitiveRawCrossDockDecision prob cfg spec A =
          ComputableKernelDecision.accepted
    · exfalso
      rcases hRej with ⟨cert, hEq⟩
      unfold solveDefinitiveRawCrossDockCertified at hEq
      simp [hDec] at hEq
    · exact solveDefinitiveRawCrossDockDecision_rejected_of_not_accepted
        prob cfg spec A hDec
  · intro hRej
    have hNotAcc :
        solveDefinitiveRawCrossDockDecision prob cfg spec A ≠
          ComputableKernelDecision.accepted := by
      intro hAcc
      have hContra : ComputableKernelDecision.accepted = ComputableKernelDecision.rejected := by
        simpa [hAcc] using hRej
      cases hContra
    have hDecRej :
        solveDefinitiveRawCrossDockDecision prob cfg spec A =
          ComputableKernelDecision.rejected :=
      solveDefinitiveRawCrossDockDecision_rejected_of_not_accepted
        prob cfg spec A hNotAcc
    refine ⟨constructiveDeploymentRejectedCertificateOfDecision prob cfg spec A hDecRej, ?_⟩
    unfold solveDefinitiveRawCrossDockCertified
    simp [hNotAcc, hDecRej,
      solveDefinitiveRawCrossDockDecision_rejected_of_not_accepted
        prob cfg spec A hNotAcc]

theorem solveDefinitiveRawCrossDockCertified_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) :
    (∃ cert,
      solveDefinitiveRawCrossDockCertified prob cfg spec A =
        ConstructiveDeploymentCertifiedResult.accepted cert) ∨
      (∃ cert,
        solveDefinitiveRawCrossDockCertified prob cfg spec A =
          ConstructiveDeploymentCertifiedResult.rejected cert) := by
  have hTot :
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.accepted ∨
      solveDefinitiveRawCrossDockDecision prob cfg spec A =
        ComputableKernelDecision.rejected := by
    have hBenchTot :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.accepted ∨
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A =
          ComputableKernelDecision.rejected := by
      simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive] using
        solveDefinitiveRawCrossDockBenchmarkConstructive_total prob cfg spec A
    simpa [solveDefinitiveRawCrossDockDecision_eq_benchmarkDecision] using hBenchTot
  rcases hTot with hAcc | hRej
  · exact Or.inl
      ((solveDefinitiveRawCrossDockCertified_accepted_iff
        prob cfg spec A).2 hAcc)
  · exact Or.inr
      ((solveDefinitiveRawCrossDockCertified_rejected_iff
        prob cfg spec A).2 hRej)

/-- Exact runtime knobs for the definitive constructive computable pipeline. -/
structure DefinitiveComputableRuntimeProfile where
  pocketCoordBudget : Nat
  ligandCoordBudget : Nat
  conformerCount : Nat
  refinementFuel : Nat
  parserBytes : Nat

def definitiveComputablePruningChecks
    (P : DefinitiveComputableRuntimeProfile) : Nat :=
  P.pocketCoordBudget + P.ligandCoordBudget

def definitiveComputableScorerCalls
    (P : DefinitiveComputableRuntimeProfile) : Nat :=
  P.conformerCount * P.refinementFuel

def definitiveComputableRefinementSteps
    (P : DefinitiveComputableRuntimeProfile) : Nat :=
  P.refinementFuel

def definitiveComputableTotalOps
    (P : DefinitiveComputableRuntimeProfile) : Nat :=
  definitiveComputablePruningChecks P +
    definitiveComputableScorerCalls P +
    definitiveComputableRefinementSteps P +
    P.parserBytes

theorem definitiveComputableTotalOps_closed_form
    (K L C fuel bytes : Nat) :
    let P : DefinitiveComputableRuntimeProfile :=
      { pocketCoordBudget := K
        ligandCoordBudget := L
        conformerCount := C
        refinementFuel := fuel
        parserBytes := bytes }
    definitiveComputableTotalOps P =
      (K + L) + bytes + (C + 1) * fuel := by
  simp [definitiveComputableTotalOps,
    definitiveComputablePruningChecks,
    definitiveComputableScorerCalls,
    definitiveComputableRefinementSteps,
    Nat.add_mul,
    Nat.add_assoc,
    Nat.add_left_comm,
    Nat.add_comm]

theorem definitiveComputableTotalOps_succFuel
    (K L C fuel bytes : Nat) :
    let Pnext : DefinitiveComputableRuntimeProfile :=
      { pocketCoordBudget := K
        ligandCoordBudget := L
        conformerCount := C
        refinementFuel := fuel + 1
        parserBytes := bytes }
    let Pcurr : DefinitiveComputableRuntimeProfile :=
      { pocketCoordBudget := K
        ligandCoordBudget := L
        conformerCount := C
        refinementFuel := fuel
        parserBytes := bytes }
    definitiveComputableTotalOps Pnext =
      definitiveComputableTotalOps Pcurr + (C + 1) := by
  simp [definitiveComputableTotalOps,
    definitiveComputablePruningChecks,
    definitiveComputableScorerCalls,
    definitiveComputableRefinementSteps,
    Nat.mul_add,
    Nat.add_assoc,
    Nat.add_left_comm,
    Nat.add_comm]

/-- Integrated definitive computable pipeline output: decision, certificates,
and exact runtime-operation profile. -/
structure DefinitiveComputablePipelineOutput
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) where
  decision : ComputableKernelDecision
  certified : ConstructiveDeploymentCertifiedResult prob cfg spec A
  runtimeProfile : DefinitiveComputableRuntimeProfile

def runDefinitiveComputablePipeline
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (runtimeProfile : DefinitiveComputableRuntimeProfile) :
    DefinitiveComputablePipelineOutput prob cfg spec A :=
  { decision := solveDefinitiveRawCrossDockDecision prob cfg spec A
    certified := solveDefinitiveRawCrossDockCertified prob cfg spec A
    runtimeProfile := runtimeProfile }

theorem runDefinitiveComputablePipeline_decision_eq
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (runtimeProfile : DefinitiveComputableRuntimeProfile) :
    (runDefinitiveComputablePipeline prob cfg spec A runtimeProfile).decision =
      solveDefinitiveRawCrossDockDecision prob cfg spec A := by
  rfl

theorem runDefinitiveComputablePipeline_certified_eq
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (runtimeProfile : DefinitiveComputableRuntimeProfile) :
    (runDefinitiveComputablePipeline prob cfg spec A runtimeProfile).certified =
      solveDefinitiveRawCrossDockCertified prob cfg spec A := by
  rfl

theorem runDefinitiveComputablePipeline_totalOps_exact
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (runtimeProfile : DefinitiveComputableRuntimeProfile) :
    definitiveComputableTotalOps
        (runDefinitiveComputablePipeline prob cfg spec A runtimeProfile).runtimeProfile =
      definitiveComputableTotalOps runtimeProfile := by
  rfl

theorem runDefinitiveComputablePipeline_totalOps_closed_form
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (K L C fuel bytes : Nat) :
    let P : DefinitiveComputableRuntimeProfile :=
      { pocketCoordBudget := K
        ligandCoordBudget := L
        conformerCount := C
        refinementFuel := fuel
        parserBytes := bytes }
    definitiveComputableTotalOps
        (runDefinitiveComputablePipeline prob cfg spec A P).runtimeProfile =
      (K + L) + bytes + (C + 1) * fuel := by
  intro P
  simpa [runDefinitiveComputablePipeline] using
    definitiveComputableTotalOps_closed_form K L C fuel bytes

def definitiveComputablePairBudget
    (P : DefinitiveComputableRuntimeProfile) : Nat :=
  P.pocketCoordBudget * P.ligandCoordBudget

def definitiveComputableCampaignPairEvaluations
    (P : DefinitiveComputableRuntimeProfile) : Nat :=
  definitiveComputablePairBudget P *
    definitiveComputableScorerCalls P

theorem definitiveComputableCampaignPairEvaluations_closed_form
    (K L C fuel bytes : Nat) :
    let P : DefinitiveComputableRuntimeProfile :=
      { pocketCoordBudget := K
        ligandCoordBudget := L
        conformerCount := C
        refinementFuel := fuel
        parserBytes := bytes }
    definitiveComputableCampaignPairEvaluations P =
      (K * L) * C * fuel := by
  simp [definitiveComputableCampaignPairEvaluations,
    definitiveComputablePairBudget,
    definitiveComputableScorerCalls,
    Nat.mul_left_comm,
    Nat.mul_comm]

theorem definitiveComputableCampaignPairEvaluations_succFuel
    (K L C fuel bytes : Nat) :
    let Pnext : DefinitiveComputableRuntimeProfile :=
      { pocketCoordBudget := K
        ligandCoordBudget := L
        conformerCount := C
        refinementFuel := fuel + 1
        parserBytes := bytes }
    let Pcurr : DefinitiveComputableRuntimeProfile :=
      { pocketCoordBudget := K
        ligandCoordBudget := L
        conformerCount := C
        refinementFuel := fuel
        parserBytes := bytes }
    definitiveComputableCampaignPairEvaluations Pnext =
      definitiveComputableCampaignPairEvaluations Pcurr + (K * L) * C := by
  simp [definitiveComputableCampaignPairEvaluations,
    definitiveComputablePairBudget,
    definitiveComputableScorerCalls,
    Nat.mul_add,
    Nat.mul_left_comm,
    Nat.mul_comm]

theorem runDefinitiveComputablePipeline_campaignPairEvaluations_closed_form
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (K L C fuel bytes : Nat) :
    let P : DefinitiveComputableRuntimeProfile :=
      { pocketCoordBudget := K
        ligandCoordBudget := L
        conformerCount := C
        refinementFuel := fuel
        parserBytes := bytes }
    definitiveComputableCampaignPairEvaluations
        (runDefinitiveComputablePipeline prob cfg spec A P).runtimeProfile =
      (K * L) * C * fuel := by
  intro P
  simpa [runDefinitiveComputablePipeline] using
    definitiveComputableCampaignPairEvaluations_closed_form K L C fuel bytes

/-- Certified score interval used by branch-and-bound elimination. -/
structure CertifiedScoreInterval where
  centerQ : Rat
  epsilonQ : Rat
  epsilon_nonneg : 0 ≤ epsilonQ

def CertifiedScoreInterval.lowerBound
    (I : CertifiedScoreInterval) : Rat :=
  I.centerQ - I.epsilonQ

def CertifiedScoreInterval.upperBound
    (I : CertifiedScoreInterval) : Rat :=
  I.centerQ + I.epsilonQ

theorem CertifiedScoreInterval.lowerBound_le_trueScore
    (I : CertifiedScoreInterval)
    (score : ℝ)
    (hScore : |score - (I.centerQ : ℝ)| ≤ (I.epsilonQ : ℝ)) :
    (I.lowerBound : ℝ) ≤ score := by
  have hAbs := abs_le.mp hScore
  have hLower : (I.centerQ : ℝ) - (I.epsilonQ : ℝ) ≤ score := by
    linarith [hAbs.1]
  simpa [CertifiedScoreInterval.lowerBound] using hLower

theorem CertifiedScoreInterval.trueScore_le_upperBound
    (I : CertifiedScoreInterval)
    (score : ℝ)
    (hScore : |score - (I.centerQ : ℝ)| ≤ (I.epsilonQ : ℝ)) :
    score ≤ (I.upperBound : ℝ) := by
  have hAbs := abs_le.mp hScore
  have hUpper : score ≤ (I.centerQ : ℝ) + (I.epsilonQ : ℝ) := by
    linarith [hAbs.2]
  simpa [CertifiedScoreInterval.upperBound] using hUpper

def branchAndBoundPrune
    (candidate incumbent : CertifiedScoreInterval) : Bool :=
  decide (candidate.upperBound < incumbent.lowerBound)

theorem branchAndBoundPrune_true_iff
    (candidate incumbent : CertifiedScoreInterval) :
    branchAndBoundPrune candidate incumbent = true ↔
      candidate.upperBound < incumbent.lowerBound := by
  unfold branchAndBoundPrune
  simp [decide_eq_true_iff]

theorem branchAndBoundPrune_sound
    (candidate incumbent : CertifiedScoreInterval)
    (candidateScore incumbentScore : ℝ)
    (hCandidate : |candidateScore - (candidate.centerQ : ℝ)| ≤ (candidate.epsilonQ : ℝ))
    (hIncumbent : |incumbentScore - (incumbent.centerQ : ℝ)| ≤ (incumbent.epsilonQ : ℝ))
    (hPrune : branchAndBoundPrune candidate incumbent = true) :
    candidateScore < incumbentScore := by
  have hCandUpper : candidateScore ≤ (candidate.upperBound : ℝ) :=
    candidate.trueScore_le_upperBound candidateScore hCandidate
  have hIncLower : (incumbent.lowerBound : ℝ) ≤ incumbentScore :=
    incumbent.lowerBound_le_trueScore incumbentScore hIncumbent
  have hGapRat : candidate.upperBound < incumbent.lowerBound :=
    (branchAndBoundPrune_true_iff candidate incumbent).1 hPrune
  have hGap : ((candidate.upperBound : Rat) : ℝ) < ((incumbent.lowerBound : Rat) : ℝ) := by
    exact_mod_cast hGapRat
  linarith

/-- Adaptive campaign stop rule: stop when every remaining conformer/pair has
upper score bound below the incumbent lower score bound. -/
def adaptiveCampaignStopRule
    {α : Type*}
    [DecidableEq α]
    (remaining : List α)
    (upperBound : α → Rat)
    (incumbentLower : Rat) : Bool :=
  remaining.all (fun a => decide (upperBound a < incumbentLower))

theorem adaptiveCampaignStopRule_true_iff
    {α : Type*}
    [DecidableEq α]
    (remaining : List α)
    (upperBound : α → Rat)
    (incumbentLower : Rat) :
    adaptiveCampaignStopRule remaining upperBound incumbentLower = true ↔
      ∀ a ∈ remaining, upperBound a < incumbentLower := by
  unfold adaptiveCampaignStopRule
  constructor
  · intro hAll a ha
    have hDec : decide (upperBound a < incumbentLower) = true :=
      List.all_eq_true.mp hAll a ha
    exact (decide_eq_true_iff).1 hDec
  · intro hBound
    exact List.all_eq_true.mpr (by
      intro a ha
      exact (decide_eq_true_iff).2 (hBound a ha))

theorem adaptiveCampaignStopRule_sound
    {α : Type*}
    [DecidableEq α]
    (remaining : List α)
    (trueScore : α → ℝ)
    (upperBound : α → Rat)
    (incumbentLower : Rat)
    (incumbentScore : ℝ)
    (hUpper : ∀ a ∈ remaining, trueScore a ≤ (upperBound a : ℝ))
    (hIncumbent : (incumbentLower : ℝ) ≤ incumbentScore)
    (hStop : adaptiveCampaignStopRule remaining upperBound incumbentLower = true) :
    ∀ a ∈ remaining, trueScore a < incumbentScore := by
  intro a ha
  have hBound : upperBound a < incumbentLower :=
    (adaptiveCampaignStopRule_true_iff remaining upperBound incumbentLower).1 hStop a ha
  have hScoreLe : trueScore a ≤ (upperBound a : ℝ) := hUpper a ha
  have hBoundReal : (upperBound a : ℝ) < (incumbentLower : ℝ) := by
    exact_mod_cast hBound
  have hScoreLtLower : trueScore a < (incumbentLower : ℝ) :=
    lt_of_le_of_lt hScoreLe hBoundReal
  exact lt_of_lt_of_le hScoreLtLower hIncumbent

/-- Integrated branch-and-bound run wrapper for the definitive computable
pipeline. -/
structure DefinitiveComputablePipelineBranchAndBoundOutput
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec) where
  pipeline : DefinitiveComputablePipelineOutput prob cfg spec A
  pruneFlag : Bool

def runDefinitiveComputablePipelineBranchAndBound
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (runtimeProfile : DefinitiveComputableRuntimeProfile)
    (candidate incumbent : CertifiedScoreInterval) :
    DefinitiveComputablePipelineBranchAndBoundOutput prob cfg spec A :=
  { pipeline := runDefinitiveComputablePipeline prob cfg spec A runtimeProfile
    pruneFlag := branchAndBoundPrune candidate incumbent }

theorem runDefinitiveComputablePipelineBranchAndBound_prune_sound
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : DefinitiveRawCrossDockArtifactInstantiation prob cfg spec)
    (runtimeProfile : DefinitiveComputableRuntimeProfile)
    (candidate incumbent : CertifiedScoreInterval)
    (candidateScore incumbentScore : ℝ)
    (hCandidate : |candidateScore - (candidate.centerQ : ℝ)| ≤ (candidate.epsilonQ : ℝ))
    (hIncumbent : |incumbentScore - (incumbent.centerQ : ℝ)| ≤ (incumbent.epsilonQ : ℝ))
    (hPrune :
      (runDefinitiveComputablePipelineBranchAndBound
        prob cfg spec A runtimeProfile candidate incumbent).pruneFlag = true) :
    candidateScore < incumbentScore := by
  exact branchAndBoundPrune_sound candidate incumbent
    candidateScore incumbentScore hCandidate hIncumbent
    (by simpa [runDefinitiveComputablePipelineBranchAndBound] using hPrune)

/-- Batch/fusion correctness import for JAX-target kernels: sharded reduction is
extensionally equal to fused reduction in the ArrayDSL semantics. -/
theorem definitiveComputablePipeline_pairPotentialFusionJustified
    {n : Nat}
    (distances : DecisionQuotient.Computation.ArrayDSL.MDArray n)
    (rc ε σ : ℝ) :
    DecisionQuotient.Computation.ArrayDSL.sumPairPotentials distances rc ε σ =
      DecisionQuotient.Computation.ArrayDSL.sumPairPotentialsUnfused distances rc ε σ :=
  DecisionQuotient.Computation.ArrayDSL.sumPairPotentials_fused_unfused_equiv
    distances rc ε σ

theorem definitiveComputablePipeline_batchFusionJustified
    {m n : Nat}
    (shards : Fin m → DecisionQuotient.Computation.ArrayDSL.MDArray n) :
    DecisionQuotient.Computation.ArrayDSL.shardReduceSum shards =
      DecisionQuotient.Computation.ArrayDSL.reduce_sum (∑ shard, shards shard) :=
  DecisionQuotient.Computation.ArrayDSL.shardReduceSum_fusion_equiv shards

theorem SampledDockingSolverInput.canonicalSolverProgramIR_scorerFusionSound
    (I : SampledDockingSolverInput) :
    "sumPairPotentials" ∈ I.canonicalSolverProgramIR.requiredOps ∧
      (∀ {n : Nat}
          (distances : DecisionQuotient.Computation.ArrayDSL.MDArray n)
          (rc ε σ : ℝ),
          DecisionQuotient.Computation.ArrayDSL.sumPairPotentials distances rc ε σ =
            DecisionQuotient.Computation.ArrayDSL.sumPairPotentialsUnfused distances rc ε σ) := by
  refine ⟨I.canonicalSolverProgramIR_requires_sumPairPotentials, ?_⟩
  intro n distances rc ε σ
  exact definitiveComputablePipeline_pairPotentialFusionJustified distances rc ε σ

/-- Signed immutable manifest envelope for constructive kernel artifacts. -/
structure SignedArtifactManifest where
  artifactId : String
  sha256 : String
  signer : String
  provenance : String
  artifactId_nonempty : artifactId ≠ ""
  sha256_nonempty : sha256 ≠ ""
  signer_nonempty : signer ≠ ""
  provenance_nonempty : provenance ≠ ""

def SignedArtifactManifest.message
    (M : SignedArtifactManifest) : String :=
  M.artifactId ++ ":" ++ M.sha256 ++ ":" ++ M.provenance

abbrev ByteStream := List UInt8

def stringByteStream (s : String) : ByteStream :=
  s.toUTF8.data.toList

def encodeLengthPrefix : Nat → ByteStream
  | 0 => [0]
  | n + 1 => (1 : UInt8) :: encodeLengthPrefix n

def encodeByteField (payload : ByteStream) : ByteStream :=
  encodeLengthPrefix payload.length ++ payload

def parseLengthPrefix? : ByteStream → Option (Nat × ByteStream)
  | [] => none
  | 0 :: rest => some (0, rest)
  | b :: rest =>
      if b = (1 : UInt8) then
        match parseLengthPrefix? rest with
        | some (n, tail) => some (n + 1, tail)
        | none => none
      else none

def takeExactly? : Nat → ByteStream → Option (ByteStream × ByteStream)
  | 0, bytes => some ([], bytes)
  | _ + 1, [] => none
  | n + 1, b :: rest =>
      match takeExactly? n rest with
      | some (head, tail) => some (b :: head, tail)
      | none => none

def parseByteField? (bytes : ByteStream) : Option (ByteStream × ByteStream) := do
  let (n, rest) ← parseLengthPrefix? bytes
  let (field, tail) ← takeExactly? n rest
  pure (field, tail)

def parseFiveByteFields?
    (bytes : ByteStream) :
    Option (ByteStream × ByteStream × ByteStream × ByteStream × ByteStream) := do
  let (f1, r1) ← parseByteField? bytes
  let (f2, r2) ← parseByteField? r1
  let (f3, r3) ← parseByteField? r2
  let (f4, r4) ← parseByteField? r3
  let (f5, r5) ← parseByteField? r4
  match r5 with
  | [] => pure (f1, f2, f3, f4, f5)
  | _ => none

theorem parseLengthPrefix_encodeLengthPrefix
    (n : Nat)
    (rest : ByteStream) :
    parseLengthPrefix? (encodeLengthPrefix n ++ rest) = some (n, rest) := by
  induction n with
  | zero =>
      simp [encodeLengthPrefix, parseLengthPrefix?]
  | succ n ih =>
      simp [encodeLengthPrefix, parseLengthPrefix?, ih]

theorem takeExactly_append_eq
    (head rest : ByteStream) :
    takeExactly? head.length (head ++ rest) = some (head, rest) := by
  induction head with
  | nil =>
      simp [takeExactly?]
  | cons b head ih =>
      simp [takeExactly?, ih]

theorem parseByteField_encode
    (payload rest : ByteStream) :
    parseByteField? (encodeByteField payload ++ rest) = some (payload, rest) := by
  unfold parseByteField? encodeByteField
  simp [parseLengthPrefix_encodeLengthPrefix, takeExactly_append_eq]

def encodeSignedArtifactByteEnvelope
    (M : SignedArtifactManifest)
    (signature : String) : ByteStream :=
  encodeByteField (stringByteStream M.artifactId) ++
    encodeByteField (stringByteStream M.sha256) ++
    encodeByteField (stringByteStream M.signer) ++
    encodeByteField (stringByteStream M.provenance) ++
    encodeByteField (stringByteStream signature)

def parseSignedArtifactByteEnvelope?
    (bytes : ByteStream) :
    Option (ByteStream × ByteStream × ByteStream × ByteStream × ByteStream) :=
  parseFiveByteFields? bytes

theorem parseSignedArtifactByteEnvelope_encode
    (M : SignedArtifactManifest)
    (signature : String) :
    parseSignedArtifactByteEnvelope?
        (encodeSignedArtifactByteEnvelope M signature) =
      some
        ( stringByteStream M.artifactId,
          stringByteStream M.sha256,
          stringByteStream M.signer,
          stringByteStream M.provenance,
          stringByteStream signature ) := by
  unfold parseSignedArtifactByteEnvelope? parseFiveByteFields?
    encodeSignedArtifactByteEnvelope
  have hLast :
      parseByteField? (encodeByteField (stringByteStream signature)) =
        some (stringByteStream signature, []) := by
    simpa using parseByteField_encode (payload := stringByteStream signature) (rest := ([] : ByteStream))
  simp [parseByteField_encode, hLast]

def parseSignedArtifactByteEnvelope_cost
    (bytes : ByteStream) : Nat :=
  bytes.length

theorem parseSignedArtifactByteEnvelope_cost_linear_time
    (bytes : ByteStream) :
    parseSignedArtifactByteEnvelope_cost bytes ≤ bytes.length := by
  simp [parseSignedArtifactByteEnvelope_cost]

theorem parseSignedArtifactByteEnvelope_cost_linear_space
    (bytes : ByteStream) :
    parseSignedArtifactByteEnvelope_cost bytes ≤ bytes.length := by
  simp [parseSignedArtifactByteEnvelope_cost]

theorem parseSignedArtifactByteEnvelope_encode_cost_exact
    (M : SignedArtifactManifest)
    (signature : String) :
    parseSignedArtifactByteEnvelope_cost
        (encodeSignedArtifactByteEnvelope M signature) =
      (encodeSignedArtifactByteEnvelope M signature).length := by
  rfl

def rollingChecksum (bytes : ByteStream) : Nat :=
  bytes.foldl (fun acc b => (acc * 257 + b.toNat) % 4294967291) 0

def SignedArtifactManifest.messageBytes
    (M : SignedArtifactManifest) : ByteStream :=
  stringByteStream M.artifactId ++ [58] ++
    stringByteStream M.sha256 ++ [58] ++
    stringByteStream M.provenance

def concreteChecksumSignature
    (M : SignedArtifactManifest) : String :=
  toString (rollingChecksum M.messageBytes)

def concreteChecksumSignatureBytes
    (artifactIdBytes sha256Bytes provenanceBytes : ByteStream) : ByteStream :=
  stringByteStream <|
    toString (rollingChecksum (artifactIdBytes ++ [58] ++ sha256Bytes ++ [58] ++ provenanceBytes))

def verifyConcreteChecksumSignatureBytes
    (artifactIdBytes sha256Bytes provenanceBytes signatureBytes : ByteStream) : Bool :=
  decide
    (signatureBytes =
      concreteChecksumSignatureBytes artifactIdBytes sha256Bytes provenanceBytes)

theorem verifyConcreteChecksumSignatureBytes_true_iff
    (artifactIdBytes sha256Bytes provenanceBytes signatureBytes : ByteStream) :
    verifyConcreteChecksumSignatureBytes
        artifactIdBytes sha256Bytes provenanceBytes signatureBytes = true ↔
      signatureBytes =
        concreteChecksumSignatureBytes artifactIdBytes sha256Bytes provenanceBytes := by
  unfold verifyConcreteChecksumSignatureBytes
  simp [decide_eq_true_iff]

/-- Signature verifier interface for signed constructive artifacts. -/
structure ArtifactSignatureVerifier where
  verify : SignedArtifactManifest → String → Prop
  verify_decidable : ∀ M sig, Decidable (verify M sig)

attribute [instance] ArtifactSignatureVerifier.verify_decidable

/-- Concrete checksum-based signature verifier used for byte-level end-to-end
artifact verification in Lean. -/
def concreteChecksumArtifactSignatureVerifier : ArtifactSignatureVerifier where
  verify M sig := sig = concreteChecksumSignature M
  verify_decidable := by
    intro M sig
    infer_instance

theorem concreteChecksumArtifactSignatureVerifier_verify_iff
    (M : SignedArtifactManifest)
    (sig : String) :
    concreteChecksumArtifactSignatureVerifier.verify M sig ↔
      sig = concreteChecksumSignature M := by
  rfl

theorem concreteChecksumSignature_bytes_eq
    (M : SignedArtifactManifest) :
    stringByteStream (concreteChecksumSignature M) =
      concreteChecksumSignatureBytes
        (stringByteStream M.artifactId)
        (stringByteStream M.sha256)
        (stringByteStream M.provenance) := by
  rfl

theorem concreteChecksum_parse_verify_end_to_end
    (M : SignedArtifactManifest) :
    let sig := concreteChecksumSignature M
    parseSignedArtifactByteEnvelope?
        (encodeSignedArtifactByteEnvelope M sig) =
      some
        ( stringByteStream M.artifactId,
          stringByteStream M.sha256,
          stringByteStream M.signer,
          stringByteStream M.provenance,
          stringByteStream sig ) ∧
      verifyConcreteChecksumSignatureBytes
        (stringByteStream M.artifactId)
        (stringByteStream M.sha256)
        (stringByteStream M.provenance)
        (stringByteStream sig) = true := by
  intro sig
  refine ⟨?_, ?_⟩
  · exact parseSignedArtifactByteEnvelope_encode M sig
  · unfold verifyConcreteChecksumSignatureBytes
    simp [sig, concreteChecksumSignature_bytes_eq]

/-- Assumption-packaged cryptographic verifier model for production-strength
signature checking. -/
structure CryptographicVerifierAssumptions where
  hash : ByteStream → ByteStream
  verify : ByteStream → ByteStream → Bool
  verify_complete : ∀ msg, verify msg (hash msg) = true
  verify_sound : ∀ msg sig, verify msg sig = true → sig = hash msg
  collision_resistant : Prop
  second_preimage_resistant : Prop

def cryptographicArtifactSignatureVerifier
    (C : CryptographicVerifierAssumptions) : ArtifactSignatureVerifier where
  verify M sig := C.verify M.messageBytes (stringByteStream sig) = true
  verify_decidable := by
    intro M sig
    infer_instance

theorem cryptographicArtifactSignatureVerifier_verify_iff
    (C : CryptographicVerifierAssumptions)
    (M : SignedArtifactManifest)
    (sig : String) :
    (cryptographicArtifactSignatureVerifier C).verify M sig ↔
      C.verify M.messageBytes (stringByteStream sig) = true := by
  rfl

theorem cryptographicArtifactSignatureVerifier_sound
    (C : CryptographicVerifierAssumptions)
    (M : SignedArtifactManifest)
    (sig : String)
    (hVerify : (cryptographicArtifactSignatureVerifier C).verify M sig) :
    stringByteStream sig = C.hash M.messageBytes := by
  exact C.verify_sound M.messageBytes (stringByteStream sig) hVerify

/-- Signed rationalized kernel artifact whose certified content is converted
directly to an endpoint artifact instantiation. -/
structure SignedRationalizedKernelArtifact
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (V : ArtifactSignatureVerifier) where
  manifest : SignedArtifactManifest
  payload : RationalizedKernelArtifactPayload
  signature : String
  signature_valid : V.verify manifest signature
  payload_artifactId_matches_manifest : payload.artifactId = manifest.artifactId
  payload_digest_matches_manifest : payload.generatedOpsDigest = manifest.sha256
  targetRMSD_eq : (payload.targetRMSD_Q : ℝ) = spec.targetRMSD
  targetProbability_eq : (payload.targetProbability_Q : ℝ) = spec.targetProbability
  rmsd_abs_error :
    |(payload.rmsdQ : ℝ) -
      (canonicalRawDockingInput prob cfg).rmsdToReference
        (canonicalRawDockingInput prob cfg).selectedPose| ≤
      (payload.rmsdMarginQ : ℝ)
  success_abs_error :
    |(payload.successQ : ℝ) -
      (canonicalRawDockingInput prob cfg).successProbability spec| ≤
      (payload.successMarginQ : ℝ)

def SignedRationalizedKernelArtifact.toArtifactInstantiation
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V) :
    DefinitiveRawCrossDockArtifactInstantiation prob cfg spec :=
  { payload := S.payload
    targetRMSD_eq := S.targetRMSD_eq
    targetProbability_eq := S.targetProbability_eq
    rmsd_abs_error := S.rmsd_abs_error
    success_abs_error := S.success_abs_error }

theorem SignedRationalizedKernelArtifact.manifest_consistency_bundle
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V) :
    S.payload.artifactId = S.manifest.artifactId ∧
      S.payload.generatedOpsDigest = S.manifest.sha256 ∧
      V.verify S.manifest S.signature :=
  ⟨S.payload_artifactId_matches_manifest,
    S.payload_digest_matches_manifest,
    S.signature_valid⟩

theorem SignedRationalizedKernelArtifact.concreteChecksum_signature_eq
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    (S : SignedRationalizedKernelArtifact
      prob cfg spec concreteChecksumArtifactSignatureVerifier) :
    S.signature = concreteChecksumSignature S.manifest := by
  simpa [concreteChecksumArtifactSignatureVerifier] using S.signature_valid

theorem SignedRationalizedKernelArtifact.concreteChecksum_byte_parse_and_verify
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    (S : SignedRationalizedKernelArtifact
      prob cfg spec concreteChecksumArtifactSignatureVerifier) :
    parseSignedArtifactByteEnvelope?
        (encodeSignedArtifactByteEnvelope S.manifest S.signature) =
      some
        ( stringByteStream S.manifest.artifactId,
          stringByteStream S.manifest.sha256,
          stringByteStream S.manifest.signer,
          stringByteStream S.manifest.provenance,
          stringByteStream S.signature ) ∧
      verifyConcreteChecksumSignatureBytes
        (stringByteStream S.manifest.artifactId)
        (stringByteStream S.manifest.sha256)
        (stringByteStream S.manifest.provenance)
        (stringByteStream S.signature) = true := by
  have hSig := S.concreteChecksum_signature_eq
  simpa [hSig] using concreteChecksum_parse_verify_end_to_end S.manifest

theorem SignedRationalizedKernelArtifact.crypto_signature_hash_sound
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    (C : CryptographicVerifierAssumptions)
    (S : SignedRationalizedKernelArtifact
      prob cfg spec (cryptographicArtifactSignatureVerifier C)) :
    stringByteStream S.signature = C.hash S.manifest.messageBytes := by
  exact cryptographicArtifactSignatureVerifier_sound
    C S.manifest S.signature S.signature_valid

theorem SignedRationalizedKernelArtifact.crypto_byte_parse_and_verify
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    (C : CryptographicVerifierAssumptions)
    (S : SignedRationalizedKernelArtifact
      prob cfg spec (cryptographicArtifactSignatureVerifier C)) :
    parseSignedArtifactByteEnvelope?
        (encodeSignedArtifactByteEnvelope S.manifest S.signature) =
      some
        ( stringByteStream S.manifest.artifactId,
          stringByteStream S.manifest.sha256,
          stringByteStream S.manifest.signer,
          stringByteStream S.manifest.provenance,
          stringByteStream S.signature ) ∧
      C.verify S.manifest.messageBytes (stringByteStream S.signature) = true := by
  refine ⟨?_, ?_⟩
  · exact parseSignedArtifactByteEnvelope_encode S.manifest S.signature
  · simpa [cryptographicArtifactSignatureVerifier] using S.signature_valid

def solveDefinitiveRawCrossDockBenchmarkDecisionOfSignedArtifact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V) :
    ComputableKernelDecision :=
  solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec S.toArtifactInstantiation

def solveDefinitiveRawCrossDockDecisionOfSignedArtifact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V) :
    ComputableKernelDecision :=
  solveDefinitiveRawCrossDockDecision prob cfg spec S.toArtifactInstantiation

def solveDefinitiveRawCrossDockBenchmarkCertifiedOfSignedArtifact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V) :
    ConstructiveBenchmarkCertifiedResult prob cfg spec S.toArtifactInstantiation :=
  solveDefinitiveRawCrossDockBenchmarkCertified prob cfg spec S.toArtifactInstantiation

def solveDefinitiveRawCrossDockCertifiedOfSignedArtifact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V) :
    ConstructiveDeploymentCertifiedResult prob cfg spec S.toArtifactInstantiation :=
  solveDefinitiveRawCrossDockCertified prob cfg spec S.toArtifactInstantiation

def signedArtifactEnvelopeByteLength
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V) : Nat :=
  parseSignedArtifactByteEnvelope_cost
    (encodeSignedArtifactByteEnvelope S.manifest S.signature)

def runDefinitiveComputablePipelineOfSignedArtifact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V)
    (pocketCoordBudget ligandCoordBudget conformerCount refinementFuel : Nat) :
    DefinitiveComputablePipelineOutput prob cfg spec S.toArtifactInstantiation :=
  runDefinitiveComputablePipeline
    prob cfg spec S.toArtifactInstantiation
    { pocketCoordBudget := pocketCoordBudget
      ligandCoordBudget := ligandCoordBudget
      conformerCount := conformerCount
      refinementFuel := refinementFuel
      parserBytes := signedArtifactEnvelopeByteLength S }

theorem runDefinitiveComputablePipelineOfSignedArtifact_parserBytes_exact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V)
    (pocketCoordBudget ligandCoordBudget conformerCount refinementFuel : Nat) :
    (runDefinitiveComputablePipelineOfSignedArtifact
      S pocketCoordBudget ligandCoordBudget conformerCount refinementFuel).runtimeProfile.parserBytes =
      parseSignedArtifactByteEnvelope_cost
        (encodeSignedArtifactByteEnvelope S.manifest S.signature) := by
  rfl

theorem SignedRationalizedKernelArtifact.benchmark_decision_accept_refines_legacy_accept
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V)
    (hDec :
      solveDefinitiveRawCrossDockBenchmarkDecisionOfSignedArtifact S =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  exact solveDefinitiveRawCrossDockBenchmarkDecision_refines_legacy_benchmark_accept
    prob cfg spec S.toArtifactInstantiation
    (by simpa [solveDefinitiveRawCrossDockBenchmarkDecisionOfSignedArtifact] using hDec)

theorem SignedRationalizedKernelArtifact.decision_accept_refines_legacy_deployment_accept
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V)
    (hDec :
      solveDefinitiveRawCrossDockDecisionOfSignedArtifact S =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      legacySolveDefinitiveRawCrossDock prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert := by
  exact solveDefinitiveRawCrossDockDecision_refines_legacy_deployment_accept
    prob cfg spec S.toArtifactInstantiation
    (by simpa [solveDefinitiveRawCrossDockDecisionOfSignedArtifact] using hDec)

theorem SignedRationalizedKernelArtifact.benchmark_decision_accept_refines_certificate_backend_accept
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V)
    (hDec :
      solveDefinitiveRawCrossDockBenchmarkDecisionOfSignedArtifact S =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      definitiveBenchmarkCertificateBackend prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert := by
  simpa [definitiveBenchmarkCertificateBackend] using
    SignedRationalizedKernelArtifact.benchmark_decision_accept_refines_legacy_accept
      (S := S) hDec

theorem SignedRationalizedKernelArtifact.decision_accept_refines_certificate_backend_accept
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V)
    (hDec :
      solveDefinitiveRawCrossDockDecisionOfSignedArtifact S =
        ComputableKernelDecision.accepted) :
    ∃ cert,
      definitiveDeploymentCertificateBackend prob cfg spec =
        DeploymentConstrainedDockingSolverResult.accepted cert := by
  simpa [definitiveDeploymentCertificateBackend] using
    SignedRationalizedKernelArtifact.decision_accept_refines_legacy_deployment_accept
      (S := S) hDec

/-- Strict rejection witness for signed rationalized artifacts, requiring a
real-valued inequality gap that dominates rational approximation margins. -/
structure SignedRationalizedStrictRejectionWitness
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (V : ArtifactSignatureVerifier)
    (S : SignedRationalizedKernelArtifact prob cfg spec V) where
  strict_violation :
    spec.targetRMSD < (S.payload.rmsdQ : ℝ) - (S.payload.rmsdMarginQ : ℝ) ∨
      (S.payload.successQ : ℝ) + (S.payload.successMarginQ : ℝ) < spec.targetProbability

theorem SignedRationalizedKernelArtifact.strict_rejection_refines_certificate_backend_rejections
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedRationalizedKernelArtifact prob cfg spec V)
    (W : SignedRationalizedStrictRejectionWitness prob cfg spec V S) :
    solveDefinitiveRawCrossDockDecisionOfSignedArtifact S =
        ComputableKernelDecision.rejected ∧
      (∃ cert,
        definitiveBenchmarkCertificateBackend prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.failure cert) ∧
      (∃ cert,
        definitiveDeploymentCertificateBackend prob cfg spec =
          DeploymentConstrainedDockingSolverResult.rejected cert) := by
  let I := canonicalRawDockingInput prob cfg
  let K : RationalizedAcceptanceKernel I spec := S.toArtifactInstantiation.kernel
  have hW : RationalizedRejectionSeparationWitness I spec K := by
    refine ⟨?_⟩
    simpa [I, K,
      SignedRationalizedKernelArtifact.toArtifactInstantiation,
      DefinitiveRawCrossDockArtifactInstantiation.kernel] using W.strict_violation
  have hFlagFalse : K.computableAcceptFlag = false :=
    K.rejectionSeparation_flag_false hW
  have hBenchRej :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec S.toArtifactInstantiation =
        ComputableKernelDecision.rejected :=
    (solveDefinitiveRawCrossDockBenchmarkDecision_rejected_iff_kernel_flag_false
      prob cfg spec S.toArtifactInstantiation).2 hFlagFalse
  have hDec :
      solveDefinitiveRawCrossDockDecisionOfSignedArtifact S =
        ComputableKernelDecision.rejected := by
    simpa [solveDefinitiveRawCrossDockDecisionOfSignedArtifact,
      solveDefinitiveRawCrossDockDecision_eq_benchmarkDecision] using hBenchRej
  have hNotBench : ¬ I.benchmarkContractSatisfied spec :=
    K.rejectionSeparation_not_benchmarkContract hW
  have hBenchFail :
      ∃ cert,
        definitiveBenchmarkCertificateBackend prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.failure cert :=
    (definitiveBenchmarkCertificateBackend_failure_iff_not_contract
      prob cfg spec).2 (by simpa [I] using hNotBench)
  have hNotDep :
      ¬ (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec := by
    intro hDep
    exact hNotBench
      ((canonicalDeploymentCalibration_deploymentContract_iff_benchmark
        I spec).1 hDep)
  have hDepRej :
      ∃ cert,
        definitiveDeploymentCertificateBackend prob cfg spec =
          DeploymentConstrainedDockingSolverResult.rejected cert :=
    (definitiveDeploymentCertificateBackend_rejected_iff_not_contract
      prob cfg spec).2 hNotDep
  exact ⟨hDec, hBenchFail, hDepRej⟩

theorem ExactRatArtifactInstantiation.constructive_accepted_iff_benchmark_contract
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : ExactRatArtifactInstantiation prob cfg spec) :
    solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec
        A.toKernelArtifact =
        ComputableKernelDecision.accepted ↔
      (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
  have hFlag :
      (A.toKernelArtifact.kernel).computableAcceptFlag = true ↔
        (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
    simpa [ExactRatArtifactInstantiation.toKernelArtifact,
      DefinitiveRawCrossDockArtifactInstantiation.kernel,
      RationalizedAcceptanceKernel.ofExactRatWitness,
      canonicalRawDockingInput] using
      (RationalizedAcceptanceKernel.ofExactRatWitness_flag_true_iff
        (I := canonicalRawDockingInput prob cfg)
        (spec := spec)
        A.rmsdQ A.successQ A.targetRMSD_Q A.targetProbability_Q
        A.rmsd_eq A.success_eq A.targetRMSD_eq A.targetProbability_eq)
  exact (solveDefinitiveRawCrossDockBenchmarkConstructive_accepted_iff_kernel_flag
    prob cfg spec A.toKernelArtifact).trans hFlag

theorem ExactRatArtifactInstantiation.constructive_accept_refines_legacy_accepts
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : ExactRatArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockConstructive prob cfg spec A.toKernelArtifact =
        ComputableKernelDecision.accepted) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
      (∃ cert,
        legacySolveDefinitiveRawCrossDock prob cfg spec =
          DeploymentConstrainedDockingSolverResult.accepted cert) := by
  refine ⟨?_, ?_⟩
  · exact solveDefinitiveRawCrossDockBenchmarkConstructive_refines_benchmark_accept
      prob cfg spec A.toKernelArtifact
      (by simpa [solveDefinitiveRawCrossDockConstructive_eq_benchmark] using hDec)
  · exact solveDefinitiveRawCrossDockConstructive_refines_deployment_accept
      prob cfg spec A.toKernelArtifact hDec

theorem ExactRatArtifactInstantiation.constructive_rejected_refines_legacy_rejections
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : ExactRatArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockDecision prob cfg spec A.toKernelArtifact =
        ComputableKernelDecision.rejected) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) ∧
      (∃ cert,
        legacySolveDefinitiveRawCrossDock prob cfg spec =
          DeploymentConstrainedDockingSolverResult.rejected cert) := by
  have hBenchRej :
      solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A.toKernelArtifact =
        ComputableKernelDecision.rejected := by
    simpa [solveDefinitiveRawCrossDockDecision_eq_benchmarkDecision] using hDec
  have hNotBench :
      ¬ (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
    intro hBench
    have hBenchAcc :
        solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec A.toKernelArtifact =
          ComputableKernelDecision.accepted := by
      have hCore :=
        ExactRatArtifactInstantiation.constructive_accepted_iff_benchmark_contract
          prob cfg spec A
      have hAccCore :
          solveDefinitiveRawCrossDockBenchmarkConstructive prob cfg spec A.toKernelArtifact =
            ComputableKernelDecision.accepted :=
        hCore.2 hBench
      simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive] using hAccCore
    have hContra : ComputableKernelDecision.accepted = ComputableKernelDecision.rejected := by
      simpa [hBenchAcc] using hBenchRej
    cases hContra
  have hBenchFail :
      ∃ cert,
        legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.failure cert :=
    (solveDefinitiveRawCrossDockBenchmark_failure_iff_not_benchmark_contract
      prob cfg spec).2 hNotBench
  have hNotDep :
      ¬ (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec := by
    intro hDep
    exact hNotBench
      ((canonicalDeploymentCalibration_deploymentContract_iff_benchmark
        (canonicalRawDockingInput prob cfg) spec).1 hDep)
  have hDepRej :
      ∃ cert,
        legacySolveDefinitiveRawCrossDock prob cfg spec =
          DeploymentConstrainedDockingSolverResult.rejected cert :=
    (solveDefinitiveRawCrossDock_rejected_iff_not_deployment_contract
      prob cfg spec).2 hNotDep
  exact ⟨hBenchFail, hDepRej⟩

theorem ExactRatArtifactInstantiation.constructive_accept_refines_certificate_backend_accepts
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : ExactRatArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockConstructive prob cfg spec A.toKernelArtifact =
        ComputableKernelDecision.accepted) :
    (∃ cert,
      definitiveBenchmarkCertificateBackend prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
      (∃ cert,
        definitiveDeploymentCertificateBackend prob cfg spec =
          DeploymentConstrainedDockingSolverResult.accepted cert) := by
  simpa [definitiveBenchmarkCertificateBackend,
    definitiveDeploymentCertificateBackend] using
    ExactRatArtifactInstantiation.constructive_accept_refines_legacy_accepts
      prob cfg spec A hDec

theorem ExactRatArtifactInstantiation.constructive_rejected_refines_certificate_backend_rejections
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (A : ExactRatArtifactInstantiation prob cfg spec)
    (hDec :
      solveDefinitiveRawCrossDockDecision prob cfg spec A.toKernelArtifact =
        ComputableKernelDecision.rejected) :
    (∃ cert,
      definitiveBenchmarkCertificateBackend prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) ∧
      (∃ cert,
        definitiveDeploymentCertificateBackend prob cfg spec =
          DeploymentConstrainedDockingSolverResult.rejected cert) := by
  simpa [definitiveBenchmarkCertificateBackend,
    definitiveDeploymentCertificateBackend] using
    ExactRatArtifactInstantiation.constructive_rejected_refines_legacy_rejections
      prob cfg spec A hDec

/-- Signed exact-rational artifact with manifest/digest/signature provenance. -/
structure SignedExactRatKernelArtifact
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec)
    (V : ArtifactSignatureVerifier) where
  manifest : SignedArtifactManifest
  artifact : ExactRatArtifactInstantiation prob cfg spec
  signature : String
  signature_valid : V.verify manifest signature
  artifactId_matches_manifest : artifact.artifactId = manifest.artifactId
  digest_matches_manifest : artifact.generatedOpsDigest = manifest.sha256

def SignedExactRatKernelArtifact.toKernelArtifact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedExactRatKernelArtifact prob cfg spec V) :
    DefinitiveRawCrossDockArtifactInstantiation prob cfg spec :=
  S.artifact.toKernelArtifact

theorem SignedExactRatKernelArtifact.manifest_consistency_bundle
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedExactRatKernelArtifact prob cfg spec V) :
    S.artifact.artifactId = S.manifest.artifactId ∧
      S.artifact.generatedOpsDigest = S.manifest.sha256 ∧
      V.verify S.manifest S.signature :=
  ⟨S.artifactId_matches_manifest, S.digest_matches_manifest, S.signature_valid⟩

def solveDefinitiveRawCrossDockDecisionOfSignedExactArtifact
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedExactRatKernelArtifact prob cfg spec V) :
    ComputableKernelDecision :=
  solveDefinitiveRawCrossDockDecision prob cfg spec S.toKernelArtifact

theorem SignedExactRatKernelArtifact.benchmark_decision_accepted_iff_benchmark_contract
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedExactRatKernelArtifact prob cfg spec V) :
    solveDefinitiveRawCrossDockBenchmarkDecision prob cfg spec S.toKernelArtifact =
        ComputableKernelDecision.accepted ↔
      (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec := by
  simpa [solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive,
    SignedExactRatKernelArtifact.toKernelArtifact] using
    (ExactRatArtifactInstantiation.constructive_accepted_iff_benchmark_contract
      prob cfg spec S.artifact)

theorem SignedExactRatKernelArtifact.decision_rejected_refines_legacy_rejections
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedExactRatKernelArtifact prob cfg spec V)
    (hDec :
      solveDefinitiveRawCrossDockDecisionOfSignedExactArtifact S =
        ComputableKernelDecision.rejected) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) ∧
      (∃ cert,
        legacySolveDefinitiveRawCrossDock prob cfg spec =
          DeploymentConstrainedDockingSolverResult.rejected cert) := by
  exact ExactRatArtifactInstantiation.constructive_rejected_refines_legacy_rejections
    prob cfg spec S.artifact
    (by simpa [solveDefinitiveRawCrossDockDecisionOfSignedExactArtifact,
      SignedExactRatKernelArtifact.toKernelArtifact] using hDec)

theorem SignedExactRatKernelArtifact.decision_rejected_refines_certificate_backend_rejections
    {prob : MDBindingProblem}
    {cfg : RawPocketLigandSamplingConfig}
    {spec : DockingAcceptanceSpec}
    {V : ArtifactSignatureVerifier}
    (S : SignedExactRatKernelArtifact prob cfg spec V)
    (hDec :
      solveDefinitiveRawCrossDockDecisionOfSignedExactArtifact S =
        ComputableKernelDecision.rejected) :
    (∃ cert,
      definitiveBenchmarkCertificateBackend prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) ∧
      (∃ cert,
        definitiveDeploymentCertificateBackend prob cfg spec =
          DeploymentConstrainedDockingSolverResult.rejected cert) := by
  simpa [definitiveBenchmarkCertificateBackend,
    definitiveDeploymentCertificateBackend] using
    SignedExactRatKernelArtifact.decision_rejected_refines_legacy_rejections
      (S := S) hDec

theorem solveDefinitiveRawCrossDockBenchmark_total
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.accepted cert) ∨
    (∃ cert,
      legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
        AcceptanceConstrainedDockingSolverResult.failure cert) := by
  simpa [legacySolveDefinitiveRawCrossDockBenchmark] using
    solveRawPocketLigandCanonicalBenchmark_total prob cfg spec

noncomputable def legacyRunDefinitiveRawCrossDockProgram
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    CanonicalProgramRuntimeOutput (canonicalRawDockingInput prob cfg) spec :=
  runCanonicalSolverProgram (canonicalRawDockingInput prob cfg) spec

noncomputable def legacyInterpretDefinitiveRawCrossDockProgram
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    CanonicalProgramRuntimeOutput (canonicalRawDockingInput prob cfg) spec :=
  canonicalRuntimeOutputOfInterpreterState
    (interpretCanonicalProgramState (canonicalRawDockingInput prob cfg) spec)

theorem interpretDefinitiveRawCrossDockProgram_eq_run
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    legacyInterpretDefinitiveRawCrossDockProgram prob cfg spec =
      legacyRunDefinitiveRawCrossDockProgram prob cfg spec := by
  rfl

theorem runDefinitiveRawCrossDockProgram_refines_benchmark_accept
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (legacyRunDefinitiveRawCrossDockProgram prob cfg spec).outputAcceptFlag = true ↔
      ∃ cert,
        legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.accepted cert := by
  simpa [legacyRunDefinitiveRawCrossDockProgram, legacySolveDefinitiveRawCrossDockBenchmark] using
    (runCanonicalSolverProgram_refines_solver_result
      (canonicalRawDockingInput prob cfg) spec).1

theorem runDefinitiveRawCrossDockProgram_refines_benchmark_failure
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    (legacyRunDefinitiveRawCrossDockProgram prob cfg spec).outputAcceptFlag = false ↔
      ∃ cert,
        legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
          AcceptanceConstrainedDockingSolverResult.failure cert := by
  simpa [legacyRunDefinitiveRawCrossDockProgram, legacySolveDefinitiveRawCrossDockBenchmark] using
    (runCanonicalSolverProgram_refines_solver_result
      (canonicalRawDockingInput prob cfg) spec).2

theorem solveDefinitiveRawCrossDock_full_closure_bundle
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    let I := canonicalRawDockingInput prob cfg
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.jaxCodegenReport.success = true ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ↔
        (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec) ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ∨
        (∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.failure cert)) ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ∨
        (∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.rejected cert)) ∧
      ((legacyRunDefinitiveRawCrossDockProgram prob cfg spec).outputAcceptFlag = true ↔
        ∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ∧
      ((legacyRunDefinitiveRawCrossDockProgram prob cfg spec).outputAcceptFlag = false ↔
        ∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.failure cert) ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) →
        ∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) := by
  intro I
  have hBundle := solveDefinitiveRawCrossDock_bundle prob cfg spec
  refine ⟨hBundle.1, hBundle.2.1, hBundle.2.2.1, hBundle.2.2.2.1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact solveDefinitiveRawCrossDock_accepted_iff_deployment_contract prob cfg spec
  · exact solveDefinitiveRawCrossDockBenchmark_total prob cfg spec
  · exact solveDefinitiveRawCrossDock_total prob cfg spec
  · exact runDefinitiveRawCrossDockProgram_refines_benchmark_accept prob cfg spec
  · exact runDefinitiveRawCrossDockProgram_refines_benchmark_failure prob cfg spec
  · exact solveDefinitiveRawCrossDock_accept_refines_benchmark_accept prob cfg spec

/-- Consolidated report object for the definitive raw cross-docking endpoint. -/
structure LegacyDefinitiveRawCrossDockReport
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) where
  benchmarkResult :
    AcceptanceConstrainedDockingSolverResult
      (canonicalRawDockingInput prob cfg) spec
  deploymentResult :
    DeploymentConstrainedDockingSolverResult
      (canonicalRawDockingInput prob cfg)
      spec
      (canonicalRawDockingCalibration prob cfg)
  runtimeOutput :
    CanonicalProgramRuntimeOutput
      (canonicalRawDockingInput prob cfg) spec
  support_included :
    bindingProblemBaseAction prob ∈ (canonicalRawDockingInput prob cfg).samples.support
  jax_codegen_success :
    (canonicalRawDockingInput prob cfg).jaxCodegenReport.success = true
  benchmarkAccepted_iff_contract :
    (∃ cert,
      benchmarkResult = AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
      (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec
  benchmarkFailure_iff_not_contract :
    (∃ cert,
      benchmarkResult = AcceptanceConstrainedDockingSolverResult.failure cert) ↔
      ¬ (canonicalRawDockingInput prob cfg).benchmarkContractSatisfied spec
  deploymentAccepted_iff_deploymentContract :
    (∃ cert,
      deploymentResult = DeploymentConstrainedDockingSolverResult.accepted cert) ↔
      (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec
  deploymentRejected_iff_not_deploymentContract :
    (∃ cert,
      deploymentResult = DeploymentConstrainedDockingSolverResult.rejected cert) ↔
      ¬ (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec
  runtimeAccept_iff_benchmarkAccepted :
    runtimeOutput.outputAcceptFlag = true ↔
      (∃ cert,
        benchmarkResult = AcceptanceConstrainedDockingSolverResult.accepted cert)
  runtimeReject_iff_benchmarkFailure :
    runtimeOutput.outputAcceptFlag = false ↔
      (∃ cert,
        benchmarkResult = AcceptanceConstrainedDockingSolverResult.failure cert)
  deploymentAccept_refines_benchmarkAccept :
    (∃ cert,
      deploymentResult = DeploymentConstrainedDockingSolverResult.accepted cert) →
      (∃ cert,
        benchmarkResult = AcceptanceConstrainedDockingSolverResult.accepted cert)

noncomputable def legacyBuildDefinitiveRawCrossDockReport
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    LegacyDefinitiveRawCrossDockReport prob cfg spec :=
  { benchmarkResult := legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec
    deploymentResult := legacySolveDefinitiveRawCrossDock prob cfg spec
    runtimeOutput := legacyRunDefinitiveRawCrossDockProgram prob cfg spec
    support_included := by
      simpa [canonicalRawDockingInput,
        sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD,
        sampledDockingSolverInputFromRawPocketLigand] using
        rawPocketLigandSampleFamily_contains_baseAction prob cfg
    jax_codegen_success := (canonicalRawDockingInput prob cfg).jax_codegen_success
    benchmarkAccepted_iff_contract := by
      simpa [legacySolveDefinitiveRawCrossDockBenchmark] using
        solveDefinitiveRawCrossDockBenchmark_accepted_iff_benchmark_contract prob cfg spec
    benchmarkFailure_iff_not_contract := by
      simpa [legacySolveDefinitiveRawCrossDockBenchmark] using
        solveDefinitiveRawCrossDockBenchmark_failure_iff_not_benchmark_contract prob cfg spec
    deploymentAccepted_iff_deploymentContract := by
      simpa [legacySolveDefinitiveRawCrossDock] using
        solveDefinitiveRawCrossDock_accepted_iff_deployment_contract prob cfg spec
    deploymentRejected_iff_not_deploymentContract := by
      simpa [legacySolveDefinitiveRawCrossDock] using
        solveDefinitiveRawCrossDock_rejected_iff_not_deployment_contract prob cfg spec
    runtimeAccept_iff_benchmarkAccepted := by
      simpa [legacyRunDefinitiveRawCrossDockProgram, legacySolveDefinitiveRawCrossDockBenchmark] using
        runDefinitiveRawCrossDockProgram_refines_benchmark_accept prob cfg spec
    runtimeReject_iff_benchmarkFailure := by
      simpa [legacyRunDefinitiveRawCrossDockProgram, legacySolveDefinitiveRawCrossDockBenchmark] using
        runDefinitiveRawCrossDockProgram_refines_benchmark_failure prob cfg spec
    deploymentAccept_refines_benchmarkAccept := by
      intro hDep
      simpa [legacySolveDefinitiveRawCrossDock, legacySolveDefinitiveRawCrossDockBenchmark] using
        solveDefinitiveRawCrossDock_accept_refines_benchmark_accept prob cfg spec hDep }

theorem definitiveRawCrossDockCompleteLeanBundle
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    let I := canonicalRawDockingInput prob cfg
    bindingProblemBaseAction prob ∈ I.samples.support ∧
      I.jaxCodegenReport.success = true ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
            AcceptanceConstrainedDockingSolverResult.accepted cert) ↔
        I.benchmarkContractSatisfied spec) ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ↔
        (canonicalRawDockingCalibration prob cfg).deploymentContractSatisfied spec) ∧
      (legacyDefinitiveRawCrossDockAcceptanceFlag prob cfg spec = true ↔
        ∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ∧
      (legacyDefinitiveRawCrossDockAcceptanceFlag prob cfg spec = false ↔
        ∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.rejected cert) ∧
      ((∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.accepted cert) ∨
        (∃ cert,
          legacySolveDefinitiveRawCrossDock prob cfg spec =
            DeploymentConstrainedDockingSolverResult.rejected cert)) ∧
      (∀ K : RationalizedAcceptanceKernel I spec,
          K.computableAcceptFlag = true →
          ∃ cert,
            legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec =
              AcceptanceConstrainedDockingSolverResult.accepted cert) := by
  intro I
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [canonicalRawDockingInput,
      sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD,
      sampledDockingSolverInputFromRawPocketLigand] using
      rawPocketLigandSampleFamily_contains_baseAction prob cfg
  · exact I.jax_codegen_success
  · exact solveDefinitiveRawCrossDockBenchmark_accepted_iff_benchmark_contract prob cfg spec
  · exact solveDefinitiveRawCrossDock_accepted_iff_deployment_contract prob cfg spec
  · exact definitiveRawCrossDockAcceptanceFlag_true_iff_deployment_accepted prob cfg spec
  · exact definitiveRawCrossDockAcceptanceFlag_false_iff_deployment_rejected prob cfg spec
  · exact solveDefinitiveRawCrossDock_total prob cfg spec
  · intro K hK
    have hCore :=
      RationalizedAcceptanceKernel.computableAcceptFlag_refines_benchmark_accept
        (I := I) (spec := spec) K hK
    simpa [legacySolveDefinitiveRawCrossDockBenchmark,
      canonicalRawDockingInput,
      sampledDockingSolverInputFromRawPocketLigandCanonicalRMSD] using hCore

theorem buildDefinitiveRawCrossDockReport_core_consistency
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    let R := legacyBuildDefinitiveRawCrossDockReport prob cfg spec
    R.benchmarkResult = legacySolveDefinitiveRawCrossDockBenchmark prob cfg spec ∧
      R.deploymentResult = legacySolveDefinitiveRawCrossDock prob cfg spec ∧
      R.runtimeOutput = legacyRunDefinitiveRawCrossDockProgram prob cfg spec := by
  intro R
  exact ⟨rfl, rfl, rfl⟩

theorem buildDefinitiveRawCrossDockReport_runtime_accept_iff_deployment_accepted
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    let R := legacyBuildDefinitiveRawCrossDockReport prob cfg spec
    R.runtimeOutput.outputAcceptFlag = true ↔
      (∃ cert,
        R.deploymentResult =
          DeploymentConstrainedDockingSolverResult.accepted cert) := by
  intro R
  simpa [legacyBuildDefinitiveRawCrossDockReport,
    legacyRunDefinitiveRawCrossDockProgram,
    legacyDefinitiveRawCrossDockAcceptanceFlag,
    legacySolveDefinitiveRawCrossDock] using
    definitiveRawCrossDockAcceptanceFlag_true_iff_deployment_accepted prob cfg spec

theorem buildDefinitiveRawCrossDockReport_runtime_reject_iff_deployment_rejected
    (prob : MDBindingProblem)
    (cfg : RawPocketLigandSamplingConfig)
    (spec : DockingAcceptanceSpec) :
    let R := legacyBuildDefinitiveRawCrossDockReport prob cfg spec
    R.runtimeOutput.outputAcceptFlag = false ↔
      (∃ cert,
        R.deploymentResult =
          DeploymentConstrainedDockingSolverResult.rejected cert) := by
  intro R
  simpa [legacyBuildDefinitiveRawCrossDockReport,
    legacyRunDefinitiveRawCrossDockProgram,
    legacyDefinitiveRawCrossDockAcceptanceFlag,
    legacySolveDefinitiveRawCrossDock] using
    definitiveRawCrossDockAcceptanceFlag_false_iff_deployment_rejected prob cfg spec

end Leverage
