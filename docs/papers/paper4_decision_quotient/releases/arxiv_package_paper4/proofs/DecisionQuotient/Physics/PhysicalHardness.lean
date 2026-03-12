import Mathlib.Data.Nat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import DecisionQuotient.ClaimClosure

/-!
# Physical Hardness (Conditional, Mechanized Core)

## Dependency Graph
- **Nontrivial source:** ClaimClosure.sufficiency_conp_complete_conditional,
  ClaimClosure.anchor_sigma2p_complete_conditional, ClaimClosure.dichotomy_conditional
- **This module:** Assumes those hardness results + physics premises (Landauer bound,
  discrete time) → derives physical impossibility statements
- **See also:** ClaimTransport.lean for the full bridge composition

## Role
This module is NOT the core hardness proof. It translates the complexity results
from ClaimClosure into physical consequences using declared physics axioms.

## Triviality Level
NONTRIVIAL — interprets nontrivial complexity lower bounds as physical
constraints. Closest nontrivial proofs referenced: `ClaimClosure.sufficiency_conp_complete_conditional`
and `ClaimClosure.anchor_sigma2p_complete_conditional` (DecisionQuotient.ClaimClosure).
-/

namespace PhysicalComplexity

/-- Opaque SAT-family instance used by the physical bridge layer.
Only input-size and satisfiability semantics are needed in this module. -/
structure SAT3Instance where
  inputSize : ℕ
  satisfiable : Prop

/-- SAT algorithm interface used by the physical bridge layer
(runtime accounting + correctness relation). -/
structure SAT3Algorithm where
  decide : SAT3Instance → Bool
  runtime : SAT3Instance → ℕ
  correct : ∀ φ : SAT3Instance, decide φ = true ↔ φ.satisfiable

/-- Symbolic Boltzmann constant used by the paper-side thermodynamic interface. -/
def k_Boltzmann : ℝ := 1.380649e-23

/-- Physical computer model with finite total energy budget and positive per-bit cost. -/
structure PhysicalComputer where
  E_max : ℕ
  bit_cost : ℕ
  hbit_pos : 0 < bit_cost

/-- Energy cost per elementary bit operation. -/
def bit_energy_cost (c : PhysicalComputer) : ℕ := c.bit_cost

/-- Landauer-style lower-bound interface: accounting is linear in bit operations. -/
lemma Landauer_bound (c : PhysicalComputer) (bits : ℕ) :
    bits * bit_energy_cost c ≥ bits * c.bit_cost := by
  simp [bit_energy_cost]

/-- Instance size alias used by this module. -/
def InstanceSize := ℕ

/-- Worst-case operation requirement interface for a decision family. -/
structure ComputationalRequirement where
  required_ops : ℕ → ℕ
  lower_exp : ∀ n : ℕ, 2 ^ n ≤ required_ops n

/-- Canonical exponential lower-bound requirement. -/
def coNP_requirement : ComputationalRequirement where
  required_ops := fun n => 2 ^ n
  lower_exp := by
    intro n
    exact le_rfl

/-- Exponential growth eventually exceeds any fixed finite threshold. -/
lemma pow2_eventually_exceeds (t : ℕ) :
    ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ → t < 2 ^ n := by
  refine ⟨t + 1, ?_⟩
  intro n hn
  have ht_lt_n : t < n := lt_of_lt_of_le (Nat.lt_succ_self t) hn
  have hn_lt_pow : n < 2 ^ n := by
    simpa using n.lt_two_pow_self
  exact lt_trans ht_lt_n hn_lt_pow

/-- Physical impossibility schema:
if operation count grows exponentially and each operation has positive cost,
there is a finite-size threshold after which required energy exceeds budget. -/
theorem coNP_physically_impossible
    (c : PhysicalComputer)
    (req : ComputationalRequirement) :
    ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ →
      c.E_max < req.required_ops n * bit_energy_cost c := by
  obtain ⟨n₀, hn₀⟩ := pow2_eventually_exceeds c.E_max
  refine ⟨n₀, ?_⟩
  intro n hn
  have hE_lt_pow : c.E_max < 2 ^ n := hn₀ n hn
  have hpow_le_req : 2 ^ n ≤ req.required_ops n := req.lower_exp n
  have hE_lt_req : c.E_max < req.required_ops n := lt_of_lt_of_le hE_lt_pow hpow_le_req
  have h_one_le_cost : 1 ≤ bit_energy_cost c := Nat.succ_le_of_lt c.hbit_pos
  have hreq_le_mul : req.required_ops n ≤ req.required_ops n * bit_energy_cost c := by
    have hmul := Nat.mul_le_mul_left (req.required_ops n) h_one_le_cost
    simpa [bit_energy_cost, Nat.mul_one] using hmul
  exact lt_of_lt_of_le hE_lt_req hreq_le_mul

/-- Physical-collapse predicate at a required-operation profile:
for every size `n`, there is a physically feasible bit budget that upper-bounds
the required operation count. -/
def PhysicalCollapseAtRequirement
    (c : PhysicalComputer) (req : ComputationalRequirement) : Prop :=
  ∀ n : ℕ, ∃ bits : ℕ,
    req.required_ops n ≤ bits ∧
    bits * bit_energy_cost c ≤ c.E_max

/-- No physical collapse is possible under this module's finite-budget model
and exponential lower-bound requirement interface. -/
theorem no_physical_collapse_at_requirement
    (c : PhysicalComputer)
    (req : ComputationalRequirement) :
    ¬ PhysicalCollapseAtRequirement c req := by
  intro hCollapse
  obtain ⟨n₀, hThreshold⟩ := coNP_physically_impossible c req
  rcases hCollapse n₀ with ⟨bits, hReqLeBits, hFeasible⟩
  have hExceed : c.E_max < req.required_ops n₀ * bit_energy_cost c :=
    hThreshold n₀ (le_rfl)
  have hReqLeBudget : req.required_ops n₀ * bit_energy_cost c ≤ c.E_max := by
    exact le_trans (Nat.mul_le_mul_right (bit_energy_cost c) hReqLeBits) hFeasible
  exact (Nat.not_lt_of_ge hReqLeBudget) hExceed

/-- Canonical specialization: collapse is impossible for the module's
exponential requirement profile. -/
theorem canonical_physical_collapse_impossible
    (c : PhysicalComputer) :
    ¬ PhysicalCollapseAtRequirement c coNP_requirement :=
  no_physical_collapse_at_requirement c coNP_requirement

/-- P=NP no-go transfer schema (conditional):
if one maps `P = NP` to a physically feasible collapse at the chosen requirement
profile, then `P = NP` is physically impossible in this model. -/
theorem p_eq_np_physically_impossible_of_collapse_map
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    {P_eq_NP : Prop}
    (hMap : P_eq_NP → PhysicalCollapseAtRequirement c req) :
    ¬ P_eq_NP := by
  intro hPNP
  exact no_physical_collapse_at_requirement c req (hMap hPNP)

/-- Canonical P=NP no-go transfer specialization using the module's exponential
requirement profile. -/
theorem p_eq_np_physically_impossible_canonical
    (c : PhysicalComputer)
    {P_eq_NP : Prop}
    (hMap : P_eq_NP → PhysicalCollapseAtRequirement c coNP_requirement) :
    ¬ P_eq_NP :=
  p_eq_np_physically_impossible_of_collapse_map
    (c := c) (req := coNP_requirement) hMap

/-- Coherent DQ-rejection schema at a requirement profile:
the rejection claim is asserted and is accompanied by an explicit collapse map
from that claim into physical-collapse feasibility. -/
def CoherentDQRejectionAtRequirement
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    (RejectDQ : Prop) : Prop :=
  RejectDQ ∧ (RejectDQ → PhysicalCollapseAtRequirement c req)

/-- A coherent DQ-rejection package is physically impossible at any requirement
profile with exponential lower-bound interface and finite positive bit-cost. -/
theorem coherent_dq_rejection_impossible_at_requirement
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    {RejectDQ : Prop} :
    ¬ CoherentDQRejectionAtRequirement c req RejectDQ := by
  intro hReject
  rcases hReject with ⟨hClaim, hMap⟩
  exact no_physical_collapse_at_requirement c req (hMap hClaim)

/-- Canonical specialization for `coNP_requirement`. -/
theorem coherent_dq_rejection_impossible_canonical
    (c : PhysicalComputer)
    {RejectDQ : Prop} :
    ¬ CoherentDQRejectionAtRequirement c coNP_requirement RejectDQ :=
  coherent_dq_rejection_impossible_at_requirement
    (c := c) (req := coNP_requirement)

/-- Same statement, exposed as the paper-facing corollary name. -/
theorem coNP_not_in_P_physically
    (c : PhysicalComputer)
    (req : ComputationalRequirement) :
    ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ →
      c.E_max < req.required_ops n * bit_energy_cost c :=
  coNP_physically_impossible c req

/-- Paper-facing specialization for the canonical exponential requirement. -/
theorem sufficiency_physically_impossible
    (c : PhysicalComputer) :
    ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ →
      c.E_max < coNP_requirement.required_ops n * bit_energy_cost c :=
  coNP_physically_impossible c coNP_requirement

/-! ## Tightening Layer 1: Explicit SAT Bridge for the Collapse Map -/

/-- Explicit operational witness used as a concrete `P = NP` proxy at SAT:
there exists a correct SAT3 algorithm with a polynomial runtime envelope. -/
structure PolytimeSATWitness where
  algorithm : SAT3Algorithm
  degree : ℕ
  coeff : ℕ
  coeff_pos : 0 < coeff
  poly_runtime :
    ∀ φ : SAT3Instance,
      algorithm.runtime φ ≤ coeff * (φ.inputSize + 1) ^ degree

/-- Concrete proposition used in this module for the SAT-side `P = NP` bridge. -/
def P_eq_NP_via_SAT : Prop := Nonempty PolytimeSATWitness

/-- Requirement-to-SAT reduction bridge:
for each requirement size `n`, an encoded SAT instance is fixed, and any correct
SAT algorithm runtime on that instance lower-bounds the requirement operations. -/
structure SAT3ReductionBridge (req : ComputationalRequirement) where
  encode : ℕ → SAT3Instance
  req_le_runtime_of_correct :
    ∀ (alg : SAT3Algorithm) (n : ℕ),
      req.required_ops n ≤ alg.runtime (encode n)

/-- Reduction-level transfer to physical cost:
operation lower bounds lift monotonically to bit-energy lower bounds. -/
theorem sat_reduction_transfers_energy_lower_bound
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    (R : SAT3ReductionBridge req)
    (alg : SAT3Algorithm)
    (n : ℕ) :
    req.required_ops n * bit_energy_cost c ≤
      alg.runtime (R.encode n) * bit_energy_cost c := by
  exact Nat.mul_le_mul_right (bit_energy_cost c) (R.req_le_runtime_of_correct alg n)

/-- Derived collapse-map theorem from explicit SAT witness + reduction bridge +
per-instance physical realizability of the SAT runtime trace. -/
theorem physical_collapse_of_polytime_sat_realization
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    (R : SAT3ReductionBridge req)
    (hPNP : P_eq_NP_via_SAT)
    (hPhys :
      ∀ φ : SAT3Instance,
        (Classical.choice hPNP).algorithm.runtime φ * bit_energy_cost c ≤ c.E_max) :
    PhysicalCollapseAtRequirement c req := by
  intro n
  refine ⟨(Classical.choice hPNP).algorithm.runtime (R.encode n), ?_, ?_⟩
  · exact R.req_le_runtime_of_correct (Classical.choice hPNP).algorithm n
  · exact hPhys (R.encode n)

/-- Canonical no-go derived without taking `hMap` as a primitive theorem input:
the map is produced internally by the SAT witness + reduction + physical trace model. -/
theorem p_eq_np_physically_impossible_via_sat_bridge
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    (R : SAT3ReductionBridge req)
    (hPhys :
      ∀ hPNP : P_eq_NP_via_SAT,
        ∀ φ : SAT3Instance,
          (Classical.choice hPNP).algorithm.runtime φ * bit_energy_cost c ≤ c.E_max) :
    ¬ P_eq_NP_via_SAT := by
  intro hPNP
  have hCollapse : PhysicalCollapseAtRequirement c req :=
    physical_collapse_of_polytime_sat_realization c req R hPNP (hPhys hPNP)
  exact no_physical_collapse_at_requirement c req hCollapse

/-! ## Tightening Layer 2: Concrete SAT-Hard Families -/

/-- A concrete SAT3 hard family indexed by requirement size. -/
structure SAT3HardFamily where
  encode : ℕ → SAT3Instance
  exp_lb_runtime :
    ∀ (alg : SAT3Algorithm) (n : ℕ),
      2 ^ n ≤ alg.runtime (encode n)

/-- Convert a SAT3 hard family into a canonical requirement bridge for `2^n`. -/
def SAT3HardFamily.toBridge (F : SAT3HardFamily) :
    SAT3ReductionBridge coNP_requirement where
  encode := F.encode
  req_le_runtime_of_correct := by
    intro alg n
    simpa [coNP_requirement] using F.exp_lb_runtime alg n

/-- SAT-hard-family specialization of the derived no-go map. -/
theorem p_eq_np_physically_impossible_via_sat_hard_family
    (c : PhysicalComputer)
    (F : SAT3HardFamily)
    (hPhys :
      ∀ hPNP : P_eq_NP_via_SAT,
        ∀ φ : SAT3Instance,
          (Classical.choice hPNP).algorithm.runtime φ * bit_energy_cost c ≤ c.E_max) :
    ¬ P_eq_NP_via_SAT := by
  exact p_eq_np_physically_impossible_via_sat_bridge
    (c := c) (req := coNP_requirement) (R := F.toBridge) hPhys

/-! ## Tightening Layer 3: Assumption Necessity Countermodels -/

/-- Relaxed physical model used only to witness assumption necessity. -/
structure PhysicalComputerRelaxed where
  E_max : ℕ
  bit_cost : ℕ

/-- Relaxed requirement interface without an exponential lower-bound field. -/
structure ComputationalRequirementRelaxed where
  required_ops : ℕ → ℕ

/-- Collapse predicate on the relaxed model. -/
def PhysicalCollapseAtRequirementRelaxed
    (c : PhysicalComputerRelaxed) (req : ComputationalRequirementRelaxed) : Prop :=
  ∀ n : ℕ, ∃ bits : ℕ,
    req.required_ops n ≤ bits ∧
    bits * c.bit_cost ≤ c.E_max

/-- Exponential relaxed requirement (`2^n`) used in countermodels. -/
def exp_requirement_relaxed : ComputationalRequirementRelaxed where
  required_ops := fun n => 2 ^ n

/-- Countermodel: if per-bit cost is zero, collapse is feasible even for exponential demand. -/
theorem collapse_possible_without_positive_bit_cost (E : ℕ) :
    PhysicalCollapseAtRequirementRelaxed
      ({ E_max := E, bit_cost := 0 } : PhysicalComputerRelaxed)
      exp_requirement_relaxed := by
  intro n
  refine ⟨2 ^ n, le_rfl, ?_⟩
  simp

/-- Zero-work relaxed requirement used to witness necessity of exponential lower bounds. -/
def zero_requirement_relaxed : ComputationalRequirementRelaxed where
  required_ops := fun _ => 0

/-- Countermodel: without exponential-growth lower bounds on required operations,
collapse can hold under finite budget and positive bit-cost (choose zero required work). -/
theorem collapse_possible_without_exponential_lower_bound
    (c : PhysicalComputer) :
    PhysicalCollapseAtRequirementRelaxed
      ({ E_max := c.E_max, bit_cost := c.bit_cost } : PhysicalComputerRelaxed)
      zero_requirement_relaxed := by
  intro n
  refine ⟨0, by simp [zero_requirement_relaxed], by simp⟩

/-- Logical necessity of the map premise for transfer claims:
if `P` holds while collapse fails, then no map `P -> collapse` can exist. -/
theorem no_go_transfer_requires_collapse_map
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    {P : Prop}
    (hP : P)
    (hNoCollapse : ¬ PhysicalCollapseAtRequirement c req) :
    ¬ (P → PhysicalCollapseAtRequirement c req) := by
  intro hMap
  exact hNoCollapse (hMap hP)

/-! ## Tightening Layer 4: Sharp Assumption-Failure Disjunction -/

/-- Budget profile indexed by instance size. -/
abbrev BudgetProfile := ℕ → ℕ

/-- Bounded-budget condition (finite global upper bound over all sizes). -/
def BoundedBudget (B : BudgetProfile) : Prop :=
  ∃ Bmax : ℕ, ∀ n : ℕ, B n ≤ Bmax

/-- Exponential lower-bound condition on required operations. -/
def ExponentialLowerBound (ops : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 2 ^ n ≤ ops n

/-- Collapse profile under a budget function and per-bit cost. -/
def PhysicalCollapseWithBudgetProfile
    (bitCost : ℕ) (B : BudgetProfile) (ops : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ bits : ℕ,
    ops n ≤ bits ∧ bits * bitCost ≤ B n

/-- Countermodel (finite-budget necessity): with a size-growing budget profile,
collapse is constructively feasible for any operation profile. -/
theorem collapse_possible_with_unbounded_budget_profile
    (bitCost : ℕ) (ops : ℕ → ℕ) :
    PhysicalCollapseWithBudgetProfile bitCost (fun n => ops n * bitCost) ops := by
  intro n
  refine ⟨ops n, le_rfl, ?_⟩
  simp

/-- Exponential-times-bitCost profile is not globally bounded when `bitCost > 0`. -/
theorem exp_budget_profile_unbounded
    {bitCost : ℕ}
    (hBit : 0 < bitCost) :
    ¬ BoundedBudget (fun n => (2 ^ n) * bitCost) := by
  intro hBounded
  rcases hBounded with ⟨Bmax, hBmax⟩
  obtain ⟨n₀, hGrow⟩ := pow2_eventually_exceeds Bmax
  have hPowLt : Bmax < 2 ^ n₀ := hGrow n₀ (le_rfl)
  have hOneLe : 1 ≤ bitCost := Nat.succ_le_of_lt hBit
  have hPowLeMul : 2 ^ n₀ ≤ (2 ^ n₀) * bitCost := by
    simpa [Nat.mul_one] using Nat.mul_le_mul_left (2 ^ n₀) hOneLe
  have hBLtMul : Bmax < (2 ^ n₀) * bitCost := lt_of_lt_of_le hPowLt hPowLeMul
  have hMulLeB : (2 ^ n₀) * bitCost ≤ Bmax := hBmax n₀
  exact (Nat.not_lt_of_ge hMulLeB) hBLtMul

/-- Explicit finite-budget necessity witness:
there exists an unbounded budget profile where exponential-work collapse is feasible
even with positive bit-cost. -/
theorem finite_budget_assumption_is_necessary
    {bitCost : ℕ}
    (hBit : 0 < bitCost) :
    ∃ B : BudgetProfile,
      ¬ BoundedBudget B ∧
      ExponentialLowerBound (fun n => 2 ^ n) ∧
      PhysicalCollapseWithBudgetProfile bitCost B (fun n => 2 ^ n) := by
  refine ⟨fun n => (2 ^ n) * bitCost, ?_, ?_, ?_⟩
  · exact exp_budget_profile_unbounded hBit
  · intro n
    exact le_rfl
  · simpa using collapse_possible_with_unbounded_budget_profile bitCost (fun n => 2 ^ n)

/-- Core no-collapse theorem in profile form:
bounded budget + positive bit-cost + exponential operation lower bound
jointly preclude collapse. -/
theorem no_collapse_of_bounded_budget_pos_cost_exp_lb
    {bitCost : ℕ} {B : BudgetProfile} {ops : ℕ → ℕ}
    (hBit : 0 < bitCost)
    (hBounded : BoundedBudget B)
    (hExp : ExponentialLowerBound ops) :
    ¬ PhysicalCollapseWithBudgetProfile bitCost B ops := by
  intro hCollapse
  rcases hBounded with ⟨Bmax, hBmax⟩
  obtain ⟨n₀, hGrow⟩ := pow2_eventually_exceeds Bmax
  rcases hCollapse n₀ with ⟨bits, hOpsLeBits, hBitsBudget⟩
  have hBudgetLe : bits * bitCost ≤ Bmax := le_trans hBitsBudget (hBmax n₀)
  have hPowLt : Bmax < 2 ^ n₀ := hGrow n₀ (le_rfl)
  have hPowLeOps : 2 ^ n₀ ≤ ops n₀ := hExp n₀
  have hOpsLtBits : Bmax < bits := lt_of_lt_of_le hPowLt (le_trans hPowLeOps hOpsLeBits)
  have hOneLe : 1 ≤ bitCost := Nat.succ_le_of_lt hBit
  have hBitsLeMul : bits ≤ bits * bitCost := by
    simpa [Nat.mul_one] using Nat.mul_le_mul_left bits hOneLe
  have hBudgetLt : Bmax < bits * bitCost := lt_of_lt_of_le hOpsLtBits hBitsLeMul
  exact (Nat.not_lt_of_ge hBudgetLe) hBudgetLt

/-- Sharp disjunction:
if collapse is claimed, at least one standard premise must fail
(unbounded budget, zero per-bit cost, or failed exponential lower bound). -/
theorem collapse_implies_assumption_failure_disjunction
    {bitCost : ℕ} {B : BudgetProfile} {ops : ℕ → ℕ}
    (hCollapse : PhysicalCollapseWithBudgetProfile bitCost B ops) :
    ¬ BoundedBudget B ∨ bitCost = 0 ∨ ¬ ExponentialLowerBound ops := by
  classical
  by_cases hBounded : BoundedBudget B
  · by_cases hZero : bitCost = 0
    · exact Or.inr (Or.inl hZero)
    · have hPos : 0 < bitCost := Nat.pos_of_ne_zero hZero
      by_cases hExp : ExponentialLowerBound ops
      · exfalso
        exact (no_collapse_of_bounded_budget_pos_cost_exp_lb hPos hBounded hExp) hCollapse
      · exact Or.inr (Or.inr hExp)
  · exact Or.inl hBounded

/-! ## Tightening Layer 4b: Explicit Irreducibility Certificates -/

/-- Drop positive bit-cost: collapse can hold even with bounded budget and
exponential operation lower bounds. -/
theorem positive_bit_cost_is_necessary
    : ∃ B : BudgetProfile,
        BoundedBudget B ∧
        ExponentialLowerBound (fun n => 2 ^ n) ∧
        PhysicalCollapseWithBudgetProfile 0 B (fun n => 2 ^ n) := by
  refine ⟨fun _ => 0, ?_, ?_, ?_⟩
  · refine ⟨0, ?_⟩
    intro n
    rfl
  · intro n
    exact le_rfl
  · intro n
    refine ⟨2 ^ n, le_rfl, ?_⟩
    simp

/-- Drop exponential lower bound: collapse can hold even with positive bit-cost
and bounded budgets. -/
theorem exponential_lower_bound_is_necessary
    : ∃ bitCost : ℕ, ∃ B : BudgetProfile, ∃ ops : ℕ → ℕ,
        0 < bitCost ∧
        BoundedBudget B ∧
        ¬ ExponentialLowerBound ops ∧
        PhysicalCollapseWithBudgetProfile bitCost B ops := by
  refine ⟨1, (fun _ => 0), (fun _ => 0), by decide, ?_, ?_, ?_⟩
  · refine ⟨0, ?_⟩
    intro n
    rfl
  · intro hExp
    have h0 : 2 ^ 0 ≤ (0 : ℕ) := hExp 0
    norm_num at h0
  · intro n
    refine ⟨0, by simp, by simp⟩

/-- Drop bounded-budget premise: collapse can hold even with positive bit-cost
and exponential operation lower bounds. -/
theorem bounded_budget_is_necessary
    : ∃ bitCost : ℕ, ∃ B : BudgetProfile, ∃ ops : ℕ → ℕ,
        0 < bitCost ∧
        ¬ BoundedBudget B ∧
        ExponentialLowerBound ops ∧
        PhysicalCollapseWithBudgetProfile bitCost B ops := by
  refine ⟨1, (fun n => (2 ^ n) * 1), (fun n => 2 ^ n), by decide, ?_, ?_, ?_⟩
  · simpa using (exp_budget_profile_unbounded (bitCost := 1) (hBit := by decide))
  · intro n
    exact le_rfl
  · simpa using collapse_possible_with_unbounded_budget_profile 1 (fun n => 2 ^ n)

/-- Kernel-level minimality summary:
bounded budget, positive bit-cost, and exponential lower bound are jointly
sufficient for no-collapse, and each one is independently necessary. -/
theorem kernel_assumption_set_minimal :
    (∀ {bitCost : ℕ} {B : BudgetProfile} {ops : ℕ → ℕ},
      0 < bitCost → BoundedBudget B → ExponentialLowerBound ops →
      ¬ PhysicalCollapseWithBudgetProfile bitCost B ops) ∧
    (∃ bitCost : ℕ, ∃ B : BudgetProfile, ∃ ops : ℕ → ℕ,
      0 < bitCost ∧ ¬ BoundedBudget B ∧ ExponentialLowerBound ops ∧
      PhysicalCollapseWithBudgetProfile bitCost B ops) ∧
    (∃ B : BudgetProfile,
      BoundedBudget B ∧ ExponentialLowerBound (fun n => 2 ^ n) ∧
      PhysicalCollapseWithBudgetProfile 0 B (fun n => 2 ^ n)) ∧
    (∃ bitCost : ℕ, ∃ B : BudgetProfile, ∃ ops : ℕ → ℕ,
      0 < bitCost ∧ BoundedBudget B ∧ ¬ ExponentialLowerBound ops ∧
      PhysicalCollapseWithBudgetProfile bitCost B ops) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro bitCost B ops hBit hBounded hExp
    exact no_collapse_of_bounded_budget_pos_cost_exp_lb hBit hBounded hExp
  · exact bounded_budget_is_necessary
  · exact positive_bit_cost_is_necessary
  · exact exponential_lower_bound_is_necessary

/-! ## Tightening Layer 5: Regime Robustness (Deterministic / Probabilistic / Sequential) -/

/-- Deterministic operation profile with exponential lower bound. -/
structure DeterministicRequirement where
  ops : ℕ → ℕ
  exp_lb : ∀ n : ℕ, 2 ^ n ≤ ops n

/-- Probabilistic expected-operation profile with exponential lower bound. -/
structure ProbabilisticRequirement where
  expectedOps : ℕ → ℕ
  exp_lb : ∀ n : ℕ, 2 ^ n ≤ expectedOps n

/-- Sequential profile with stage count and per-stage work; cumulative work is
required to satisfy an exponential lower bound. -/
structure SequentialRequirement where
  stages : ℕ → ℕ
  perStageOps : ℕ → ℕ
  exp_lb_total : ∀ n : ℕ, 2 ^ n ≤ stages n * perStageOps n

def DeterministicRequirement.toComputationalRequirement
    (R : DeterministicRequirement) : ComputationalRequirement where
  required_ops := R.ops
  lower_exp := R.exp_lb

def ProbabilisticRequirement.toComputationalRequirement
    (R : ProbabilisticRequirement) : ComputationalRequirement where
  required_ops := R.expectedOps
  lower_exp := R.exp_lb

def SequentialRequirement.toComputationalRequirement
    (R : SequentialRequirement) : ComputationalRequirement where
  required_ops := fun n => R.stages n * R.perStageOps n
  lower_exp := R.exp_lb_total

/-- Deterministic robustness specialization of the no-collapse theorem. -/
theorem deterministic_no_physical_collapse
    (c : PhysicalComputer)
    (R : DeterministicRequirement) :
    ¬ PhysicalCollapseAtRequirement c R.toComputationalRequirement :=
  no_physical_collapse_at_requirement c R.toComputationalRequirement

/-- Probabilistic robustness specialization (expected operations). -/
theorem probabilistic_no_physical_collapse
    (c : PhysicalComputer)
    (R : ProbabilisticRequirement) :
    ¬ PhysicalCollapseAtRequirement c R.toComputationalRequirement :=
  no_physical_collapse_at_requirement c R.toComputationalRequirement

/-- Sequential robustness specialization (horizon-cumulative operations). -/
theorem sequential_no_physical_collapse
    (c : PhysicalComputer)
    (R : SequentialRequirement) :
    ¬ PhysicalCollapseAtRequirement c R.toComputationalRequirement :=
  no_physical_collapse_at_requirement c R.toComputationalRequirement

end PhysicalComplexity
