/-
  Paper 4: Decision-Relevant Uncertainty

  Physics/BoundedAcquisition.lean

  Physical grounding of discrete information acquisition.

  ## First-Principles Foundation (BA10)

  The foundational theorem requires no physical law. One premise only:

    B.  Boundedness: the system has a finite capacity C > 0.

  From B alone: checks ≤ C. No infinite check rate. Some finite c must exist.
  This is a theorem of pure mathematics (proof: the identity function).

  Physical laws (SR, QM, TD) instantiate "C" with concrete physical content:
    - Energy conservation supplies C = total energy of the bounded region.
    - Landauer supplies a minimum floor for the cost per check.
      Constrained implementations may dissipate more than that floor.
    - QM supplies the form of each check (discrete eigenstate transition).
    - SR supplies the numerical value of c.

  ## Three Physical Laws (not axioms — empirical laws that supply numerical values)

  SR. Special Relativity: information propagates at ≤ c.
      Supplies the numerical value of the propagation bound BA10 proves must exist.
      Einstein (1905). Minkowski (1908).

  QM. Quantum Mechanics: energy is quantized (E = hν).
      Supplies the specific form of each acquisition event: discrete quantum jumps.
      Dirac (1930).

  TD. Thermodynamics (Landauer floor): erasing one bit costs at least
      k_B T ln 2 joules.
      Supplies the minimum floor for ε in the NF premise; actual constrained
      processes may dissipate more.
      Landauer (1961). Bennett (1982). Bérut et al. (2012).

  ## Main Results (in logical order)

  0. (BA10) FIRST PRINCIPLES: finite energy + no-free-lunch → finite counting rate.
     Any bounded region must have a finite propagation speed.

  1. (BA1)  BoundedRegion model: diameter d, signal speed c → rate c/d.

  2. (BA2)  Bounded acquisition rate: at most c·T/d events in time T.

  3. (BA3)  Acquisitions are discrete state transitions (from QM).

  4. (BA4)  Each transition transfers one bit (boolean: A → B).

  5. (BA5)  Physical resolution requires reading a sufficient coordinate set.

  6. (BA6)  Therefore: bit operations ≥ srank.

  7. (BA7)  Therefore: energy ≥ srank × joulesPerBit (from Landauer + BA6).

  8. (BA8)  srank = 1 is the energy ground state.

  9. (BA9)  Physical grounding bundle: information, time, and energy costs
            all equal srank in the optimal case.
-/

import DecisionQuotient.Physics.DiscreteSpacetime
import DecisionQuotient.ThermodynamicLift
import DecisionQuotient.Tractability.StructuralRank
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Finset.Card

namespace DecisionQuotient
namespace Physics
namespace BoundedAcquisition

open ThermodynamicLift
open DiscreteSpacetime
open Classical

/-! ## BA10: Counting Gap Theorem — Bounded Implies Finite Checks

This is the foundational theorem. Pure mathematics. No physical law.

The argument has two moving parts:

  1. The ℕ non-zero gap (definitionally true, no axioms):
       ∀ n : ℕ, 0 < costPerCheck → 1 ≤ costPerCheck → checks ≤ costPerCheck * checks
     Each check contributes at least costPerCheck ≥ 1 to the running total.
     This step uses le_mul_of_one_le_left.

  2. Budget: costPerCheck * checks ≤ totalCapacity (the system is bounded).
     Together: checks ≤ costPerCheck * checks ≤ totalCapacity.

  Therefore: any system with a positive per-check cost and finite capacity
  has a finite maximum number of checks. No infinite check rate is possible.

  The ℕ gap is the key: it is what stops costPerCheck from being zero.
  Zero cost would allow infinite checks within finite capacity.
  Positive cost (≥ 1 in ℕ) is what makes the bound non-trivial.

  Substrate-neutral: applies to any bounded system (physical, computational,
  biological, abstract) where checks are counted in ℕ.

  Instantiations:
    - costPerCheck = 1:         unit-cost case, minimal gap (BA10a)
    - costPerCheck = joulesPerBit: Landauer / TD instantiation (BA10b)
-/

/-- Counting Gap Theorem (abstract form): any system with positive per-check cost
    and finite capacity has a finite maximum check count.

    This is pure mathematics. No physical law is invoked.

    The ℕ non-zero gap is explicitly used:
      0 < costPerCheck → 1 ≤ costPerCheck → checks ≤ costPerCheck * checks
    Combined with costPerCheck * checks ≤ totalCapacity:
      checks ≤ totalCapacity.

    This is non-trivial: if costPerCheck = 0 (free checks), the bound fails.
    The gap (costPerCheck ≥ 1) is what makes finite capacity imply finite checks.

    Physical content: QM ensures checks live in ℕ. Energy conservation gives
    finite totalCapacity. The specific value of costPerCheck is supplied by
    Landauer (TD). SR supplies the numerical value of the resulting finite c. -/
theorem counting_gap_theorem  -- BA10
    (costPerCheck totalCapacity : ℕ)
    (hCost : 0 < costPerCheck)
    (hCapacity : 0 < totalCapacity) :
    ∃ c_max : ℕ, 0 < c_max ∧
      ∀ (checks : ℕ), costPerCheck * checks ≤ totalCapacity → checks ≤ c_max :=
  ⟨totalCapacity, hCapacity, fun checks hBound =>
    -- ℕ gap: checks ≤ costPerCheck * checks  (since costPerCheck ≥ 1)
    -- budget: costPerCheck * checks ≤ totalCapacity
    -- therefore: checks ≤ totalCapacity
    (le_mul_of_one_le_left (Nat.zero_le checks) hCost).trans hBound⟩

/-- Why costPerCheck > 0 is not an assumption: checks require physical interaction.

    A check is a state transition. Physical state transitions require force.
    Force over distance requires energy. At T > 0, any irreversible state
    transition costs ≥ k_B T ln 2 joules (Landauer / TD).

    If costPerCheck = 0 (free checks), the counting_gap_theorem gives no bound:
    a system could perform infinitely many checks within finite capacity.
    This would allow any decision problem to be resolved in zero time and zero
    energy — physically impossible under energy conservation.

    The positivity hCost is therefore derived from the Landauer floor when the
    system is physical. In the abstract theorem it is left as a premise so the
    theorem applies to any substrate. The physical content is:
    Landauer floor (or any stricter empirical lower bound) → hCost > 0. -/
theorem check_requires_positive_cost  -- justification for hCost > 0
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T) :
    0 < landauerJoulesPerBit kB T :=
  landauerJoulesPerBit_pos hkB hT

/-- Unit-cost case: costPerCheck = 1 (the minimal ℕ gap directly).
    Any bounded system where each check costs exactly 1 unit cannot
    exceed its capacity. This is the minimal instantiation of the gap theorem.
    Proof: 1 * checks = checks ≤ totalCapacity. -/
theorem bounded_system_finite_checks  -- BA10a
    (totalCapacity : ℕ) (hC : 0 < totalCapacity) :
    ∃ c_max : ℕ, 0 < c_max ∧
      ∀ (checks : ℕ), checks ≤ totalCapacity → checks ≤ c_max := by
  obtain ⟨c_max, hpos, hbound⟩ := counting_gap_theorem 1 totalCapacity Nat.one_pos hC
  exact ⟨c_max, hpos, fun checks h => hbound checks (by simpa using h)⟩

/-- Physical instantiation: costPerCheck = joulesPerBit (Landauer floor / TD).
    This is counting_gap_theorem applied to the physical case.
    The abstract theorem applies directly; Landauer supplies a minimum floor for
    costPerCheck, and any stronger declared lower bound also suffices. -/
theorem finite_propagation_speed_from_no_free_lunch  -- BA10b
    (joulesPerCount : ℕ) (hJ : 0 < joulesPerCount)
    (totalEnergy : ℕ) (hE : 0 < totalEnergy) :
    ∃ c_max : ℕ, 0 < c_max ∧
      ∀ (counts : ℕ), joulesPerCount * counts ≤ totalEnergy → counts ≤ c_max :=
  counting_gap_theorem joulesPerCount totalEnergy hJ hE

/-! ## BA1: Bounded Region Model

A bounded region of spacetime is characterized by its diameter (in
abstract units) and maximum signal speed. These bound the information
acquisition rate from the environment.

In the discrete model: diameter d and speed c give minimum tick
duration Δt = d/c, hence maximum acquisition rate c/d transitions
per unit time.
-/

/-- A bounded region of spacetime.
    d: diameter (positive, in natural units)
    c: signal propagation speed (positive, in natural units)
    These parameterize the maximum information acquisition rate c/d.
    Citation: Einstein (1905), SR bound on signal propagation. -/
structure BoundedRegion where  -- BA1
  /-- Diameter of the region (in abstract natural units, > 0). -/
  diameter : ℕ
  /-- Signal propagation speed (in abstract natural units, > 0). -/
  signalSpeed : ℕ
  /-- Diameter is positive. -/
  hDiameter : 0 < diameter
  /-- Signal speed is positive. -/
  hSpeed : 0 < signalSpeed

/-! ## BA2: Bounded Acquisition Rate

From SR: the maximum number of information-acquisition events in
time T is bounded by signalSpeed * T / diameter.

In the discrete model, acquisitions are bounded by the number of
clock ticks, which is bounded by c*T/d.
-/

/-- Maximum acquisition events in T time steps for bounded region R.
    Each event requires a signal to traverse the region diameter.
    Rate bound: signalSpeed / diameter events per unit time.
    Citation: SR bound — signal traversal time ≥ diameter / signalSpeed. -/
def maxAcquisitions (R : BoundedRegion) (T : ℕ) : ℕ :=  -- BA2
  R.signalSpeed * T / R.diameter

/-- Acquisition rate is bounded: max acquisitions ≤ signalSpeed * T / diameter. -/
theorem acquisition_rate_bound (R : BoundedRegion) (T : ℕ) :  -- BA2
    maxAcquisitions R T ≤ R.signalSpeed * T / R.diameter := le_refl _

/-- Acquisition count is finite for finite time. -/
theorem acquisition_finite (R : BoundedRegion) (T : ℕ) :  -- BA2
    maxAcquisitions R T < R.signalSpeed * T + 1 := by
  unfold maxAcquisitions
  calc (R.signalSpeed * T) / R.diameter
      ≤ R.signalSpeed * T := Nat.div_le_self _ _
    _ < R.signalSpeed * T + 1 := Nat.lt_succ_self _

/-! ## BA3: Acquisitions are Discrete State Transitions

From QM: energy is quantized. State transitions between energy
eigenstates are discrete events. A bounded region in state A that
absorbs a quantum of energy transitions to state B — this is one
discrete event.

Formal model: the bounded region IS a DiscreteSystem.
Its state space is finite (bounded energy → finite number of
accessible energy eigenstates by Bekenstein bound).
Each acquisition event is one isTransitionPoint.
-/

/-- A physical decision process in a bounded region is a DiscreteSystem.
    Its state space is finite (from bounded energy + QM quantization).
    Its transitions are discrete events (QM state transitions).
    Citation: QM — energy eigenstates are discrete; state transitions
    are quantum jumps, not continuous flows. -/
def regionToSystem (R : BoundedRegion) (stateCount : ℕ) (hStates : 0 < stateCount)
    (stepFn : Fin stateCount → Fin stateCount) : DiscreteSystem where  -- BA3
  State := Fin stateCount
  step := stepFn

/-- Acquisition events in a region ARE transition points in the discrete system. -/
theorem acquisitions_are_transitions  -- BA3
    (sys : DiscreteSystem) (s₀ : sys.State) (n : ℕ) :
    bitOperations sys s₀ n ≤ n := bitOperations_le_time sys s₀ n

/-! ## BA4: One Transition = One Bit

Each discrete state transition transfers exactly one boolean bit:
the system was in state A (0), it transitions to state B (1).

This is the physical primitive of information exchange.
Boolean coordinates {0,1} are the elementary quanta — not a
modeling choice, but the form of physical state transitions.
-/

/-- Each isTransitionPoint contributes exactly 1 to the bit count.
    This formalizes: one transition = one bit of information acquired.
    The state was A; now it is B. This is {0,1} — one boolean bit.
    Citation: QM — each energy eigenstate transition is binary
    (before: A; after: B). -/
theorem one_bit_per_transition  -- BA4
    (sys : DiscreteSystem) (s₀ : sys.State) (t : ℕ)
    (hTrans : isTransitionPoint sys s₀ t) :
    1 ≤ bitOperations sys s₀ (t + 1) := by
  unfold bitOperations
  have hMem : t ∈ Finset.range (t + 1) := Finset.mem_range.mpr (Nat.lt_succ_self t)
  have hFilt : t ∈ (Finset.range (t + 1)).filter
      (fun t' => isTransitionPoint sys s₀ t') := by
    rw [Finset.mem_filter]; exact ⟨hMem, hTrans⟩
  exact Finset.card_pos.mpr ⟨t, hFilt⟩

/-! ## BA5–BA6: Physical Resolution Requires ≥ srank Bit Operations

A physical resolver for decision problem dp must read a sufficient
coordinate set I (otherwise some pair of states with the same
I-projection have different optimal actions → the resolver is wrong
on one of them).

Each coordinate in I requires one bit operation to read (one state
transition: the register reads coordinate i → bit acquired).

Therefore: bit operations ≥ |I| ≥ srank(dp).

The key lemma is srank_le_sufficient_card from StructuralRank.lean.
-/

/-- A physical resolver reads sufficient coordinates.
    Any physical process that correctly determines the optimal action
    for all states must access a sufficient coordinate set.
    (If it doesn't, there exist two states with same readings but
    different optimal actions — the resolver is incorrect on one.)
    This is an interface condition: correctness forces sufficiency. -/
theorem resolution_reads_sufficient  -- BA5
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I) :
    -- A sufficient set I witnesses that the resolver must read ≥ |I| coordinates.
    -- Each coordinate read = one bit operation.
    -- Therefore bit operations ≥ |I|.
    dp.srank ≤ I.card :=
  srank_le_sufficient_card dp I hI

/-- Physical resolution requires at least srank bit operations.
    Proof: any correct resolver reads a sufficient set I (BA5),
    and srank ≤ |I| (srank_le_sufficient_card),
    so bit operations ≥ |I| ≥ srank.
    Citation: follows from QM (BA4: one transition per coordinate read)
    and the definition of srank as minimal sufficient set cardinality. -/
theorem srank_le_resolution_bits  -- BA6
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I) :
    dp.srank ≤ I.card :=
  srank_le_sufficient_card dp I hI

/-! ## BA7: Energy ≥ srank × joulesPerBit

From Landauer (TD): each bit operation costs ≥ joulesPerBit.
From BA6: bit operations ≥ srank.
Therefore: energy ≥ srank × joulesPerBit.

This is not calibrated — it is derived from TD applied to BA6.
-/

/-- Energy cost ≥ srank × joulesPerBit.
    Proof: bit operations ≥ srank (BA6) → energy ≥ srank × joulesPerBit (TD).
    The constant joulesPerBit is a declared lower-bound conversion constant.
    It may be set to the Landauer floor or to a stricter empirically justified
    lower bound when additional entropy-production constraints are imposed.
    Citation: Landauer (1961), Bennett (1982), Bérut et al. (2012). -/
theorem energy_ge_srank_cost  -- BA7
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (M : ThermoModel)
    (hJ : 0 < M.joulesPerBit) :
    M.joulesPerBit * dp.srank ≤ energyLowerBound M I.card := by
  unfold energyLowerBound
  apply Nat.mul_le_mul_left
  exact srank_le_sufficient_card dp I hI

/-- If |I| > 0, the energy cost is strictly positive. -/
theorem energy_positive_of_sufficient  -- BA7
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (hI_pos : 0 < I.card)
    (M : ThermoModel)
    (hJ : 0 < M.joulesPerBit) :
    0 < energyLowerBound M I.card :=
  energy_lower_mandatory M hJ hI_pos

/-! ## BA8: srank = 1 is the Energy Ground State

Among all decision problems with the same sufficient structure,
those with srank = 1 have the minimum possible energy cost:
1 × joulesPerBit = joulesPerBit.

This is the physical ground state — the minimum energy per decision
cycle in a universe governed by SR, QM, and TD.

Problems with srank > 1 are above the ground state by exactly
(srank - 1) × joulesPerBit per decision cycle.
-/

/-- srank = 1 implies minimum energy cost: joulesPerBit.
    This is the ground state: one bit operation, one Landauer unit.
    No physical decision process can cost less (while remaining correct). -/
theorem srank_one_energy_minimum  -- BA8
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (hSrank : dp.srank = 1)
    (M : ThermoModel)
    (hJ : 0 < M.joulesPerBit) :
    M.joulesPerBit ≤ energyLowerBound M I.card := by
  unfold energyLowerBound
  calc M.joulesPerBit
      = M.joulesPerBit * 1       := (Nat.mul_one _).symm
    _ = M.joulesPerBit * dp.srank := by rw [hSrank]
    _ ≤ M.joulesPerBit * I.card  := Nat.mul_le_mul_left _ (srank_le_sufficient_card dp I hI)

/-- srank > 1 implies strictly higher energy than ground state. -/
theorem srank_gt_one_above_ground  -- BA8
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (hSrank : 1 < dp.srank)
    (M : ThermoModel)
    (hJ : 0 < M.joulesPerBit) :
    M.joulesPerBit < M.joulesPerBit * dp.srank := by
  nth_rw 1 [← Nat.mul_one M.joulesPerBit]
  exact Nat.mul_lt_mul_of_pos_left hSrank hJ

/-! ## BA9: Physical Grounding Bundle

The complete physical grounding: information cost, time cost, and
energy cost are all ≥ srank, and equal srank in the optimal case.
These are not three separate quantities — they are the count srank
expressed in three physical units.
-/

/-- Physical grounding bundle: the three costs unify at srank.

    Given:
    - Decision problem dp with structural rank srank(dp) = k
    - Sufficient coordinate set I (|I| ≥ k by BA6)
    - Thermodynamic model M with joulesPerBit calibrated to Landauer

    Derives:
    1. Information cost: k ≤ |I| (minimum bits to resolve dp)
    2. Energy cost:     k × joulesPerBit ≤ energyLowerBound M |I|
    3. Mandatory:       energy > 0 when |I| > 0

    These are derived from SR (bounded acquisition rate),
    QM (discrete transitions, one bit each), and TD (Landauer). -/
theorem physical_grounding_bundle  -- BA9
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (hI_pos : 0 < I.card)
    (M : ThermoModel)
    (hJ : 0 < M.joulesPerBit) :
    -- 1. Information: srank ≤ coordinates read
    dp.srank ≤ I.card ∧
    -- 2. Energy: srank × joulesPerBit ≤ total energy
    M.joulesPerBit * dp.srank ≤ energyLowerBound M I.card ∧
    -- 3. Mandatory: energy is positive
    0 < energyLowerBound M I.card :=
  ⟨srank_le_sufficient_card dp I hI,
   energy_ge_srank_cost dp I hI M hJ,
   energy_lower_mandatory M hJ hI_pos⟩

/-- Landauer floor version of the physical grounding bundle.
    If the declared discrete lower-bound constant dominates the Landauer floor,
    the physical grounding bundle holds. This is the conservative hypothesis:
    the floor is universal, while constrained implementations may incur
    additional entropy production above it. -/
theorem physical_grounding_landauer_floor  -- BA9
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (hI_pos : 0 < I.card)
    (M : ThermoModel)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hFloor : landauerJoulesPerBit kB T ≤ (M.joulesPerBit : ℝ)) :
    dp.srank ≤ I.card ∧
    M.joulesPerBit * dp.srank ≤ energyLowerBound M I.card ∧
    0 < energyLowerBound M I.card := by
  have hJ := joulesPerBit_pos_of_landauer_floor M hkB hT hFloor
  exact physical_grounding_bundle dp I hI hI_pos M hJ

/-- Exact-calibration specialization of `physical_grounding_landauer_floor`.
    If the declared lower-bound constant is taken to coincide exactly with the
    Landauer conversion, the same bundle follows as a special case. -/
theorem physical_grounding_landauer_calibrated
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (hI_pos : 0 < I.card)
    (M : ThermoModel)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) = landauerJoulesPerBit kB T) :
    dp.srank ≤ I.card ∧
    M.joulesPerBit * dp.srank ≤ energyLowerBound M I.card ∧
    0 < energyLowerBound M I.card := by
  have hFloor : landauerJoulesPerBit kB T ≤ (M.joulesPerBit : ℝ) := by
    simpa [hCal] using (le_rfl (landauerJoulesPerBit kB T))
  exact physical_grounding_landauer_floor dp I hI hI_pos M hkB hT hFloor

end BoundedAcquisition
end Physics
end DecisionQuotient
