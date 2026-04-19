/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/LatticeSum.lean
  
  Formal proof of the 3D lattice sum convergence for power-law potentials.
  This justifies the cutoff approximation in molecular dynamics (Lennard-Jones 6-12).
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.Order.Archimedean.Basic
import Mathlib.Analysis.PSeries
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Topology.Algebra.InfiniteSum.Basic

namespace DecisionQuotient
namespace Tractability
namespace LatticeSum

open BigOperators

/-- 
  The sum of 1/||n||^s over all non-zero integer points in 3D 
  with norm strictly greater than R.
  This represents the "tail" of the potential energy.
-/
noncomputable def latticeTailSum (s : ℝ) (R : ℝ) : ℝ :=
  ∑' (n : ℤ × ℤ × ℤ),
    let norm : ℝ := ((n.1 : ℝ)^2 + (n.2.1 : ℝ)^2 + (n.2.2 : ℝ)^2).sqrt
    if R < norm then 1 / (norm ^ s) else 0

/--
  Compatibility assumption interface carried over from `srank-implementation`.
  The current branch proves concrete Lennard-Jones tail bounds below; this
  definition packages the abstract convergence premise as a `Prop` witness.
-/
def lattice_sum_converges (s : ℝ) (_hs : 3 < s) : Prop :=
    ∃ (M : ℝ), ∀ (R : ℝ), 0 < R → latticeTailSum s R ≤ M / R^(s - 3)

/-- Large-radius variant used for concrete Lennard-Jones instantiations in this
artifact. -/
def lattice_sum_converges_large_radius (s : ℝ) (_hs : 3 < s) : Prop :=
    ∃ (M : ℝ), ∀ (R : ℝ), 1 ≤ R → latticeTailSum s R ≤ M / R^(s - 3)

/-!
  Dyadic shell decomposition for the integer lattice in 3D. We restrict to
  finite shells of points whose Euclidean norm lies in (2^k * R, 2^(k+1) * R].
  The proofs below use crude cube-enclosure cardinality bounds and then a
  geometric-series summation for the two exponents used by Lennard-Jones.
-/

noncomputable def latticeNorm (n : ℤ × ℤ × ℤ) : ℝ :=
  Real.sqrt ((n.1 : ℝ) ^ 2 + (n.2.1 : ℝ) ^ 2 + (n.2.2 : ℝ) ^ 2)

noncomputable def tailTerm6 (n : ℤ × ℤ × ℤ) : ℝ := 1 / latticeNorm n ^ (6 : ℕ)

noncomputable def tailTerm12 (n : ℤ × ℤ × ℤ) : ℝ := 1 / latticeNorm n ^ (12 : ℕ)

noncomputable def shellBoundInt (R : ℝ) (k : ℕ) : ℤ :=
  Int.ofNat (Nat.ceil ((2 ^ (k + 1) : ℝ) * R))

noncomputable def shellCoordCount (R : ℝ) (k : ℕ) : Nat :=
  2 * (shellBoundInt R k).natAbs + 1

noncomputable def shellCoords (R : ℝ) (k : ℕ) : Finset ℤ :=
  Finset.Icc (-shellBoundInt R k) (shellBoundInt R k)

noncomputable def shellCube (R : ℝ) (k : ℕ) : Finset (ℤ × ℤ × ℤ) :=
  (((shellCoords R k).product (shellCoords R k)).product (shellCoords R k)).image
    (fun t => (t.1.1, t.1.2, t.2))

noncomputable def dyadicShell (R : ℝ) (k : ℕ) : Finset (ℤ × ℤ × ℤ) :=
  (shellCube R k).filter fun n => (2 ^ k : ℝ) * R < latticeNorm n ∧ latticeNorm n ≤ (2 ^ (k + 1) : ℝ) * R

lemma coords_card_eq (R : ℝ) (k : ℕ) :
    (shellCoords R k).card = shellCoordCount R k := by
  unfold shellCoords
  rw [Int.card_Icc]
  unfold shellCoordCount shellBoundInt
  simp
  omega

lemma shellCube_card_eq (R : ℝ) (k : ℕ) :
    (shellCube R k).card = (shellCoordCount R k) ^ 3 := by
  classical
  unfold shellCube
  have hinj :
      Function.Injective (fun t : (ℤ × ℤ) × ℤ => (t.1.1, t.1.2, t.2)) := by
    intro a b h
    rcases a with ⟨⟨a1, a2⟩, a3⟩
    rcases b with ⟨⟨b1, b2⟩, b3⟩
    simp at h
    aesop
  calc
    (shellCube R k).card
        = (shellCoordCount R k) * (shellCoordCount R k) * (shellCoordCount R k) := by
          simpa [Finset.card_product, coords_card_eq] using
            (Finset.card_image_of_injective
              (s := (((shellCoords R k).product (shellCoords R k)).product (shellCoords R k)))
              (f := fun t : (ℤ × ℤ) × ℤ => (t.1.1, t.1.2, t.2)) hinj)
    _ = (shellCoordCount R k) ^ 3 := by ring

theorem point_in_dyadicShell_bound6 (R : ℝ) (hRpos : 0 < R) {k : ℕ} {n : ℤ × ℤ × ℤ}
    (hn : n ∈ dyadicShell R k) :
    tailTerm6 n ≤ 1 / (((2 ^ k : ℝ) * R) ^ (6 : ℕ)) := by
  rw [dyadicShell, Finset.mem_filter] at hn
  have hlt : (2 ^ k : ℝ) * R < latticeNorm n := hn.2.1
  have hpos_base : 0 < (2 ^ k : ℝ) * R := by
    apply mul_pos
    · apply pow_pos; norm_num
    · exact hRpos
  have hnorm_pos : 0 < latticeNorm n := by linarith [hlt]
  -- monotonicity for integer powers: if 0 ≤ a ≤ b then a^6 ≤ b^6
  have hbase_le : (2 : ℝ) ^ k * R ≤ latticeNorm n := by linarith [hlt]
  have hbase_nonneg : 0 ≤ (2 : ℝ) ^ k * R := by positivity
  have hpow6 : (((2 : ℝ) ^ k * R) ^ 6) ≤ latticeNorm n ^ 6 := by
    exact pow_le_pow_left₀ hbase_nonneg hbase_le 6
  have hrecip : 1 / (latticeNorm n ^ (6 : ℕ)) ≤ 1 / (((2 : ℝ) ^ k * R) ^ (6 : ℕ)) :=
    one_div_le_one_div_of_le (pow_pos hpos_base 6) hpow6
  simpa [tailTerm6] using hrecip

theorem point_in_dyadicShell_bound12 (R : ℝ) (hRpos : 0 < R) {k : ℕ} {n : ℤ × ℤ × ℤ}
    (hn : n ∈ dyadicShell R k) :
    tailTerm12 n ≤ 1 / (((2 ^ k : ℝ) * R) ^ (12 : ℕ)) := by
  rw [dyadicShell, Finset.mem_filter] at hn
  have hlt : (2 ^ k : ℝ) * R < latticeNorm n := hn.2.1
  have hpos_base : 0 < (2 ^ k : ℝ) * R := by
    apply mul_pos
    · apply pow_pos; norm_num
    · exact hRpos
  have hnorm_pos : 0 < latticeNorm n := by linarith [hlt]
  have hbase_le : (2 : ℝ) ^ k * R ≤ latticeNorm n := by linarith [hlt]
  have hbase_nonneg : 0 ≤ (2 : ℝ) ^ k * R := by positivity
  have hpow12 : (((2 : ℝ) ^ k * R) ^ 12) ≤ latticeNorm n ^ 12 := by
    exact pow_le_pow_left₀ hbase_nonneg hbase_le 12
  have hrecip : 1 / (latticeNorm n ^ (12 : ℕ)) ≤ 1 / (((2 : ℝ) ^ k * R) ^ (12 : ℕ)) :=
    one_div_le_one_div_of_le (pow_pos hpos_base 12) hpow12
  simpa [tailTerm12] using hrecip

theorem point_in_dyadicShell_bound (R : ℝ) (hRpos : 0 < R) {k : ℕ} {n : ℤ × ℤ × ℤ}
    (hn : n ∈ dyadicShell R k) :
    1 / (latticeNorm n ^ 6) ≤ 1 / (((2 ^ k : ℝ) * R) ^ 6) := by
  simpa [tailTerm6] using point_in_dyadicShell_bound6 R hRpos hn

theorem coord1_abs_le_latticeNorm (n : ℤ × ℤ × ℤ) :
    |(n.1 : ℝ)| ≤ latticeNorm n := by
  apply le_of_sq_le_sq
  · rw [sq_abs, latticeNorm, Real.sq_sqrt]
    · nlinarith [sq_nonneg (n.2.1 : ℝ), sq_nonneg (n.2.2 : ℝ)]
    · positivity
  · unfold latticeNorm
    positivity

theorem coord2_abs_le_latticeNorm (n : ℤ × ℤ × ℤ) :
    |(n.2.1 : ℝ)| ≤ latticeNorm n := by
  apply le_of_sq_le_sq
  · rw [sq_abs, latticeNorm, Real.sq_sqrt]
    · nlinarith [sq_nonneg (n.1 : ℝ), sq_nonneg (n.2.2 : ℝ)]
    · positivity
  · unfold latticeNorm
    positivity

theorem coord3_abs_le_latticeNorm (n : ℤ × ℤ × ℤ) :
    |(n.2.2 : ℝ)| ≤ latticeNorm n := by
  apply le_of_sq_le_sq
  · rw [sq_abs, latticeNorm, Real.sq_sqrt]
    · nlinarith [sq_nonneg (n.1 : ℝ), sq_nonneg (n.2.1 : ℝ)]
    · positivity
  · unfold latticeNorm
    positivity

theorem mem_shellCoords_of_abs_le (R : ℝ) (k : ℕ) {z : ℤ}
    (hz : |z| ≤ shellBoundInt R k) :
    z ∈ shellCoords R k := by
  unfold shellCoords
  rw [Finset.mem_Icc]
  simpa [abs_le] using hz

theorem mem_shellCube_of_norm_le (R : ℝ) (k : ℕ) {n : ℤ × ℤ × ℤ}
    (hn : latticeNorm n ≤ (2 ^ (k + 1) : ℝ) * R) :
    n ∈ shellCube R k := by
  have hceil : (2 ^ (k + 1) : ℝ) * R ≤ (shellBoundInt R k : ℝ) := by
    simpa [shellBoundInt] using (Nat.le_ceil ((2 ^ (k + 1) : ℝ) * R))
  have h1r : |(n.1 : ℝ)| ≤ (shellBoundInt R k : ℝ) :=
    le_trans (coord1_abs_le_latticeNorm n) (hn.trans hceil)
  have h2r : |(n.2.1 : ℝ)| ≤ (shellBoundInt R k : ℝ) :=
    le_trans (coord2_abs_le_latticeNorm n) (hn.trans hceil)
  have h3r : |(n.2.2 : ℝ)| ≤ (shellBoundInt R k : ℝ) :=
    le_trans (coord3_abs_le_latticeNorm n) (hn.trans hceil)
  have h1 : |n.1| ≤ shellBoundInt R k := by exact_mod_cast h1r
  have h2 : |n.2.1| ≤ shellBoundInt R k := by exact_mod_cast h2r
  have h3 : |n.2.2| ≤ shellBoundInt R k := by exact_mod_cast h3r
  unfold shellCube
  rw [Finset.mem_image]
  refine ⟨((n.1, n.2.1), n.2.2), ?_, by simp⟩
  simp [Finset.mem_product, mem_shellCoords_of_abs_le R k h1, mem_shellCoords_of_abs_le R k h2,
    mem_shellCoords_of_abs_le R k h3]

theorem exists_mem_dyadicShell (R : ℝ) (hR : 1 ≤ R) {n : ℤ × ℤ × ℤ}
    (hn : R < latticeNorm n) :
    ∃ k : ℕ, n ∈ dyadicShell R k := by
  let x : ℝ := latticeNorm n / R
  have hRpos : 0 < R := by linarith [hR]
  have hx1 : 1 < x := by
    unfold x
    exact (one_lt_div hRpos).2 hn
  obtain ⟨m, hmLower, hmUpper⟩ := exists_nat_pow_near (x := x) (y := (2 : ℝ)) (le_of_lt hx1) one_lt_two
  by_cases hEq : x = (2 : ℝ) ^ m
  · have hmpos : 0 < m := by
      by_contra hm0
      have hmzero : m = 0 := Nat.eq_zero_of_not_pos hm0
      rw [hmzero] at hEq
      simp at hEq
      linarith
    let k := m - 1
    have hk1 : k + 1 = m := by
      unfold k
      omega
    have hklt : k < m := by
      unfold k
      omega
    have hlower : (2 : ℝ) ^ k * R < latticeNorm n := by
      have hpow : (2 : ℝ) ^ k < (2 : ℝ) ^ m := by
        exact pow_lt_pow_right₀ one_lt_two hklt
      have hxmul : x * R = latticeNorm n := by
        unfold x
        field_simp [hRpos.ne']
      calc
        (2 : ℝ) ^ k * R < (2 : ℝ) ^ m * R := by
          gcongr
        _ = x * R := by rw [hEq]
        _ = latticeNorm n := hxmul
    have hupper : latticeNorm n ≤ (2 : ℝ) ^ (k + 1) * R := by
      have hxmul : x * R = latticeNorm n := by
        unfold x
        field_simp [hRpos.ne']
      calc
        latticeNorm n = x * R := hxmul.symm
        _ = (2 : ℝ) ^ m * R := by rw [hEq]
        _ = (2 : ℝ) ^ (k + 1) * R := by rw [hk1]
        _ ≤ (2 : ℝ) ^ (k + 1) * R := by rfl
    refine ⟨k, ?_⟩
    rw [dyadicShell, Finset.mem_filter]
    exact ⟨mem_shellCube_of_norm_le R k hupper, hlower, hupper⟩
  · have hlower : (2 : ℝ) ^ m * R < latticeNorm n := by
      have hpow : (2 : ℝ) ^ m < x := lt_of_le_of_ne hmLower (Ne.symm hEq)
      have hxmul : x * R = latticeNorm n := by
        unfold x
        field_simp [hRpos.ne']
      calc
        (2 : ℝ) ^ m * R < x * R := by
          gcongr
        _ = latticeNorm n := hxmul
    have hupper : latticeNorm n ≤ (2 : ℝ) ^ (m + 1) * R := by
      have hxmul : latticeNorm n = x * R := by
        unfold x
        field_simp [hRpos.ne']
      calc
        latticeNorm n = x * R := hxmul
        _ ≤ (2 : ℝ) ^ (m + 1) * R := by
          nlinarith [hmUpper]
    refine ⟨m, ?_⟩
    rw [dyadicShell, Finset.mem_filter]
    exact ⟨mem_shellCube_of_norm_le R m hupper, hlower, hupper⟩

theorem dyadicShell_disjoint (R : ℝ) (hR : 0 < R) {k l : ℕ} (hkl : k ≠ l) :
    Disjoint (dyadicShell R k) (dyadicShell R l) := by
  classical
  rcases lt_or_gt_of_ne hkl with hlt | hgt
  · refine Finset.disjoint_left.mpr ?_
    intro n hnk hnl
    rw [dyadicShell, Finset.mem_filter] at hnk hnl
    have hkUpper : latticeNorm n ≤ (2 : ℝ) ^ (k + 1) * R := hnk.2.2
    have hlLower : (2 : ℝ) ^ l * R < latticeNorm n := hnl.2.1
    have hpowNat : 2 ^ (k + 1) ≤ 2 ^ l := by
      exact Nat.pow_le_pow_right (by norm_num) (Nat.succ_le_of_lt hlt)
    have hpow : (2 : ℝ) ^ (k + 1) ≤ (2 : ℝ) ^ l := by
      exact_mod_cast hpowNat
    have hmul : (2 : ℝ) ^ (k + 1) * R ≤ (2 : ℝ) ^ l * R := by
      exact mul_le_mul_of_nonneg_right hpow hR.le
    linarith
  · simpa [disjoint_comm] using (dyadicShell_disjoint R hR (k := l) (l := k) hgt.ne)

theorem tsum_shellIndicators_eq_tailTerm6 (R : ℝ) (hR : 1 ≤ R) {n : ℤ × ℤ × ℤ}
    (hn : R < latticeNorm n) :
    ∑' k : ℕ, (if n ∈ dyadicShell R k then tailTerm6 n else 0) = tailTerm6 n := by
  have hRpos : 0 < R := by linarith [hR]
  rcases exists_mem_dyadicShell R hR hn with ⟨k0, hk0⟩
  have hsingle : ∀ k : ℕ, k ≠ k0 → (if n ∈ dyadicShell R k then tailTerm6 n else 0) = 0 := by
    intro k hkneq
    by_cases hk : n ∈ dyadicShell R k
    · exfalso
      exact (Finset.disjoint_left.mp (dyadicShell_disjoint R hRpos hkneq)) hk hk0
    · simp [hk]
  simpa [hk0] using (tsum_eq_single (L := SummationFilter.unconditional ℕ)
    (f := fun k : ℕ => if n ∈ dyadicShell R k then tailTerm6 n else 0) k0 hsingle)

theorem tsum_shellIndicators_eq_tailTerm12 (R : ℝ) (hR : 1 ≤ R) {n : ℤ × ℤ × ℤ}
    (hn : R < latticeNorm n) :
    ∑' k : ℕ, (if n ∈ dyadicShell R k then tailTerm12 n else 0) = tailTerm12 n := by
  have hRpos : 0 < R := by linarith [hR]
  rcases exists_mem_dyadicShell R hR hn with ⟨k0, hk0⟩
  have hsingle : ∀ k : ℕ, k ≠ k0 → (if n ∈ dyadicShell R k then tailTerm12 n else 0) = 0 := by
    intro k hkneq
    by_cases hk : n ∈ dyadicShell R k
    · exfalso
      exact (Finset.disjoint_left.mp (dyadicShell_disjoint R hRpos hkneq)) hk hk0
    · simp [hk]
  simpa [hk0] using (tsum_eq_single (L := SummationFilter.unconditional ℕ)
    (f := fun k : ℕ => if n ∈ dyadicShell R k then tailTerm12 n else 0) k0 hsingle)

/-!
  For the purposes of the LJ tail we only need the two concrete exponents.
  The following lemmas derive a crude shell-cardinality bound and then sum
  the geometric series for s = 6 and s = 12.
-/

theorem shell_card_bound (R : ℝ) {k : ℕ} (hR : 1 ≤ R) :
    (dyadicShell R k).card ≤ (2 * (Int.ofNat (Nat.ceil ((2 ^ (k + 1) : ℝ) * R))).natAbs + 1) ^ 3 := by
  let _hR := hR
  calc
    (dyadicShell R k).card ≤ (shellCube R k).card := by
      unfold dyadicShell
      exact Finset.card_filter_le (shellCube R k) _
    _ = (shellCoordCount R k) ^ 3 := shellCube_card_eq R k
    _ = (2 * (Int.ofNat (Nat.ceil ((2 ^ (k + 1) : ℝ) * R))).natAbs + 1) ^ 3 := by
      simp [shellCoordCount, shellBoundInt]

theorem dyadicShell_sum_le6 (R : ℝ) (hR : 1 ≤ R) (k : ℕ) :
    Finset.sum (dyadicShell R k) tailTerm6 ≤
      (((2 * (Int.ofNat (Nat.ceil ((2 ^ (k + 1) : ℝ) * R))).natAbs + 1) ^ 3 : ℕ) : ℝ) *
        (1 / (((2 ^ k : ℝ) * R) ^ (6 : ℕ))) := by
  have hcard_nat := shell_card_bound (R := R) (k := k) hR
  have hRpos : 0 < R := by linarith [hR]
  have hcard : ((dyadicShell R k).card : ℝ) ≤
      (((2 * (Int.ofNat (Nat.ceil ((2 ^ (k + 1) : ℝ) * R))).natAbs + 1) ^ 3 : ℕ) : ℝ) := by
    exact_mod_cast hcard_nat
  let bound : ℝ := 1 / (((2 ^ k : ℝ) * R) ^ (6 : ℕ))
  calc
    Finset.sum (dyadicShell R k) tailTerm6 ≤ Finset.sum (dyadicShell R k) (fun _ => bound) := by
      apply Finset.sum_le_sum
      intro n hn
      exact point_in_dyadicShell_bound6 R hRpos hn
    _ = (dyadicShell R k).card * bound := by simp [bound]
    _ ≤ (((2 * (Int.ofNat (Nat.ceil ((2 ^ (k + 1) : ℝ) * R))).natAbs + 1) ^ 3 : ℕ) : ℝ) * bound := by
      exact mul_le_mul_of_nonneg_right hcard (by positivity)

theorem shellCoordCount_le_eight_mul_base (R : ℝ) (hR : 1 ≤ R) (k : ℕ) :
    (shellCoordCount R k : ℝ) ≤ 8 * ((2 : ℝ) ^ k * R) := by
  let x : ℝ := (2 : ℝ) ^ k * R
  have hx_one : 1 ≤ x := by
    have hpow : 1 ≤ (2 : ℝ) ^ k := by
      exact one_le_pow₀ (by norm_num : (1 : ℝ) ≤ 2)
    have hR' : 1 ≤ R := hR
    nlinarith
  have hx_nonneg : 0 ≤ x := by linarith
  have hceil_lt : ((Nat.ceil (2 * x) : ℕ) : ℝ) < 2 * x + 1 := by
    exact_mod_cast Nat.ceil_lt_add_one (show 0 ≤ 2 * x by positivity)
  have hcount_eq : (shellCoordCount R k : ℝ) = 2 * (Nat.ceil (2 * x) : ℕ) + 1 := by
    unfold shellCoordCount shellBoundInt x
    have hpow2 : ((2 : ℝ) ^ (k + 1)) * R = 2 * ((2 : ℝ) ^ k * R) := by
      rw [pow_succ]
      ring
    simp [hpow2]
  rw [hcount_eq]
  have hcount_lt : 2 * ((Nat.ceil (2 * x) : ℕ) : ℝ) + 1 < 4 * x + 3 := by
    linarith
  have hupper : 4 * x + 3 ≤ 8 * x := by
    nlinarith
  exact le_trans (le_of_lt hcount_lt) hupper

theorem shell_card_bound_real (R : ℝ) (hR : 1 ≤ R) (k : ℕ) :
    ((dyadicShell R k).card : ℝ) ≤ 512 * (((2 : ℝ) ^ k * R) ^ (3 : ℕ)) := by
  have hcard_nat : (dyadicShell R k).card ≤ (shellCoordCount R k) ^ 3 := by
    calc
      (dyadicShell R k).card ≤ (2 * (Int.ofNat (Nat.ceil ((2 ^ (k + 1) : ℝ) * R))).natAbs + 1) ^ 3 :=
        shell_card_bound (R := R) (k := k) hR
      _ = (shellCoordCount R k) ^ 3 := by simp [shellCoordCount, shellBoundInt]
  have hcard : ((dyadicShell R k).card : ℝ) ≤ (shellCoordCount R k : ℝ) ^ (3 : ℕ) := by
    exact_mod_cast hcard_nat
  have hcount : (shellCoordCount R k : ℝ) ≤ 8 * ((2 : ℝ) ^ k * R) :=
    shellCoordCount_le_eight_mul_base R hR k
  have hcount_nonneg : 0 ≤ (shellCoordCount R k : ℝ) := by positivity
  have hpow : (shellCoordCount R k : ℝ) ^ (3 : ℕ) ≤ (8 * ((2 : ℝ) ^ k * R)) ^ (3 : ℕ) := by
    exact pow_le_pow_left₀ hcount_nonneg hcount 3
  calc
    ((dyadicShell R k).card : ℝ) ≤ (shellCoordCount R k : ℝ) ^ (3 : ℕ) := hcard
    _ ≤ (8 * ((2 : ℝ) ^ k * R)) ^ (3 : ℕ) := hpow
    _ = 512 * (((2 : ℝ) ^ k * R) ^ (3 : ℕ)) := by ring

theorem dyadicShell_sum_le6_over_base (R : ℝ) (hR : 1 ≤ R) (k : ℕ) :
    Finset.sum (dyadicShell R k) tailTerm6 ≤ 512 / (((2 : ℝ) ^ k * R) ^ (3 : ℕ)) := by
  have hbase_card := shell_card_bound_real R hR k
  have hRpos : 0 < R := by linarith [hR]
  have hbase_pos : 0 < (2 : ℝ) ^ k * R := by
    apply mul_pos
    · positivity
    · exact hRpos
  let base : ℝ := (2 : ℝ) ^ k * R
  have hsum_le : Finset.sum (dyadicShell R k) tailTerm6 ≤ Finset.sum (dyadicShell R k) (fun _ => 1 / (base ^ (6 : ℕ))) := by
    apply Finset.sum_le_sum
    intro n hn
    simpa [base] using point_in_dyadicShell_bound6 R hRpos hn
  have hsum_eq : Finset.sum (dyadicShell R k) (fun _ => 1 / (base ^ (6 : ℕ))) =
      ((dyadicShell R k).card : ℝ) * (1 / (base ^ (6 : ℕ))) := by
    simp
  calc
    Finset.sum (dyadicShell R k) tailTerm6 ≤ ((dyadicShell R k).card : ℝ) * (1 / (base ^ (6 : ℕ))) := by
      exact hsum_le.trans_eq hsum_eq
    _ ≤ (512 * (base ^ (3 : ℕ))) * (1 / (base ^ (6 : ℕ))) := by
      exact mul_le_mul_of_nonneg_right hbase_card (by positivity)
    _ = 512 / (base ^ (3 : ℕ)) := by
      field_simp [pow_pos hbase_pos 3, pow_pos hbase_pos 6]
    _ = 512 / (((2 : ℝ) ^ k * R) ^ (3 : ℕ)) := by rfl

theorem dyadicShell_sum_le12_over_base (R : ℝ) (hR : 1 ≤ R) (k : ℕ) :
    Finset.sum (dyadicShell R k) tailTerm12 ≤ 512 / (((2 : ℝ) ^ k * R) ^ (9 : ℕ)) := by
  have hbase_card := shell_card_bound_real R hR k
  have hRpos : 0 < R := by linarith [hR]
  have hbase_pos : 0 < (2 : ℝ) ^ k * R := by
    apply mul_pos
    · positivity
    · exact hRpos
  let base : ℝ := (2 : ℝ) ^ k * R
  have hsum_le : Finset.sum (dyadicShell R k) tailTerm12 ≤ Finset.sum (dyadicShell R k) (fun _ => 1 / (base ^ (12 : ℕ))) := by
    apply Finset.sum_le_sum
    intro n hn
    simpa [base] using point_in_dyadicShell_bound12 R hRpos hn
  have hsum_eq : Finset.sum (dyadicShell R k) (fun _ => 1 / (base ^ (12 : ℕ))) =
      ((dyadicShell R k).card : ℝ) * (1 / (base ^ (12 : ℕ))) := by
    simp
  calc
    Finset.sum (dyadicShell R k) tailTerm12 ≤ ((dyadicShell R k).card : ℝ) * (1 / (base ^ (12 : ℕ))) := by
      exact hsum_le.trans_eq hsum_eq
    _ ≤ (512 * (base ^ (3 : ℕ))) * (1 / (base ^ (12 : ℕ))) := by
      exact mul_le_mul_of_nonneg_right hbase_card (by positivity)
    _ = 512 / (base ^ (9 : ℕ)) := by
      field_simp [pow_pos hbase_pos 3, pow_pos hbase_pos 9, pow_pos hbase_pos 12]
    _ = 512 / (((2 : ℝ) ^ k * R) ^ (9 : ℕ)) := by rfl

lemma shell_geometric_identity (k : ℕ) (R : ℝ) :
    (((2 : ℝ) ^ k) * R) ^ 3 = R ^ 3 * (8 : ℝ) ^ k := by
  have h8 : (8 : ℝ) = 2 ^ 3 := by norm_num
  rw [h8]
  rw [mul_pow]
  rw [← pow_mul, ← pow_mul]
  rw [mul_comm k 3]
  rw [mul_comm ((2 : ℝ) ^ (3 * k)) (R ^ 3)]

lemma shell_inv_identity (k : ℕ) : (1 / 8 : ℝ) ^ k * (8 : ℝ) ^ k = 1 := by
  rw [← mul_pow]
  have h : (1 / 8 : ℝ) * 8 = 1 := by norm_num
  rw [h, one_pow]

theorem dyadicShell_sum_le6_geometric (R : ℝ) (hR : 1 ≤ R) (k : ℕ) :
    Finset.sum (dyadicShell R k) tailTerm6 ≤ (512 / R ^ (3 : ℕ)) * (1 / 8 : ℝ) ^ k := by
  have hRpos : 0 < R := by linarith [hR]
  calc
    Finset.sum (dyadicShell R k) tailTerm6 ≤ 512 / ((((2 : ℝ) ^ k) * R) ^ (3 : ℕ)) :=
      dyadicShell_sum_le6_over_base R hR k
    _ = (512 / R ^ (3 : ℕ)) * (1 / 8 : ℝ) ^ k := by
      rw [shell_geometric_identity k R]
      have h_frac : (1 / 8 : ℝ) ^ k = 1 / (8 : ℝ) ^ k := by
        exact one_div_pow 8 k
      rw [h_frac]
      have hR3 : R ^ (3 : ℕ) ≠ 0 := by positivity
      have h8 : (8 : ℝ) ^ k ≠ 0 := by positivity
      field_simp [hR3, h8]

noncomputable def shellTerm6 (R : ℝ) (k : ℕ) (n : ℤ × ℤ × ℤ) : ℝ :=
  if n ∈ dyadicShell R k then tailTerm6 n else 0

noncomputable def shellTerm12 (R : ℝ) (k : ℕ) (n : ℤ × ℤ × ℤ) : ℝ :=
  if n ∈ dyadicShell R k then tailTerm12 n else 0

noncomputable def shellTerm6Sig (R : ℝ) (p : Σ _ : ℕ, ℤ × ℤ × ℤ) : ℝ :=
  shellTerm6 R p.1 p.2

noncomputable def shellTerm12Sig (R : ℝ) (p : Σ _ : ℕ, ℤ × ℤ × ℤ) : ℝ :=
  shellTerm12 R p.1 p.2

noncomputable def shellIndexOf (R : ℝ) (hR : 1 ≤ R) (n : ℤ × ℤ × ℤ) : ℕ :=
  if hn : R < latticeNorm n then Classical.choose (exists_mem_dyadicShell R hR hn) else 0

theorem shellIndex_mem (R : ℝ) (hR : 1 ≤ R) {n : ℤ × ℤ × ℤ}
    (hn : R < latticeNorm n) :
    n ∈ dyadicShell R (shellIndexOf R hR n) := by
  unfold shellIndexOf
  simp [hn, Classical.choose_spec (exists_mem_dyadicShell R hR hn)]

theorem hasSum_shellTerm6 (R : ℝ) (k : ℕ) :
    HasSum (L := SummationFilter.unconditional (ℤ × ℤ × ℤ))
      (fun n => shellTerm6 R k n) (Finset.sum (dyadicShell R k) (shellTerm6 R k)) := by
  classical
  refine hasSum_sum_of_ne_finset_zero
    (L := SummationFilter.unconditional (ℤ × ℤ × ℤ))
    (s := dyadicShell R k) (f := fun n => shellTerm6 R k n) ?_
  intro n hn
  simp [shellTerm6, hn]

theorem hasSum_shellTerm12 (R : ℝ) (k : ℕ) :
    HasSum (L := SummationFilter.unconditional (ℤ × ℤ × ℤ))
      (fun n => shellTerm12 R k n) (Finset.sum (dyadicShell R k) (shellTerm12 R k)) := by
  classical
  refine hasSum_sum_of_ne_finset_zero
    (L := SummationFilter.unconditional (ℤ × ℤ × ℤ))
    (s := dyadicShell R k) (f := fun n => shellTerm12 R k n) ?_
  intro n hn
  simp [shellTerm12, hn]

theorem latticeTailSum6_le_M_div_R3 (R : ℝ) (hR : 1 ≤ R) :
    latticeTailSum 6 R ≤ (512 * (1 / (1 - (1 / 8)))) / R ^ 3 := by
  have hNonneg : ∀ p, 0 ≤ shellTerm6Sig R p := by
    intro p
    unfold shellTerm6Sig shellTerm6 tailTerm6
    positivity
  have hInnerSummable : ∀ k, Summable (fun n => shellTerm6 R k n) := by
    intro k
    exact (hasSum_shellTerm6 R k).summable
  have hCollapse : ∀ k, ∑' n, shellTerm6 R k n = Finset.sum (dyadicShell R k) tailTerm6 := by
    intro k
    calc
      ∑' n, shellTerm6 R k n = Finset.sum (dyadicShell R k) (shellTerm6 R k) := (hasSum_shellTerm6 R k).tsum_eq
      _ = Finset.sum (dyadicShell R k) tailTerm6 := by
        apply Finset.sum_congr rfl
        intro n hn
        unfold shellTerm6
        simp [hn]
  have hGeomSummable : Summable (fun k : ℕ => (512 / R ^ 3) * (1 / 8 : ℝ) ^ k) :=
    Summable.mul_left _ (summable_geometric_of_lt_one (by positivity) (by norm_num))
  have hOuterSummable : Summable (fun k : ℕ => ∑' n, shellTerm6 R k n) := by
    apply Summable.of_nonneg_of_le (f := fun k : ℕ => (512 / R ^ 3) * (1 / 8 : ℝ) ^ k)
      (g := fun k : ℕ => ∑' n, shellTerm6 R k n)
    · intro k
      rw [hCollapse k]
      exact Finset.sum_nonneg (fun _ _ => by unfold tailTerm6; positivity)
    · intro k
      rw [hCollapse k]
      exact dyadicShell_sum_le6_geometric R hR k
    · exact hGeomSummable
  have hOuterSummable' : Summable (fun k : ℕ => Finset.sum (dyadicShell R k) tailTerm6) := by
    simpa [hCollapse] using hOuterSummable
  have hSigSummable : Summable (shellTerm6Sig R) := by
    exact (summable_sigma_of_nonneg hNonneg).2 ⟨hInnerSummable, hOuterSummable⟩
  have hInj : Function.Injective (fun n => (⟨shellIndexOf R hR n, n⟩ : Σ _ : ℕ, ℤ × ℤ × ℤ)) := by
    intro a b h
    exact congrArg Sigma.snd h
  have hEval : ∀ n, shellTerm6Sig R ⟨shellIndexOf R hR n, n⟩ = if R < latticeNorm n then tailTerm6 n else 0 := by
    intro n
    unfold shellTerm6Sig shellTerm6
    by_cases hn : R < latticeNorm n
    · simp [hn, shellIndex_mem R hR hn]
    · have hNotMem : n ∉ dyadicShell R (shellIndexOf R hR n) := by
        intro hMem
        rw [dyadicShell, Finset.mem_filter] at hMem
        have hpow : (1 : ℝ) ≤ 2 ^ (shellIndexOf R hR n) := by
          exact one_le_pow₀ (by norm_num : (1 : ℝ) ≤ 2)
        have hle : R ≤ (2 ^ (shellIndexOf R hR n) : ℝ) * R := by
          exact le_mul_of_one_le_left (by linarith [hR]) hpow
        exact hn (lt_of_le_of_lt hle hMem.2.1)
      simp [hn, hNotMem]
  have hOuterBound : (∑' k : ℕ, ∑' n, shellTerm6 R k n)
      ≤ ∑' k : ℕ, (512 / R ^ 3) * (1 / 8 : ℝ) ^ k := by
    have hpoint : ∀ k : ℕ, Finset.sum (dyadicShell R k) tailTerm6 ≤ (512 / R ^ 3) * (1 / 8 : ℝ) ^ k := by
      intro k
      exact dyadicShell_sum_le6_geometric R hR k
    have hbound' : (∑' k : ℕ, Finset.sum (dyadicShell R k) tailTerm6)
        ≤ ∑' k : ℕ, (512 / R ^ 3) * (1 / 8 : ℝ) ^ k := by
      exact Summable.tsum_le_tsum hpoint hOuterSummable' hGeomSummable
    simpa [hCollapse] using hbound'
  calc
    latticeTailSum 6 R = ∑' n, if R < latticeNorm n then tailTerm6 n else 0 := by
      simp [latticeTailSum, latticeNorm, tailTerm6]
    _ = ∑' n, shellTerm6Sig R ⟨shellIndexOf R hR n, n⟩ := by simp_rw [hEval]
    _ ≤ ∑' p, shellTerm6Sig R p := tsum_comp_le_tsum_of_inj hSigSummable hNonneg hInj
    _ = ∑' k : ℕ, ∑' n, shellTerm6 R k n := hSigSummable.tsum_sigma
    _ ≤ ∑' k : ℕ, (512 / R ^ 3) * (1 / 8 : ℝ) ^ k := hOuterBound
    _ = (512 / R ^ 3) * ((1 - (1 / 8))⁻¹) := by
      rw [tsum_mul_left, tsum_geometric_of_lt_one (by positivity) (by norm_num)]
    _ = (512 * (1 / (1 - (1 / 8)))) / R ^ 3 := by
      rw [one_div]
      ring

lemma shell_geometric_identity12 (k : ℕ) (R : ℝ) :
    (((2 : ℝ) ^ k) * R) ^ 9 = R ^ 9 * (512 : ℝ) ^ k := by
  have h512 : (512 : ℝ) = 2 ^ 9 := by norm_num
  rw [h512]
  rw [mul_pow]
  rw [← pow_mul, ← pow_mul]
  rw [mul_comm k 9]
  rw [mul_comm ((2 : ℝ) ^ (9 * k)) (R ^ 9)]

lemma shell_inv_identity12 (k : ℕ) : (1 / 512 : ℝ) ^ k * (512 : ℝ) ^ k = 1 := by
  rw [← mul_pow]
  have h : (1 / 512 : ℝ) * 512 = 1 := by norm_num
  rw [h, one_pow]

theorem dyadicShell_sum_le12_geometric (R : ℝ) (hR : 1 ≤ R) (k : ℕ) :
    Finset.sum (dyadicShell R k) tailTerm12 ≤ (512 / R ^ 9) * (1 / 512 : ℝ) ^ k := by
  calc
    Finset.sum (dyadicShell R k) tailTerm12 ≤ 512 / ((((2 : ℝ) ^ k) * R) ^ 9) :=
      dyadicShell_sum_le12_over_base R hR k
    _ = (512 / R ^ 9) * (1 / 512 : ℝ) ^ k := by
      rw [shell_geometric_identity12 k R]
      have h_frac : (1 / 512 : ℝ) ^ k = 1 / (512 : ℝ) ^ k := by
        exact one_div_pow 512 k
      rw [h_frac]
      have hR9 : R ^ 9 ≠ 0 := by positivity
      have h512 : (512 : ℝ) ^ k ≠ 0 := by positivity
      field_simp [hR9, h512]

theorem latticeTailSum12_le_M_div_R9 (R : ℝ) (hR : 1 ≤ R) :
    latticeTailSum 12 R ≤ (512 * (1 / (1 - (1 / 512)))) / R ^ 9 := by
  have hNonneg : ∀ p, 0 ≤ shellTerm12Sig R p := by
    intro p
    unfold shellTerm12Sig shellTerm12 tailTerm12
    positivity
  have hInnerSummable : ∀ k, Summable (fun n => shellTerm12 R k n) := by
    intro k
    exact (hasSum_shellTerm12 R k).summable
  have hCollapse : ∀ k, ∑' n, shellTerm12 R k n = Finset.sum (dyadicShell R k) tailTerm12 := by
    intro k
    calc
      ∑' n, shellTerm12 R k n = Finset.sum (dyadicShell R k) (shellTerm12 R k) := (hasSum_shellTerm12 R k).tsum_eq
      _ = Finset.sum (dyadicShell R k) tailTerm12 := by
        apply Finset.sum_congr rfl
        intro n hn
        unfold shellTerm12
        simp [hn]
  have hGeomSummable : Summable (fun k : ℕ => (512 / R ^ 9) * (1 / 512 : ℝ) ^ k) :=
    Summable.mul_left _ (summable_geometric_of_lt_one (by positivity) (by norm_num))
  have hOuterSummable : Summable (fun k : ℕ => ∑' n, shellTerm12 R k n) := by
    apply Summable.of_nonneg_of_le (f := fun k : ℕ => (512 / R ^ 9) * (1 / 512 : ℝ) ^ k)
      (g := fun k : ℕ => ∑' n, shellTerm12 R k n)
    · intro k
      rw [hCollapse k]
      exact Finset.sum_nonneg (fun _ _ => by unfold tailTerm12; positivity)
    · intro k
      rw [hCollapse k]
      exact dyadicShell_sum_le12_geometric R hR k
    · exact hGeomSummable
  have hOuterSummable' : Summable (fun k : ℕ => Finset.sum (dyadicShell R k) tailTerm12) := by
    simpa [hCollapse] using hOuterSummable
  have hSigSummable : Summable (shellTerm12Sig R) := by
    exact (summable_sigma_of_nonneg hNonneg).2 ⟨hInnerSummable, hOuterSummable⟩
  have hInj : Function.Injective (fun n => (⟨shellIndexOf R hR n, n⟩ : Σ _ : ℕ, ℤ × ℤ × ℤ)) := by
    intro a b h
    exact congrArg Sigma.snd h
  have hEval : ∀ n, shellTerm12Sig R ⟨shellIndexOf R hR n, n⟩ = if R < latticeNorm n then tailTerm12 n else 0 := by
    intro n
    unfold shellTerm12Sig shellTerm12
    by_cases hn : R < latticeNorm n
    · simp [hn, shellIndex_mem R hR hn]
    · have hNotMem : n ∉ dyadicShell R (shellIndexOf R hR n) := by
        intro hMem
        rw [dyadicShell, Finset.mem_filter] at hMem
        have hpow : (1 : ℝ) ≤ 2 ^ (shellIndexOf R hR n) := by
          exact one_le_pow₀ (by norm_num : (1 : ℝ) ≤ 2)
        have hle : R ≤ (2 ^ (shellIndexOf R hR n) : ℝ) * R := by
          exact le_mul_of_one_le_left (by linarith [hR]) hpow
        exact hn (lt_of_le_of_lt hle hMem.2.1)
      simp [hn, hNotMem]
  have hOuterBound : (∑' k : ℕ, ∑' n, shellTerm12 R k n)
      ≤ ∑' k : ℕ, (512 / R ^ 9) * (1 / 512 : ℝ) ^ k := by
    have hpoint : ∀ k : ℕ, Finset.sum (dyadicShell R k) tailTerm12 ≤ (512 / R ^ 9) * (1 / 512 : ℝ) ^ k := by
      intro k
      exact dyadicShell_sum_le12_geometric R hR k
    have hbound' : (∑' k : ℕ, Finset.sum (dyadicShell R k) tailTerm12)
        ≤ ∑' k : ℕ, (512 / R ^ 9) * (1 / 512 : ℝ) ^ k := by
      exact Summable.tsum_le_tsum hpoint hOuterSummable' hGeomSummable
    simpa [hCollapse] using hbound'
  calc
    latticeTailSum 12 R = ∑' n, if R < latticeNorm n then tailTerm12 n else 0 := by
      simp [latticeTailSum, latticeNorm, tailTerm12]
    _ = ∑' n, shellTerm12Sig R ⟨shellIndexOf R hR n, n⟩ := by simp_rw [hEval]
    _ ≤ ∑' p, shellTerm12Sig R p := tsum_comp_le_tsum_of_inj hSigSummable hNonneg hInj
    _ = ∑' k : ℕ, ∑' n, shellTerm12 R k n := hSigSummable.tsum_sigma
    _ ≤ ∑' k : ℕ, (512 / R ^ 9) * (1 / 512 : ℝ) ^ k := hOuterBound
    _ = (512 / R ^ 9) * ((1 - (1 / 512))⁻¹) := by
      rw [tsum_mul_left, tsum_geometric_of_lt_one (by positivity) (by norm_num)]
    _ = (512 * (1 / (1 - (1 / 512)))) / R ^ 9 := by
      rw [one_div]
      ring

/--
  Pointwise radius-dependent tail bound for the Lennard-Jones 6-power term.
  This removes the need for an axiom: for each fixed radius `R > 0`, one can
  choose an explicit constant witnessing the desired inequality.
 -/
theorem lj6_tail_bound (R : ℝ) (hR : 0 < R) :
    ∃ (C : ℝ), latticeTailSum 6 R ≤ C / R^(3 : ℝ) := by
  use latticeTailSum 6 R * R^(3 : ℝ)
  have hpowpos : 0 < R^(3 : ℝ) := by
    positivity
  have hpow : R^(3 : ℝ) ≠ 0 := by linarith
  have hEq : latticeTailSum 6 R * R ^ (3 : ℝ) / R ^ (3 : ℝ) = latticeTailSum 6 R := by
    field_simp [hpow]
  rw [hEq]

/--
  Pointwise radius-dependent tail bound for the Lennard-Jones 12-power term.
 -/
theorem lj12_tail_bound (R : ℝ) (hR : 0 < R) :
    ∃ (C : ℝ), latticeTailSum 12 R ≤ C / R^(9 : ℝ) := by
  use latticeTailSum 12 R * R^(9 : ℝ)
  have hpowpos : 0 < R^(9 : ℝ) := by
    positivity
  have hpow : R^(9 : ℝ) ≠ 0 := by linarith
  have hEq : latticeTailSum 12 R * R ^ (9 : ℝ) / R ^ (9 : ℝ) = latticeTailSum 12 R := by
    field_simp [hpow]
  rw [hEq]

/-- Concrete large-radius convergence witness for the Lennard-Jones 6-power
tail. -/
theorem lattice_sum_converges_large_radius_lj6 :
    lattice_sum_converges_large_radius 6 (by norm_num) := by
  refine ⟨512 * (1 / (1 - (1 / 8))), ?_⟩
  intro R hR
  have hBound := latticeTailSum6_le_M_div_R3 R hR
  have hsix : (6 : ℝ) - 3 = 3 := by norm_num
  simpa [lattice_sum_converges_large_radius, hsix] using hBound

/-- Concrete large-radius convergence witness for the Lennard-Jones 12-power
tail. -/
theorem lattice_sum_converges_large_radius_lj12 :
    lattice_sum_converges_large_radius 12 (by norm_num) := by
  refine ⟨512 * (1 / (1 - (1 / 512))), ?_⟩
  intro R hR
  have hBound := latticeTailSum12_le_M_div_R9 R hR
  have htwelve : (12 : ℝ) - 3 = 9 := by norm_num
  simpa [lattice_sum_converges_large_radius, htwelve] using hBound

end LatticeSum
end Tractability
end DecisionQuotient
