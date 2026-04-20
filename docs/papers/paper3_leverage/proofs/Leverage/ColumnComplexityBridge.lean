import Mathlib.Data.Finsupp.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Order.Preorder.Finite
import Paper1IT.GraphEntropy

open scoped BigOperators

namespace Leverage
namespace ColumnComplexityBridge

open Ssot.GraphEntropy

section Scaffolding

variable {SourceMsg Codeword : Type*}
variable [Fintype SourceMsg] [Fintype Codeword]
variable [DecidableEq SourceMsg] [DecidableEq Codeword]

/-- The paper1 finite fiber geometry, reused here with `Enc` interpreted as the tie-broken
encoder induced by a compression Hamiltonian. -/
noncomputable abbrev Fiber (Enc : SourceMsg → Codeword) (cw : Codeword) : Finset SourceMsg :=
  observeFiber Enc cw

/-- Finite cardinality of the encoder fiber over a codeword. -/
noncomputable def FiberCard (Enc : SourceMsg → Codeword) (cw : Codeword) : ℕ :=
  Fintype.card {c : SourceMsg // Enc c = cw}

/-- The scaled log-fiber size `ρ = (1/N) * log |F_E(c̄)|` at finite block size `N`. -/
noncomputable def ScaledLogFiberSize (N : ℕ) (Enc : SourceMsg → Codeword) (cw : Codeword) : ℝ :=
  (N : ℝ)⁻¹ * Real.log (FiberCard Enc cw : ℝ)

/-- The finite grid of scaled log-fiber sizes realized by the encoder. -/
noncomputable def ColOccupancyGrid (N : ℕ) (Enc : SourceMsg → Codeword) : Finset ℝ :=
  Finset.univ.image (ScaledLogFiberSize N Enc)

/-- Raw multiplicity of a scaled log-fiber size in the codeword ensemble. -/
noncomputable def ColOccupancyWeight (N : ℕ) (Enc : SourceMsg → Codeword) (ρ : ℝ) : ℕ :=
  (Finset.univ.filter fun cw : Codeword => ScaledLogFiberSize N Enc cw = ρ).card

/-- The empirical column-occupancy measure over the finite `ρ`-grid, represented as a finitely
supported function. -/
noncomputable def ColOccupancyMeasure (N : ℕ) (Enc : SourceMsg → Codeword) : ℝ →₀ ℕ :=
  Finsupp.onFinset
    (ColOccupancyGrid N Enc)
    (ColOccupancyWeight N Enc)
    (by
      intro ρ hρ
      rcases Finset.card_ne_zero.mp hρ with ⟨cw, hcw⟩
      exact Finset.mem_image.mpr ⟨cw, Finset.mem_univ _, (Finset.mem_filter.mp hcw).2⟩)

/-- The finite-`N` column moment `∑_{c̄} |F_E(c̄)|^s`. -/
noncomputable def FiberMoment (Enc : SourceMsg → Codeword) (s : ℕ) : ℕ :=
  ∑ cw : Codeword, (FiberCard Enc cw) ^ s

/-- The finite-`N` column moment potential
`Φ_col(s) = (1/N) * log (∑_{c̄} |F_E(c̄)|^s)`. -/
noncomputable def FiberMomentPotential (N : ℕ) (Enc : SourceMsg → Codeword) (s : ℕ) : ℝ :=
  (N : ℝ)⁻¹ * Real.log (FiberMoment Enc s : ℝ)

/-- Direct combinatorial model of source `s`-tuples colliding on a shared codeword. -/
def SharedCodewordFamilies (Enc : SourceMsg → Codeword) (s : ℕ) : Type _ :=
  Σ cw : Codeword, (Fin s → {c : SourceMsg // Enc c = cw})

noncomputable instance sharedCodewordFamiliesFintype
    (Enc : SourceMsg → Codeword) (s : ℕ) :
    Fintype (SharedCodewordFamilies Enc s) := by
  classical
  unfold SharedCodewordFamilies
  infer_instance

/-- Finite combinatorial count of source `s`-tuples that collide on a common codeword. -/
noncomputable def SharedCodewordCount (Enc : SourceMsg → Codeword) (s : ℕ) : ℕ :=
  Fintype.card (SharedCodewordFamilies Enc s)

theorem fiberCard_eq_subtypeCard (Enc : SourceMsg → Codeword) (cw : Codeword) :
    FiberCard Enc cw = Fintype.card {c : SourceMsg // Enc c = cw} := by
  rfl

/-- Exact finite combinatorial bridge: shared-codeword tuple counts equal fiber moments. -/
theorem SharedCodewordCount_eq_FiberMoment (Enc : SourceMsg → Codeword) (s : ℕ) :
    SharedCodewordCount Enc s = FiberMoment Enc s := by
  classical
  unfold SharedCodewordCount SharedCodewordFamilies FiberMoment
  rw [Fintype.card_sigma]
  refine Finset.sum_congr rfl ?_
  intro cw _
  rw [Fintype.card_fun, Fintype.card_fin, ← fiberCard_eq_subtypeCard]

/-- Finite zero identity debt means every encoder fiber fits into the available identity budget. -/
def ZeroIdentityDebt (Enc : SourceMsg → Codeword) (bits : ℕ) : Prop :=
  ∀ cw : Codeword, FiberCard Enc cw ≤ 2 ^ bits

theorem zeroIdentityDebt_iff_uniform_fiber_bound
    (Enc : SourceMsg → Codeword) (bits : ℕ) :
    ZeroIdentityDebt Enc bits ↔ ∀ cw : Codeword, FiberCard Enc cw ≤ 2 ^ bits := by
  rfl

theorem positiveIdentityDebt_iff_exists_large_fiber
    (Enc : SourceMsg → Codeword) (bits : ℕ) :
    ¬ ZeroIdentityDebt Enc bits ↔ ∃ cw : Codeword, 2 ^ bits < FiberCard Enc cw := by
  rw [zeroIdentityDebt_iff_uniform_fiber_bound]
  constructor
  · intro h
    by_contra hNo
    apply h
    intro cw
    by_contra hcw
    exact hNo ⟨cw, lt_of_not_ge hcw⟩
  · intro hExists hAll
    rcases hExists with ⟨cw, hcw⟩
    exact not_lt_of_ge (hAll cw) hcw

/-- Finite scaffold column-complexity profile:
scaled log multiplicity of each occupancy level `ρ` in the encoder column ensemble. -/
noncomputable def ColComplexity (N : ℕ) (Enc : SourceMsg → Codeword) : ℝ → ℝ :=
  fun ρ => (N : ℝ)⁻¹ * Real.log (ColOccupancyWeight N Enc ρ : ℝ)

/-- The conjectural identity-debt threshold
`R_id★ = sup {ρ : ColComplexity ρ ≥ 0}` at finite scaffold level. -/
noncomputable def IdentityDebtThreshold (N : ℕ) (Enc : SourceMsg → Codeword) : ℝ :=
  sSup {ρ : ℝ | 0 ≤ ColComplexity N Enc ρ}

end Scaffolding

section ArgminCompression

variable {SourceMsg Codeword : Type*}
variable [Fintype SourceMsg] [Fintype Codeword] [Nonempty Codeword]
variable [DecidableEq SourceMsg] [DecidableEq Codeword] [LinearOrder Codeword]

/-- The raw compression relation induced by a finite Hamiltonian: `cw` is admissible for source
`c` exactly when it minimizes `H c ·`. -/
def ArgminRelation (H : SourceMsg → Codeword → ℝ) (c : SourceMsg) (cw : Codeword) : Prop :=
  ∀ cw' : Codeword, H c cw ≤ H c cw'

/-- The finite argmin set of the compression Hamiltonian at source `c`. -/
noncomputable def ArgminCodewords
    (H : SourceMsg → Codeword → ℝ) (c : SourceMsg) : Finset Codeword := by
  classical
  exact Finset.univ.filter (ArgminRelation H c)

theorem argminCodewords_nonempty
    (H : SourceMsg → Codeword → ℝ) (c : SourceMsg) :
    (ArgminCodewords H c).Nonempty := by
  classical
  obtain ⟨cw, hcw⟩ :=
    Finset.exists_minimalFor (f := H c) (s := Finset.univ) Finset.univ_nonempty
  refine ⟨cw, by
    simp [ArgminCodewords, ArgminRelation]
    intro cw'
    by_cases hle : H c cw' ≤ H c cw
    · exact hcw.le_of_le (by simp) hle
    · exact le_of_not_ge hle⟩

/-- Deterministic tie-broken encoder: choose the least codeword among the Hamiltonian minimizers. -/
noncomputable def TieBrokenArgminEncoder
    (H : SourceMsg → Codeword → ℝ) : SourceMsg → Codeword := by
  classical
  exact fun c => (ArgminCodewords H c).min' (argminCodewords_nonempty H c)

/-- Direct compression-language relation for the least Hamiltonian minimizer. -/
def TieBrokenArgminRelation
    (H : SourceMsg → Codeword → ℝ) (c : SourceMsg) (cw : Codeword) : Prop :=
  ArgminRelation H c cw ∧
    ∀ cw' : Codeword, cw' < cw → ¬ ArgminRelation H c cw'

theorem tieBrokenArgminEncoder_argmin
    (H : SourceMsg → Codeword → ℝ) (c : SourceMsg) :
    ArgminRelation H c (TieBrokenArgminEncoder H c) := by
  classical
  have hmem : TieBrokenArgminEncoder H c ∈ ArgminCodewords H c := by
    simpa [TieBrokenArgminEncoder] using
      (Finset.min'_mem (s := ArgminCodewords H c) (H := argminCodewords_nonempty H c))
  have hmem' : TieBrokenArgminEncoder H c ∈ Finset.univ.filter (ArgminRelation H c) := by
    simpa [ArgminCodewords] using hmem
  exact (Finset.mem_filter.mp hmem').2

/-- Exact bridge between the least-minimizer compression relation and the deterministic tie-broken
encoder it induces. -/
theorem tieBrokenArgminRelation_iff_encoder_eq
    (H : SourceMsg → Codeword → ℝ) (c : SourceMsg) (cw : Codeword) :
    TieBrokenArgminRelation H c cw ↔ TieBrokenArgminEncoder H c = cw := by
  classical
  constructor
  · intro hrel
    apply le_antisymm
    · change (ArgminCodewords H c).min' (argminCodewords_nonempty H c) ≤ cw
      exact Finset.min'_le _ _ (by simpa [ArgminCodewords] using
        (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hrel.1⟩))
    · change cw ≤ (ArgminCodewords H c).min' (argminCodewords_nonempty H c)
      exact Finset.le_min' (s := ArgminCodewords H c) (H := argminCodewords_nonempty H c) cw <| by
        intro y hy
        have hy' : y ∈ Finset.univ.filter (ArgminRelation H c) := by
          simpa [ArgminCodewords] using hy
        have hyArg : ArgminRelation H c y := (Finset.mem_filter.mp hy').2
        exact le_of_not_gt (fun hylt => hrel.2 y hylt hyArg)
  · intro hEq
    subst hEq
    constructor
    · exact tieBrokenArgminEncoder_argmin H c
    · intro cw' hlt hcw'
      have hcw'mem : cw' ∈ ArgminCodewords H c :=
        by simpa [ArgminCodewords] using (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hcw'⟩)
      have hmin : TieBrokenArgminEncoder H c ≤ cw' := by
        change (ArgminCodewords H c).min' (argminCodewords_nonempty H c) ≤ cw'
        exact Finset.min'_le _ _ hcw'mem
      exact (not_lt_of_ge hmin) hlt

/-- Finite raw argmin-fiber size above a codeword. -/
noncomputable def ArgminRelationFiberCard
    (H : SourceMsg → Codeword → ℝ) (cw : Codeword) : ℕ := by
  classical
  exact Fintype.card {c : SourceMsg // ArgminRelation H c cw}

/-- Finite least-minimizer fiber size above a codeword. -/
noncomputable def TieBrokenRelationFiberCard
    (H : SourceMsg → Codeword → ℝ) (cw : Codeword) : ℕ := by
  classical
  exact Fintype.card {c : SourceMsg // TieBrokenArgminRelation H c cw}

/-- The finite collision moment of the least-minimizer compression relation. -/
noncomputable def TieBrokenRelationMoment
    (H : SourceMsg → Codeword → ℝ) (s : ℕ) : ℕ :=
  ∑ cw : Codeword, (TieBrokenRelationFiberCard H cw) ^ s

theorem tieBrokenRelationFiberCard_le_argminRelationFiberCard
    (H : SourceMsg → Codeword → ℝ) (cw : Codeword) :
    TieBrokenRelationFiberCard H cw ≤ ArgminRelationFiberCard H cw := by
  classical
  unfold TieBrokenRelationFiberCard ArgminRelationFiberCard
  exact Fintype.card_subtype_mono
    (fun c : SourceMsg => TieBrokenArgminRelation H c cw)
    (fun c : SourceMsg => ArgminRelation H c cw)
    (by intro c hc; exact hc.1)

theorem tieBrokenRelationFiberCard_eq_FiberCard
    (H : SourceMsg → Codeword → ℝ) (cw : Codeword) :
    TieBrokenRelationFiberCard H cw = FiberCard (TieBrokenArgminEncoder H) cw := by
  classical
  unfold TieBrokenRelationFiberCard FiberCard
  let e : {c : SourceMsg // TieBrokenArgminRelation H c cw} ≃
      {c : SourceMsg // TieBrokenArgminEncoder H c = cw} :=
    Equiv.subtypeEquivRight (fun c => tieBrokenArgminRelation_iff_encoder_eq (H := H) (c := c) (cw := cw))
  exact Fintype.card_congr e

/-- Exact finite compression-language bridge: the shared-codeword collision count of the
tie-broken Hamiltonian encoder is the least-minimizer relation moment. -/
theorem SharedCodewordCount_eq_TieBrokenRelationMoment
    (H : SourceMsg → Codeword → ℝ) (s : ℕ) :
    SharedCodewordCount (TieBrokenArgminEncoder H) s = TieBrokenRelationMoment H s := by
  classical
  rw [SharedCodewordCount_eq_FiberMoment]
  unfold TieBrokenRelationMoment FiberMoment
  refine Finset.sum_congr rfl ?_
  intro cw _
  rw [← tieBrokenRelationFiberCard_eq_FiberCard]

theorem zeroIdentityDebt_tieBrokenArgmin_iff_uniform_relation_bound
    (H : SourceMsg → Codeword → ℝ) (bits : ℕ) :
    ZeroIdentityDebt (TieBrokenArgminEncoder H) bits ↔
      ∀ cw : Codeword, TieBrokenRelationFiberCard H cw ≤ 2 ^ bits := by
  unfold ZeroIdentityDebt
  constructor <;> intro h cw <;>
    simpa [tieBrokenRelationFiberCard_eq_FiberCard] using h cw

/-- A uniform raw argmin-fiber bound is enough to force zero identity debt for the induced
deterministic tie-broken encoder. -/
theorem zeroIdentityDebt_tieBrokenArgmin_of_uniform_argmin_relation_bound
    (H : SourceMsg → Codeword → ℝ) (bits : ℕ)
    (hbound : ∀ cw : Codeword, ArgminRelationFiberCard H cw ≤ 2 ^ bits) :
    ZeroIdentityDebt (TieBrokenArgminEncoder H) bits := by
  rw [zeroIdentityDebt_tieBrokenArgmin_iff_uniform_relation_bound]
  intro cw
  exact (tieBrokenRelationFiberCard_le_argminRelationFiberCard H cw).trans (hbound cw)

end ArgminCompression

end ColumnComplexityBridge
end Leverage
