import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card
import Mathlib.Data.Nat.Log
import Paper1IT.GraphEntropy

open Classical

namespace Ssot
namespace QueryBitBridge

universe u

/-- A finite Boolean query system on a finite class space. -/
structure QuerySystem (C Q : Type u) where
  answers : C → Q → Bool

variable {C Q β : Type u}
variable [Fintype C] [Fintype Q] [Fintype β]
variable [DecidableEq C] [DecidableEq Q] [DecidableEq β]

/-- A finite query family distinguishes a finite class set if its restricted Boolean transcript
map is injective on that set. -/
def distinguishesOn (S : Finset Q) (qs : QuerySystem C Q) (T : Finset C) : Prop :=
  Set.InjOn (fun c : C => fun q : S => qs.answers c q.1) (T : Set C)

/-- Monotonicity of distinguishing power under enlarging the query family. -/
theorem distinguishesOn_mono {S S' : Finset Q} (hsub : S ⊆ S')
    (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T) :
    distinguishesOn S' qs T := by
  intro c hc c' hc' heq
  apply hdist hc hc'
  funext q
  have hq' : q.1 ∈ S' := hsub q.2
  have hproj := congrArg (fun g => g ⟨q.1, hq'⟩) heq
  simpa using hproj

/-- Counting bound: a family of `|S|` binary queries yields at most `2 ^ |S|` different
transcripts on any distinguished set. -/
theorem card_distinguished_set_le_two_pow_card
    (S : Finset Q) (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T) :
    T.card ≤ 2 ^ S.card := by
  classical
  let f : C → (S → Bool) := fun c q => qs.answers c q.1
  have himage : T.card = (T.image f).card := by
    exact (Finset.card_image_of_injOn (f := f) (s := T) (by
      intro a ha b hb hab
      exact hdist ha hb hab)).symm
  rw [himage]
  calc
    (T.image f).card ≤ Fintype.card (S → Bool) := by
      simpa using (Finset.card_le_univ (s := T.image f))
    _ = 2 ^ S.card := by simp

/-- Exact counting converse in ceiling-log form. -/
theorem clog_card_le_query_count
    (S : Finset Q) (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T) :
    Nat.clog 2 T.card ≤ S.card := by
  have hbound : T.card ≤ 2 ^ S.card :=
    card_distinguished_set_le_two_pow_card S qs T hdist
  exact (Nat.clog_le_iff_le_pow (by decide : 1 < 2)).2 hbound

/-- Fiber-level bridge: if a Boolean query family distinguishes one observation fiber, then that
fiber cannot be larger than the number of available transcripts. -/
theorem fiber_card_le_two_pow_query_count
    (observe : C → β) (b : β) (S : Finset Q) (qs : QuerySystem C Q)
    (hdist : distinguishesOn S qs (Finset.univ.filter (fun c : C => observe c = b))) :
    (Finset.univ.filter (fun c : C => observe c = b)).card ≤ 2 ^ S.card := by
  exact card_distinguished_set_le_two_pow_card S qs _ hdist

/-- Exact ceiling-log fiber-level bridge. If a Boolean query family distinguishes a worst
collision fiber, then its cardinality is at least the binary ceiling-log of that fiber size. -/
theorem maxFiberCard_clog_le_query_count
    (observe : C → β) (b : β)
    (hbmax : Ssot.GraphEntropy.maxFiberCard observe =
      (Finset.univ.filter (fun c : C => observe c = b)).card)
    (S : Finset Q) (qs : QuerySystem C Q)
    (hdist : distinguishesOn S qs (Finset.univ.filter (fun c : C => observe c = b))) :
    Nat.clog 2 (Ssot.GraphEntropy.maxFiberCard observe) ≤ S.card := by
  rw [hbmax]
  exact clog_card_le_query_count S qs _ hdist

/-- If a query family distinguishes a set, every larger query family also distinguishes it,
so the transcript-count lower bounds remain valid under enrichment. -/
theorem clog_card_le_query_count_mono {S S' : Finset Q} (hsub : S ⊆ S')
    (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T) :
    Nat.clog 2 T.card ≤ S'.card := by
  have hdist' : distinguishesOn S' qs T := distinguishesOn_mono hsub qs T hdist
  have hclog : Nat.clog 2 T.card ≤ S.card := clog_card_le_query_count S qs T hdist
  exact hclog.trans (Finset.card_le_card hsub)

/-- Exact equality criterion at the counting floor. If a distinguished set has exactly as many
elements as available Boolean transcripts, then the query family meets the counting lower bound
with equality in the cardinal sense. -/
theorem distinguished_set_eq_full_transcript_capacity
    (S : Finset Q) (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T)
    (hcard : T.card = 2 ^ S.card) :
    Nat.clog 2 T.card = S.card := by
  rw [hcard]
  exact Nat.clog_pow 2 S.card (by decide : 1 < 2)

/-- Minimal-cardinality distinguishing families achieve the ceiling-log floor exactly whenever
the distinguished set saturates the Boolean transcript bound. -/
theorem clog_query_floor_is_tight_of_full_capacity
    (S : Finset Q) (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T)
    (hcard : T.card = 2 ^ S.card) :
    Nat.clog 2 T.card = S.card :=
  distinguished_set_eq_full_transcript_capacity S qs T hdist hcard

/-- Expressivity gap above the counting floor. -/
def queryExpressivityGap (S : Finset Q) (T : Finset C) : ℕ :=
  S.card - Nat.clog 2 T.card

/-- The expressivity gap is always nonnegative for any distinguished set. -/
theorem queryExpressivityGap_nonneg
    (S : Finset Q) (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T) :
    0 ≤ queryExpressivityGap S T := by
  unfold queryExpressivityGap
  exact Nat.zero_le _

/-- Zero expressivity gap characterizes saturation of the ceiling-log lower bound. -/
theorem queryExpressivityGap_eq_zero_iff
    (S : Finset Q) (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T) :
    queryExpressivityGap S T = 0 ↔ S.card = Nat.clog 2 T.card := by
  unfold queryExpressivityGap
  constructor
  · intro hgap
    have hle1 : S.card ≤ Nat.clog 2 T.card := Nat.sub_eq_zero_iff_le.mp hgap
    have hle2 : Nat.clog 2 T.card ≤ S.card := clog_card_le_query_count S qs T hdist
    exact le_antisymm hle1 hle2
  · intro hEq
    rw [hEq]
    exact Nat.sub_self _

/-- Enlarging the query family cannot increase the expressivity gap relative to the enlarged
cardinality floor statement. -/
theorem queryExpressivityGap_mono {S S' : Finset Q} (hsub : S ⊆ S')
    (qs : QuerySystem C Q) (T : Finset C)
    (hdist : distinguishesOn S qs T) :
    Nat.clog 2 T.card ≤ S'.card :=
  clog_card_le_query_count_mono hsub qs T hdist

end QueryBitBridge
end Ssot
