import DecisionQuotient.Tractability.TreeStructure
import DecisionQuotient.Tractability.Dimensional
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient
open Classical

/-- The binary state with coordinates `i` and `j` set to `x` and `y`, and every
other coordinate set to `0`. -/
def pairState {n : ℕ} (i j : Fin n) (x y : Fin 2) : Fin n → Fin 2 :=
  fun k => if k = i then x else if k = j then y else 0

/-- Canonical binary mixed difference of a utility along coordinates `i` and `j`.
This vanishes for unary-only utilities and detects genuine pair interaction in
the binary pairwise setting. -/
def pairCrossDifference {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (a : A) (i j : Fin n) : ℤ :=
  u a (pairState i j 0 0) + u a (pairState i j 1 1) -
    u a (pairState i j 0 1) - u a (pairState i j 1 0)

/-- Utility-level notion of a genuine binary pair interaction. -/
def HasBinaryPairInteraction {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (i j : Fin n) : Prop :=
  ∃ a : A, pairCrossDifference u a i j ≠ 0

/-- Unary-coordinate decomposition: the utility is a sum of single-coordinate
terms, with no genuine pair interaction left. -/
def UnaryCoordinateDecomposition {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) : Prop :=
  ∃ unary : Fin n → A → Fin 2 → ℤ, ∀ a s, u a s = ∑ i : Fin n, unary i a (s i)

/-- Mixed difference of a binary table. -/
def binaryCrossDifference (g : Fin 2 → Fin 2 → ℤ) : ℤ :=
  g 0 0 + g 1 1 - g 0 1 - g 1 0

theorem pairState_self_left {n : ℕ} (i j : Fin n) (x y : Fin 2) :
    pairState i j x y i = x := by
  unfold pairState
  simp

theorem pairState_self_right {n : ℕ} (i j : Fin n) (hij : i ≠ j) (x y : Fin 2) :
    pairState i j x y j = y := by
  have hji : j ≠ i := hij.symm
  unfold pairState
  simp [hji]

theorem pairState_other {n : ℕ} (i j k : Fin n) (hki : k ≠ i) (hkj : k ≠ j)
    (x y : Fin 2) :
    pairState i j x y k = 0 := by
  unfold pairState
  simp [hki, hkj]

theorem pairCrossDifference_self {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (a : A) (i : Fin n) :
    pairCrossDifference u a i i = 0 := by
  have h01 : pairState i i 0 1 = pairState i i 0 0 := by
    funext k
    by_cases hk : k = i <;> simp [pairState, hk]
  have h10 : pairState i i 1 0 = pairState i i 1 1 := by
    funext k
    by_cases hk : k = i <;> simp [pairState, hk]
  unfold pairCrossDifference
  rw [h01, h10]
  ring

theorem pairState_swap_of_ne {n : ℕ} (i j : Fin n) (hij : i ≠ j) (x y : Fin 2) :
    pairState j i y x = pairState i j x y := by
  funext k
  by_cases hki : k = i
  · by_cases hkj : k = j
    · exact False.elim (hij (hki.symm.trans hkj))
    · have hij' : i ≠ j := by
        intro hEq
        exact hkj (hki.trans hEq)
      simp [pairState, hki, hij']
  · by_cases hkj : k = j
    · subst hkj
      simp [pairState, hki]
    · simp [pairState, hki, hkj]

theorem pairCrossDifference_comm {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (a : A) (i j : Fin n) :
    pairCrossDifference u a i j = pairCrossDifference u a j i := by
  by_cases hij : i = j
  · subst hij
    simp [pairCrossDifference_self]
  · unfold pairCrossDifference
    rw [pairState_swap_of_ne (i := i) (j := j) hij (x := 0) (y := 0)]
    rw [pairState_swap_of_ne (i := i) (j := j) hij (x := 1) (y := 1)]
    rw [pairState_swap_of_ne (i := i) (j := j) hij (x := 1) (y := 0)]
    rw [pairState_swap_of_ne (i := i) (j := j) hij (x := 0) (y := 1)]
    ring

theorem HasBinaryPairInteraction_symm {A : Type*} {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ} :
    ∀ i j, HasBinaryPairInteraction u i j → HasBinaryPairInteraction u j i := by
  intro i j h
  rcases h with ⟨a, ha⟩
  refine ⟨a, ?_⟩
  rw [pairCrossDifference_comm u a j i]
  exact ha

def genuineInteractionGraph {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) : SimpleGraph (Fin n) :=
  InteractionGraph (HasBinaryPairInteraction u) (HasBinaryPairInteraction_symm (u := u))

theorem pairState_permute {n : ℕ} (σ : CoordinatePermutation n) (i j : Fin n)
    (x y : Fin 2) :
    (DimensionalStateSpace.permute σ ⟨pairState i j x y⟩).state =
      pairState (σ i) (σ j) x y := by
  funext k
  by_cases hki : k = σ i
  · have hs : σ.symm k = i := by simpa using congrArg σ.symm hki
    simp [DimensionalStateSpace.permute, pairState, hki, hs]
  · have hs : σ.symm k ≠ i := by
      intro h
      apply hki
      simpa using congrArg σ h
    by_cases hkj : k = σ j
    · have hsj : σ.symm k = j := by simpa using congrArg σ.symm hkj
      simp [DimensionalStateSpace.permute, pairState, hki, hkj, hs, hsj]
    · have hsj : σ.symm k ≠ j := by
        intro h
        apply hkj
        simpa using congrArg σ h
      simp [DimensionalStateSpace.permute, pairState, hki, hkj, hs, hsj]

theorem pairCrossDifference_perm {A : Type*} {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ}
    (hsym : SymmetricUtility (fun a s => u a s.state))
    (σ : CoordinatePermutation n) (a : A) (i j : Fin n) :
    pairCrossDifference u a (σ i) (σ j) = pairCrossDifference u a i j := by
  have h00 : u a (pairState (σ i) (σ j) 0 0) = u a (pairState i j 0 0) := by
    symm
    simpa [pairState_permute] using hsym σ a (⟨pairState i j 0 0⟩ : DimensionalStateSpace 2 n)
  have h11 : u a (pairState (σ i) (σ j) 1 1) = u a (pairState i j 1 1) := by
    symm
    simpa [pairState_permute] using hsym σ a (⟨pairState i j 1 1⟩ : DimensionalStateSpace 2 n)
  have h01 : u a (pairState (σ i) (σ j) 0 1) = u a (pairState i j 0 1) := by
    symm
    simpa [pairState_permute] using hsym σ a (⟨pairState i j 0 1⟩ : DimensionalStateSpace 2 n)
  have h10 : u a (pairState (σ i) (σ j) 1 0) = u a (pairState i j 1 0) := by
    symm
    simpa [pairState_permute] using hsym σ a (⟨pairState i j 1 0⟩ : DimensionalStateSpace 2 n)
  unfold pairCrossDifference
  linarith

/-- A permutation sending the ordered pair `(i,j)` to `(p,q)`. -/
def transportPairPerm {n : ℕ} (i j p q : Fin n) : CoordinatePermutation n :=
  let r := (Equiv.swap i p) j
  (Equiv.swap i p).trans (Equiv.swap r q)

theorem transportPairPerm_aux_ne {n : ℕ} (i j p : Fin n) (hij : i ≠ j) :
    (Equiv.swap i p) j ≠ p := by
  intro h
  have h' := congrArg (Equiv.swap i p) h
  have h'' : j = i := by simpa using h'
  exact hij h''.symm

theorem transportPairPerm_apply_left {n : ℕ} (i j p q : Fin n)
    (hij : i ≠ j) (hpq : p ≠ q) :
    transportPairPerm i j p q i = p := by
  unfold transportPairPerm
  let r : Fin n := (Equiv.swap i p) j
  have hrp : r ≠ p := transportPairPerm_aux_ne i j p hij
  have hpq' : p ≠ q := hpq
  have hfix : Equiv.swap r q p = p := by
    simpa using (Equiv.swap_apply_of_ne_of_ne hrp.symm hpq')
  simp [r, hfix]

theorem transportPairPerm_apply_right {n : ℕ} (i j p q : Fin n)
    (hij : i ≠ j) (hpq : p ≠ q) :
    transportPairPerm i j p q j = q := by
  unfold transportPairPerm
  let r : Fin n := (Equiv.swap i p) j
  simp [r]

theorem pairCrossDifference_eq_of_symmetry {A : Type*} {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ}
    (hsym : SymmetricUtility (fun a s => u a s.state))
    (a : A) (i j p q : Fin n) (hij : i ≠ j) (hpq : p ≠ q) :
    pairCrossDifference u a p q = pairCrossDifference u a i j := by
  let σ := transportPairPerm i j p q
  have hσ := pairCrossDifference_perm (u := u) hsym σ a i j
  simpa [σ, transportPairPerm_apply_left i j p q hij hpq,
    transportPairPerm_apply_right i j p q hij hpq] using hσ

theorem pairState_cross_unary_zero {n : ℕ} (i j k : Fin n) (f : Fin 2 → ℤ) :
    f ((pairState i j 0 0) k) + f ((pairState i j 1 1) k) -
      f ((pairState i j 0 1) k) - f ((pairState i j 1 0) k) = 0 := by
  by_cases hki : k = i
  · subst hki
    simp [pairState]
  · by_cases hkj : k = j
    · subst hkj
      simp [pairState, hki]
    · simp [pairState, hki, hkj]

theorem pairState_cross_binary_of_lt {n : ℕ} {i j p q : Fin n}
    (hij : i < j) (hpq : p < q) (g : Fin 2 → Fin 2 → ℤ) :
    g ((pairState i j 0 0) p) ((pairState i j 0 0) q) +
      g ((pairState i j 1 1) p) ((pairState i j 1 1) q) -
      g ((pairState i j 0 1) p) ((pairState i j 0 1) q) -
      g ((pairState i j 1 0) p) ((pairState i j 1 0) q) =
      if p = i ∧ q = j then binaryCrossDifference g else 0 := by
  by_cases hpi : p = i
  · by_cases hqj : q = j
    · have hpq' : p = i ∧ q = j := ⟨hpi, hqj⟩
      have hqi : q ≠ i := by
        intro h
        exact hpq.ne (hpi.trans h.symm)
      have hji : j ≠ i := hij.ne.symm
      simp [pairState, binaryCrossDifference, hpi, hqj, hpq', hqi, hji]
    · have hqi : q ≠ i := by
        intro h
        exact hpq.ne (hpi.trans h.symm)
      simp [pairState, binaryCrossDifference, hpi, hqj, hqi]
  · by_cases hpj : p = j
    · have hqj : q ≠ j := by
        intro h
        exact hpq.ne (hpj.trans h.symm)
      have hqi : q ≠ i := by
        intro h
        have hlt : j < i := by simpa [hpj, h] using hpq
        exact (Nat.not_lt_of_ge (le_of_lt hij)) hlt
      simp [pairState, binaryCrossDifference, hpi, hqj, hqi]
    · by_cases hqi : q = i
      · have hpj' : p ≠ j := by
          intro h
          have hlt : j < i := by simpa [h, hqi] using hpq
          exact (Nat.not_lt_of_ge (le_of_lt hij)) hlt
        simp [pairState, binaryCrossDifference, hpi, hpj, hpj', hqi]
      · by_cases hqj : q = j
        · have hpq' : ¬ (p = i ∧ q = j) := by simp [hpi, hqj]
          simp [pairState, binaryCrossDifference, hpi, hpj, hqi]
        · have hpq' : ¬ (p = i ∧ q = j) := by simp [hpi, hqj]
          simp [pairState, binaryCrossDifference, hpi, hpj, hqi, hqj, hpq']

private lemma split_unary_binary_cross
    (U00 U11 U01 U10 B00 B11 B01 B10 : ℤ) :
    (U00 + B00) + (U11 + B11) - (U01 + B01) - (U10 + B10)
      = (((U00 + U11) - U01) - U10) + (((B00 + B11) - B01) - B10) := by
  ring

private lemma sum_four_split {α : Type*} (s : Finset α)
    (f g h k : α → ℤ) :
    s.sum (fun x => f x + g x - h x - k x) =
      ((s.sum f + s.sum g) - s.sum h) - s.sum k := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
      have ih' :
          s.sum (fun x => f x + g x + -h x + -k x) =
            s.sum f + s.sum g + (-s.sum h + -s.sum k) := by
        simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using ih
      simp [ha, sub_eq_add_neg, ih']
      abel_nf

theorem pairCrossDifference_eq_binaryCrossDifference_of_lt
    {A : Type*} {n : ℕ} {u : A → (Fin n → Fin 2) → ℤ}
    (pw : PairwiseUtility u) (a : A) {i j : Fin n} (hij : i < j) :
    pairCrossDifference u a i j =
      if pw.interacts i j then binaryCrossDifference (pw.binary i j a) else 0 := by
  classical
  let s00 : Fin n → Fin 2 := pairState i j 0 0
  let s11 : Fin n → Fin 2 := pairState i j 1 1
  let s01 : Fin n → Fin 2 := pairState i j 0 1
  let s10 : Fin n → Fin 2 := pairState i j 1 0
  let cond : Fin n → Fin n → Prop := fun p q => pw.interacts p q ∧ p < q
  have hunary :
      ∑ p : Fin n,
        (pw.unary p a (s00 p) + pw.unary p a (s11 p) -
          pw.unary p a (s01 p) - pw.unary p a (s10 p)) = 0 := by
    refine Finset.sum_eq_zero ?_
    intro p hp
    exact pairState_cross_unary_zero i j p (pw.unary p a)
  by_cases hInt : pw.interacts i j
  · have hbinary :
        ∑ p : Fin n,
          ∑ q : Fin n,
            ((if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
              (if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
              (if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
              (if cond p q then pw.binary p q a (s10 p) (s10 q) else 0)) =
          binaryCrossDifference (pw.binary i j a) := by
      calc
        ∑ p : Fin n,
            ∑ q : Fin n,
              ((if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                (if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                (if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                (if cond p q then pw.binary p q a (s10 p) (s10 q) else 0))
          = ∑ p : Fin n,
              ∑ q : Fin n,
                (if p = i ∧ q = j then binaryCrossDifference (pw.binary i j a) else 0) := by
              refine Finset.sum_congr rfl ?_
              intro p hp
              refine Finset.sum_congr rfl ?_
              intro q hq
              by_cases hcond : cond p q
              · have hpq : p < q := hcond.2
                rw [if_pos hcond, if_pos hcond, if_pos hcond, if_pos hcond]
                by_cases hpair : p = i ∧ q = j
                · simpa [hpair] using
                    (pairState_cross_binary_of_lt (i := i) (j := j) (p := p) (q := q) hij hpq (pw.binary p q a))
                · simpa [hpair] using
                    (pairState_cross_binary_of_lt (i := i) (j := j) (p := p) (q := q) hij hpq (pw.binary p q a))
              · have hneq : ¬ (p = i ∧ q = j) := by
                  intro hpq
                  rcases hpq with ⟨rfl, rfl⟩
                  exact hcond ⟨hInt, hij⟩
                simp [hcond, hneq]
        _ = binaryCrossDifference (pw.binary i j a) := by
              classical
              rw [Finset.sum_eq_single i]
              · rw [Finset.sum_eq_single j]
                · simp
                · intro q hq hqne
                  simp [hqne]
                · simp
              · intro p hp hpne
                simp [hpne]
              · simp
    let U00 : ℤ := ∑ p : Fin n, pw.unary p a (s00 p)
    let U11 : ℤ := ∑ p : Fin n, pw.unary p a (s11 p)
    let U01 : ℤ := ∑ p : Fin n, pw.unary p a (s01 p)
    let U10 : ℤ := ∑ p : Fin n, pw.unary p a (s10 p)
    let B00 : ℤ := ∑ p : Fin n, ∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0
    let B11 : ℤ := ∑ p : Fin n, ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0
    let B01 : ℤ := ∑ p : Fin n, ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0
    let B10 : ℤ := ∑ p : Fin n, ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0
    have hU : ((U00 + U11) - U01) - U10 = 0 := by
      calc
        ((U00 + U11) - U01) - U10
            = ∑ p : Fin n,
                (pw.unary p a (s00 p) + pw.unary p a (s11 p) -
                  pw.unary p a (s01 p) - pw.unary p a (s10 p)) := by
                symm
                simpa [U00, U11, U01, U10] using
                  (sum_four_split Finset.univ
                    (fun p => pw.unary p a (s00 p))
                    (fun p => pw.unary p a (s11 p))
                    (fun p => pw.unary p a (s01 p))
                    (fun p => pw.unary p a (s10 p)))
        _ = 0 := hunary
    have hB : ((B00 + B11) - B01) - B10 = binaryCrossDifference (pw.binary i j a) := by
      have hinner :
          ∑ p : Fin n,
            ((((∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                  ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                    ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0) =
              binaryCrossDifference (pw.binary i j a) := by
        calc
          ∑ p : Fin n,
              ((((∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                  ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                    ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                      ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0)
            = ∑ p : Fin n,
                ∑ q : Fin n,
                  ((if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                    (if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                    (if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                    (if cond p q then pw.binary p q a (s10 p) (s10 q) else 0)) := by
                refine Finset.sum_congr rfl ?_
                intro p hp
                symm
                simpa using
                  (sum_four_split Finset.univ
                    (fun q => if cond p q then pw.binary p q a (s00 p) (s00 q) else 0)
                    (fun q => if cond p q then pw.binary p q a (s11 p) (s11 q) else 0)
                    (fun q => if cond p q then pw.binary p q a (s01 p) (s01 q) else 0)
                    (fun q => if cond p q then pw.binary p q a (s10 p) (s10 q) else 0))
          _ = binaryCrossDifference (pw.binary i j a) := hbinary
      calc
        ((B00 + B11) - B01) - B10
            = ∑ p : Fin n,
                ((((∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                    ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                      ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                        ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0) := by
                symm
                simpa [B00, B11, B01, B10] using
                  (sum_four_split Finset.univ
                    (fun p => ∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0)
                    (fun p => ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0)
                    (fun p => ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0)
                    (fun p => ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0))
        _ = binaryCrossDifference (pw.binary i j a) := hinner
    have h00 : u a s00 = U00 + B00 := by
      simpa only [U00, B00, cond] using pw.decomp a s00
    have h11 : u a s11 = U11 + B11 := by
      simpa only [U11, B11, cond] using pw.decomp a s11
    have h01 : u a s01 = U01 + B01 := by
      simpa only [U01, B01, cond] using pw.decomp a s01
    have h10 : u a s10 = U10 + B10 := by
      simpa only [U10, B10, cond] using pw.decomp a s10
    rw [if_pos hInt]
    calc
      pairCrossDifference u a i j
          = (U00 + B00) + (U11 + B11) - (U01 + B01) - (U10 + B10) := by
              unfold pairCrossDifference
              rw [h00, h11, h01, h10]
      _ = (((U00 + U11) - U01) - U10) + (((B00 + B11) - B01) - B10) := by
            exact split_unary_binary_cross U00 U11 U01 U10 B00 B11 B01 B10
      _ = binaryCrossDifference (pw.binary i j a) := by
            rw [hU, hB]
            ring
  · have hbinary :
        ∑ p : Fin n,
          ∑ q : Fin n,
            ((if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
              (if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
              (if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
              (if cond p q then pw.binary p q a (s10 p) (s10 q) else 0)) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro p hp
      refine Finset.sum_eq_zero ?_
      intro q hq
      by_cases hcond : cond p q
      · have hpq : p < q := hcond.2
        have hneq : ¬ (p = i ∧ q = j) := by
          intro hpq'
          rcases hpq' with ⟨rfl, rfl⟩
          exact hInt hcond.1
        rw [if_pos hcond, if_pos hcond, if_pos hcond, if_pos hcond]
        simpa [hneq] using
          (pairState_cross_binary_of_lt (i := i) (j := j) (p := p) (q := q) hij hpq (pw.binary p q a))
      · simp [hcond]
    let U00 : ℤ := ∑ p : Fin n, pw.unary p a (s00 p)
    let U11 : ℤ := ∑ p : Fin n, pw.unary p a (s11 p)
    let U01 : ℤ := ∑ p : Fin n, pw.unary p a (s01 p)
    let U10 : ℤ := ∑ p : Fin n, pw.unary p a (s10 p)
    let B00 : ℤ := ∑ p : Fin n, ∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0
    let B11 : ℤ := ∑ p : Fin n, ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0
    let B01 : ℤ := ∑ p : Fin n, ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0
    let B10 : ℤ := ∑ p : Fin n, ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0
    have hU : ((U00 + U11) - U01) - U10 = 0 := by
      calc
        ((U00 + U11) - U01) - U10
            = ∑ p : Fin n,
                (pw.unary p a (s00 p) + pw.unary p a (s11 p) -
                  pw.unary p a (s01 p) - pw.unary p a (s10 p)) := by
                symm
                simpa [U00, U11, U01, U10] using
                  (sum_four_split Finset.univ
                    (fun p => pw.unary p a (s00 p))
                    (fun p => pw.unary p a (s11 p))
                    (fun p => pw.unary p a (s01 p))
                    (fun p => pw.unary p a (s10 p)))
        _ = 0 := hunary
    have hB : ((B00 + B11) - B01) - B10 = 0 := by
      have hinner :
          ∑ p : Fin n,
            ((((∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                  ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                    ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0) = 0 := by
        calc
          ∑ p : Fin n,
              ((((∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                  ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                    ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                      ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0)
            = ∑ p : Fin n,
                ∑ q : Fin n,
                  ((if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                    (if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                    (if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                    (if cond p q then pw.binary p q a (s10 p) (s10 q) else 0)) := by
                refine Finset.sum_congr rfl ?_
                intro p hp
                symm
                simpa using
                  (sum_four_split Finset.univ
                    (fun q => if cond p q then pw.binary p q a (s00 p) (s00 q) else 0)
                    (fun q => if cond p q then pw.binary p q a (s11 p) (s11 q) else 0)
                    (fun q => if cond p q then pw.binary p q a (s01 p) (s01 q) else 0)
                    (fun q => if cond p q then pw.binary p q a (s10 p) (s10 q) else 0))
          _ = 0 := hbinary
      calc
        ((B00 + B11) - B01) - B10
            = ∑ p : Fin n,
                ((((∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0) +
                    ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0) -
                      ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0) -
                        ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0) := by
                symm
                simpa [B00, B11, B01, B10] using
                  (sum_four_split Finset.univ
                    (fun p => ∑ q : Fin n, if cond p q then pw.binary p q a (s00 p) (s00 q) else 0)
                    (fun p => ∑ q : Fin n, if cond p q then pw.binary p q a (s11 p) (s11 q) else 0)
                    (fun p => ∑ q : Fin n, if cond p q then pw.binary p q a (s01 p) (s01 q) else 0)
                    (fun p => ∑ q : Fin n, if cond p q then pw.binary p q a (s10 p) (s10 q) else 0))
        _ = 0 := hinner
    have h00 : u a s00 = U00 + B00 := by
      simpa only [U00, B00, cond] using pw.decomp a s00
    have h11 : u a s11 = U11 + B11 := by
      simpa only [U11, B11, cond] using pw.decomp a s11
    have h01 : u a s01 = U01 + B01 := by
      simpa only [U01, B01, cond] using pw.decomp a s01
    have h10 : u a s10 = U10 + B10 := by
      simpa only [U10, B10, cond] using pw.decomp a s10
    rw [if_neg hInt]
    calc
      pairCrossDifference u a i j
          = (U00 + B00) + (U11 + B11) - (U01 + B01) - (U10 + B10) := by
              unfold pairCrossDifference
              rw [h00, h11, h01, h10]
      _ = (((U00 + U11) - U01) - U10) + (((B00 + B11) - B01) - B10) := by
            exact split_unary_binary_cross U00 U11 U01 U10 B00 B11 B01 B10
      _ = 0 := by
            rw [hU, hB]
            ring

def binaryLeftPart (g : Fin 2 → Fin 2 → ℤ) (x : Fin 2) : ℤ :=
  g x 0 - g 0 0

def binaryRightPart (g : Fin 2 → Fin 2 → ℤ) (y : Fin 2) : ℤ :=
  g 0 y

theorem binary_eq_left_right_of_zero_cross
    (g : Fin 2 → Fin 2 → ℤ) (hzero : binaryCrossDifference g = 0) (x y : Fin 2) :
    g x y = binaryLeftPart g x + binaryRightPart g y := by
  fin_cases x <;> fin_cases y <;>
    simp [binaryLeftPart, binaryRightPart, binaryCrossDifference] at hzero ⊢ <;>
    linarith

noncomputable def absorbedUnary {A : Type*} {n : ℕ} {u : A → (Fin n → Fin 2) → ℤ}
    (pw : PairwiseUtility u) (i : Fin n) (a : A) (x : Fin 2) : ℤ :=
  pw.unary i a x +
    (∑ j : Fin n,
      if pw.interacts i j ∧ i < j then
        binaryLeftPart (fun x y => pw.binary i j a x y) x
      else 0) +
    (∑ j : Fin n,
      if pw.interacts j i ∧ j < i then
        binaryRightPart (fun x y => pw.binary j i a x y) x
      else 0)

theorem pairwise_zero_crossDifference_unaryDecomposition
    {A : Type*} {n : ℕ} {u : A → (Fin n → Fin 2) → ℤ}
    (pw : PairwiseUtility u)
    (hzero : ∀ a : A, ∀ i j : Fin n, i < j → pairCrossDifference u a i j = 0) :
    UnaryCoordinateDecomposition u := by
  refine ⟨absorbedUnary pw, ?_⟩
  intro a s
  have hrewrite : ∀ i j : Fin n,
      (if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) else 0) =
        (if pw.interacts i j ∧ i < j then
          binaryLeftPart (fun x y => pw.binary i j a x y) (s i) else 0) +
        (if pw.interacts i j ∧ i < j then
          binaryRightPart (fun x y => pw.binary i j a x y) (s j) else 0) := by
    intro i j
    by_cases hcond : pw.interacts i j ∧ i < j
    · have hbd : binaryCrossDifference (fun x y => pw.binary i j a x y) = 0 := by
        have hpair := pairCrossDifference_eq_binaryCrossDifference_of_lt (pw := pw) (a := a) hcond.2
        rw [hzero a i j hcond.2, if_pos hcond.1] at hpair
        exact hpair.symm
      have hadd := binary_eq_left_right_of_zero_cross (fun x y => pw.binary i j a x y) hbd (s i) (s j)
      simp [hcond, hadd]
    · simp [hcond]
  rw [pw.decomp]
  simp_rw [hrewrite]
  let L : Fin n → Fin n → ℤ := fun i j =>
    if pw.interacts i j ∧ i < j then
      binaryLeftPart (fun x y => pw.binary i j a x y) (s i)
    else 0
  let R : Fin n → Fin n → ℤ := fun i j =>
    if pw.interacts i j ∧ i < j then
      binaryRightPart (fun x y => pw.binary i j a x y) (s j)
    else 0
  have hsplit :
      (∑ i : Fin n, ∑ j : Fin n, (L i j + R i j)) =
        (∑ i : Fin n, ∑ j : Fin n, L i j) + (∑ i : Fin n, ∑ j : Fin n, R i j) := by
    calc
      ∑ i : Fin n, ∑ j : Fin n, (L i j + R i j)
          = ∑ i : Fin n, ((∑ j : Fin n, L i j) + ∑ j : Fin n, R i j) := by
              refine Finset.sum_congr rfl ?_
              intro i hi
              symm
              simpa using
                (sum_four_split Finset.univ (fun j => L i j) (fun j => R i j) (fun _ => 0) (fun _ => 0)).symm
      _ = (∑ i : Fin n, ∑ j : Fin n, L i j) + (∑ i : Fin n, ∑ j : Fin n, R i j) := by
            rw [Finset.sum_add_distrib]
  have hswap :
      (∑ i : Fin n, ∑ j : Fin n, R i j) =
        ∑ i : Fin n,
          ∑ j : Fin n,
            if pw.interacts j i ∧ j < i then
              binaryRightPart (fun x y => pw.binary j i a x y) (s i)
            else 0 := by
    simpa [R] using (Finset.sum_comm :
      (∑ i : Fin n, ∑ j : Fin n, R i j) = ∑ j : Fin n, ∑ i : Fin n, R i j)
  rw [hsplit, hswap]
  have hpaircombine :
      ((∑ i : Fin n, ∑ j : Fin n, L i j) +
        ∑ i : Fin n,
          ∑ j : Fin n,
            if pw.interacts j i ∧ j < i then
              binaryRightPart (fun x y => pw.binary j i a x y) (s i)
            else 0) =
        ∑ i : Fin n,
          (∑ j : Fin n, L i j +
            ∑ j : Fin n,
              if pw.interacts j i ∧ j < i then
                binaryRightPart (fun x y => pw.binary j i a x y) (s i)
              else 0) := by
    rw [← Finset.sum_add_distrib]
  rw [hpaircombine]
  have hcombine :
      (∑ i : Fin n, pw.unary i a (s i)) +
        ∑ i : Fin n,
          (∑ j : Fin n, L i j +
            ∑ j : Fin n,
              if pw.interacts j i ∧ j < i then
                binaryRightPart (fun x y => pw.binary j i a x y) (s i)
              else 0) =
        ∑ i : Fin n,
          (pw.unary i a (s i) +
            (∑ j : Fin n, L i j +
              ∑ j : Fin n,
                if pw.interacts j i ∧ j < i then
                  binaryRightPart (fun x y => pw.binary j i a x y) (s i)
                else 0)) := by
    rw [← Finset.sum_add_distrib]
  rw [hcombine]
  apply Finset.sum_congr rfl
  intro i hi
  dsimp [absorbedUnary, L]
  ring

theorem binary_pairwise_symmetry_dichotomy
    {A : Type*} {n : ℕ} {u : A → (Fin n → Fin 2) → ℤ}
    (pw : PairwiseUtility u)
    (hsym : SymmetricUtility (fun a s => u a s.state)) :
    UnaryCoordinateDecomposition u ∨
      ∀ i j : Fin n, i ≠ j → HasBinaryPairInteraction u i j := by
  classical
  by_cases hzero : ∀ a : A, ∀ i j : Fin n, i < j → pairCrossDifference u a i j = 0
  · exact Or.inl (pairwise_zero_crossDifference_unaryDecomposition pw hzero)
  · push_neg at hzero
    rcases hzero with ⟨a, i, j, hij, hneq⟩
    refine Or.inr ?_
    intro p q hpq
    refine ⟨a, ?_⟩
    have heq := pairCrossDifference_eq_of_symmetry (u := u) hsym a i j p q hij.ne hpq
    rw [heq]
    exact hneq

end Paper4dFrontier
