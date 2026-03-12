/-
  SSOT Formalization - Derivation Relation
  Paper 2: Formal Foundations for the Single Source of Truth Principle

  CROSS-REFERENCE: The formal derivation theory is established in Paper 1
  (docs/papers/paper1_typing_discipline/proofs/axis_framework.lean):

  - `derivable (a b : Axis)` - axis a is derivable from axis b
  - `derivable_refl` - reflexivity of derivation
  - `derivable_trans` - transitivity of derivation
  - `axisIndependent A` - no axis in A is derivable from another
  - `semantically_minimal_implies_independent` - minimal sets are independent

  Paper 2 takes these results as established and proves NEW theorems about
  coherence and SSOT (see Coherence.lean). This file provides the bridge
  definitions connecting Paper 1's axis derivation to Paper 2's encoding model.
-/

namespace Ssot

-- Definition 2.3: Derivation (encoding-level formulation)
-- L_derived is derived from L_source iff updating L_source automatically updates L_derived
-- This mirrors Paper 1's `derivable` at the encoding-location level
structure DerivationSystem (Location : Type) where
  derived_from : Location → Location → Prop
  -- Transitivity: if A derives B and B derives C, then A derives C
  -- (Proven for axes in Paper 1: derivable_trans)
  transitive : ∀ a b c, derived_from a b → derived_from b c → derived_from a c
  -- Irreflexivity: nothing is derived from itself
  irrefl : ∀ a, ¬derived_from a a

-- Theorem 2.4: Derivation Excludes from DOF
-- ESTABLISHED IN PAPER 1: axisIndependent A ↔ no axis derivable from another
-- The contrapositive: if derived, then NOT independent (not counted in DOF)
--
-- Paper 1 proof: semantically_minimal_implies_independent shows that
-- minimal (= no redundancy) implies independent. Derivability creates
-- redundancy, therefore derived locations are not independent.
--
-- We state this as a definitional bridge: derived locations don't add to DOF.
def excludes_from_dof {Location : Type} (D : DerivationSystem Location)
    (source derived : Location) : Prop :=
  D.derived_from source derived

-- Corollary 2.5: Metaprogramming Achieves SSOT
-- If all encodings except one are derived from that one, DOF = 1
--
-- PAPER 1 FOUNDATION: collapseDerivable reduces an axis set to its
-- independent core. If all axes derive from one, collapse yields size 1.
--
-- This is the key insight connecting Papers 1 and 2:
-- - Paper 1: derivation reduces axis count
-- - Paper 2: reduced DOF means coherence is guaranteed (Coherence.lean)
def all_derived_from_source {Location : Type} (D : DerivationSystem Location)
    (source : Location) (all_locations : List Location) : Prop :=
  ∀ loc ∈ all_locations, loc = source ∨ D.derived_from source loc

-- Key insight: Derivation is the mechanism that reduces DOF
-- Definition-time metaprogramming creates derivation relationships
-- Runtime computation is too late (structure already fixed)
--
-- The formal proof chain:
-- 1. Paper 1: derivable axes are redundant (redundantWith)
-- 2. Paper 1: redundant axes collapse (collapseDerivable)
-- 3. Paper 1: collapsed sets are independent (axisIndependent)
-- 4. Paper 2: DOF = size of independent set
-- 5. Paper 2: DOF = 1 → coherence guaranteed (dof_one_implies_coherent)

end Ssot
