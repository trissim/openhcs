/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Paper1 Authors

Neurosymbolic Linking: formalization of neural-symbolic identity systems.

A neurosymbolic system combines:
1. A neural encoder that produces structural/semantic representations
2. A symbolic handle that provides nominal identity
3. The linking that binds them together

This formalizes the pattern from GPH40: the pair (encoder, handle) 
achieves what neither can achieve alone - zero-error identity recovery.
-/

import AbstractClassSystem.Core
import AbstractClassSystem.Extended
import Paper1IT.GraphEntropy

namespace AbstractClassSystem.Neurosymbolic

open Ssot.GraphEntropy

/-! ## Basic Types -/

/-- Entity: the things being identified (types, objects, classes) -/
abbrev Entity := Typ

/-- Shape: structural/semantic representation (what neural encoder produces)
    For fintype purposes, we parameterize by the attribute name type -/
abbrev Shape (Attr : Type*) := Finset Attr

/-! ## NeurosymbolicLink Structure -/

/-- Neural encoder: maps entities to structural/semantic representation -/
abbrev NeuralEncoder (Attr : Type*) := Entity → Shape Attr

/-- Symbolic handle: maps entities to nominal identity -/
abbrev SymbolicHandle (α : Type*) := Entity → α

/-- A neurosymbolic link pairs a neural encoder with a symbolic handle.

    The key invariant: the handle must be injective for zero-error identity.
    This captures the architectural pattern of neurosymbolic systems:
    - Neural encoder compresses semantic structure
    - Symbolic handle carries identity-resolving information
    - Together they achieve zero-error identity recovery
-/
structure NeurosymbolicLink (Attr : Type*) (SymbolType : Type*) [DecidableEq SymbolType] where
  /-- Neural encoder: entity → Shape (structural/semantic) -/
  encode : NeuralEncoder Attr
  /-- Symbolic handle: entity → Symbol (nominal identity) -/
  handle : SymbolicHandle SymbolType
  /-- The handle must be injective for zero-error identity -/
  handle_injective : Function.Injective handle

/-- Convenience: neurosymbolic link with string symbols -/
abbrev NeurosymbolicSystem (Attr : Type*) := NeurosymbolicLink Attr String

/-! ## Combined Observation -/

/-- The combined observation from a neurosymbolic link -/
def NeurosymbolicLink.observe 
    (Attr : Type*)
    {α : Type*} [DecidableEq α] 
    (ns : NeurosymbolicLink Attr α) : 
    Entity → Shape Attr × α :=
  fun e => (ns.encode e, ns.handle e)

/-! ## Core Theorems -/

/-! ## NSL1: Neurosymbolic Linking Achieves Zero-Error Identity

    If the symbolic handle is injective, the neurosymbolic link achieves
    zero-error identity recovery (maxFiberCard ≤ 1).
    
    This follows from GPH25/GPH26/GPH27 in GraphEntropy: an injective
    observation has maxFiberCard ≤ 1.
-/

theorem neurosymbolic_zero_error_identity 
    {Attr : Type*} [Fintype Attr] [DecidableEq Attr]
    {α : Type*} [DecidableEq α] [Fintype α] 
    [Fintype Entity] [Nonempty Entity]
    (ns : NeurosymbolicLink Attr α) :
    maxFiberCard ns.observe ≤ 1 := by
  have h : Function.Injective ns.observe := by
    intro e1 e2 hpair
    have := congrArg Prod.snd hpair
    exact ns.handle_injective this
  exact maxFiberCard_le_one_of_injective _ h

/-! ## NSL2: Shape-Alone Does NOT Achieve Zero-Error Identity

    If there exist entities with same shape but different identity,
    then shape-only observation has maxFiberCard > 1.
    
    This is the compression-theoretic reading: semantic representation
    alone cannot distinguish entities when collisions exist.
-/

theorem shape_alone_not_zero_error
    {Attr : Type*} [Fintype Attr] [DecidableEq Attr]
    (encode : NeuralEncoder Attr) 
    [Fintype Entity] [DecidableEq Entity]
    (hne : ∃ e1 e2 : Entity, e1 ≠ e2 ∧ encode e1 = encode e2) :
    1 < maxFiberCard encode := by
  obtain ⟨e1, e2, hne, henc⟩ := hne
  classical
  have h1 : e1 ∈ (Finset.univ : Finset Entity).filter (fun e' => encode e' = encode e1) := by
    simp
  have h2 : e2 ∈ (Finset.univ : Finset Entity).filter (fun e' => encode e' = encode e1) := by
    simp [henc]
  have hcard_ge_2 : 2 ≤ (Finset.univ.filter (fun e' => encode e' = encode e1)).card := by
    have : 1 < (Finset.univ.filter (fun e' => encode e' = encode e1)).card := by
      rw [Finset.one_lt_card]
      exact ⟨e1, h1, e2, h2, hne⟩
    omega
  unfold maxFiberCard
  calc 1 < 2 := by omega
       _ ≤ (Finset.univ.filter (fun e' => encode e' = encode e1)).card := hcard_ge_2
       _ ≤ Finset.sup Finset.univ (fun b => (Finset.univ.filter (fun e' => encode e' = b)).card) := 
           Finset.le_sup (s := Finset.univ) (f := fun b => (Finset.univ.filter (fun e' => encode e' = b)).card) (by simp)

/-! ## NSL3: Neurosymbolic Linking is Necessary for Zero-Error

    The synthesis direction: If we require zero-error identity (D=0) and
    the neural encoder is not injective, then a symbolic handle is necessary.
    
    This is the "no free lunch" theorem for semantic identity compression:
    if the encoder collapses entities, no decoder can recover them without
    additional identity-resolving information.
-/

theorem neurosymbolic_necessary_for_zero_error
    {Attr : Type*} [Fintype Attr]
    (encode : NeuralEncoder Attr)
    (hne : ∃ e1 e2 : Entity, e1 ≠ e2 ∧ encode e1 = encode e2) :
    ¬ ∃ decode : Shape Attr → Entity, Function.LeftInverse decode encode := by
  intro ⟨decode, hinv⟩
  have hinj : Function.Injective encode := Function.LeftInverse.injective hinv
  obtain ⟨e1, e2, hne, henc⟩ := hne
  exact hne (hinj henc)

/-! ## NSL4: GPH40 Characterizes Neurosymbolic Identity Recovery

    GPH40 (taskRecoverable_pair_iff) states that a pair (observe, side)
    recovers a task iff they jointly distinguish all task values.
    
    Specialized to identity recovery (task = id), this says:
    the neurosymbolic link recovers identity iff it distinguishes all entities.
-/

theorem neurosymbolic_identity_recovery 
    {Attr : Type*} [Fintype Attr] [DecidableEq Attr]
    {α : Type*} [DecidableEq α] [Fintype α] 
    [Fintype Entity] [Nonempty Entity] [DecidableEq Entity]
    (ns : NeurosymbolicLink Attr α) :
    TaskRecoverable ns.observe (fun e => e) ↔
    ∀ e1 e2 : Entity, ns.encode e1 = ns.encode e2 → ns.handle e1 = ns.handle e2 → e1 = e2 := by
  rw [taskRecoverable_pair_iff]
  rfl

/-! ## NSL5: Injective Handle Implies Identity Recoverable

    Corollary of NSL4: if the handle is injective (by construction),
    then the neurosymbolic link recovers identity.
-/

theorem neurosymbolic_injective_implies_recoverable 
    {Attr : Type*} [Fintype Attr] [DecidableEq Attr]
    {α : Type*} [DecidableEq α] [Fintype α] 
    [Fintype Entity] [Nonempty Entity] [DecidableEq Entity]
    (ns : NeurosymbolicLink Attr α) :
    TaskRecoverable ns.observe (fun e => e) := by
  rw [neurosymbolic_identity_recovery]
  intro e1 e2 _ heq
  exact ns.handle_injective heq

/-! ## NSL6: Neurosymbolic Linking is the Canonical Helper-View

    GPH40 establishes that (observe, side) pairs are the canonical form
    of "helper-view" that restores identity.
    
    This theorem says: for any encoder and injective handle,
    the pair achieves maxFiberCard ≤ 1 (zero-error identity).
-/

theorem neurosymbolic_is_canonical_helper_view
    {Attr : Type*} [Fintype Attr] [DecidableEq Attr]
    {α : Type*} [DecidableEq α] [Fintype α] 
    [Fintype Entity]
    (encode : NeuralEncoder Attr)
    (handle : SymbolicHandle α)
    (hinj : Function.Injective handle) :
    maxFiberCard (fun e => (encode e, handle e)) ≤ 1 := by
  apply maxFiberCard_le_one_of_injective
  intro e1 e2 hpair
  exact hinj (congrArg Prod.snd hpair)

end AbstractClassSystem.Neurosymbolic
