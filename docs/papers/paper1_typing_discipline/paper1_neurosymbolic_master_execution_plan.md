# Paper 1 Neurosymbolic Compression: Master Execution Plan

## Overview

This plan integrates:
1. **Paper text changes** (exposing existing Lean handles)
2. **New Lean mechanization** (NeurosymbolicLink structure + theorems)
3. **Paper-LaTeX integration** (citing new handles)
4. **Verification** (build, test, consistency)

The goal: ground the neurosymbolic compression story in formal Lean mechanization so that claims about open-world D=0, lossless/lossy identity compression, and nominal layer necessity are machine-verified.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PAPER FRAMING                                │
│  "Neurosymbolic linking is the architectural act of paying the      │
│   identity bit cost with symbolic handles."                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Arc 1: Exact Coding Laws                                           │
│  ├── Fixed-length zero-error: L >= log2(A_π)                        │
│  ├── Adaptive budgets: fiber-by-fiber                               │
│  └── Distortion floor: D >= 1 - 2^L/a                               │
│                                                                      │
│  Arc 2: Representation Sufficiency                                   │
│  ├── Task sufficiency: A_{π→Y} <= 1                                 │
│  ├── Helper views: (π,σ) restores identity                          │
│  └── Factorized structure: multiplicative                           │
│                                                                      │
│  Arc 3: Neurosymbolic Linking (NEW)                                  │
│  ├── Neural encoder → Shape (structural)                            │
│  ├── Symbolic handle → Symbol (nominal)                             │
│  ├── Linking → D=0 identity recovery                                │
│  └── Open-world: persistent D=0 requires nominal layer              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LEAN MECHANIZATION                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Existing (Expose):                                                  │
│  ├── GPH27-43: compression, sufficiency, adaptive                   │
│  ├── ROB1-2, UND1-2: open-world, undecidability                     │
│  ├── AXI1, IMP1, EMB1: shape cannot distinguish                     │
│  ├── PRIV1-3: nominal vs attribute-only separation                  │
│  └── NOM1-2: nominal typing properties                              │
│                                                                      │
│  New (Mechanize):                                                    │
│  ├── NeurosymbolicLink structure                                    │
│  ├── neurosymbolic_zero_error_identity                              │
│  ├── shape_alone_not_zero_error                                     │
│  ├── neurosymbolic_necessary_for_zero_error                         │
│  ├── neurosymbolic_identity_recovery (GPH40 connection)             │
│  └── neurosymbolic_is_canonical_helper_view                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Preparation

### 0.1 Verify Current Build State

**Commands:**
```bash
cd /home/ts/code/projects/papers-archive/docs/papers/paper1_typing_discipline/proofs
lake build
# Expect: Build success, 0 sorries

cd /home/ts/code/projects/papers-archive
python3 scripts/build_papers.py latex paper1_jsait
# Expect: PDF builds successfully
```

**Deliverable:** Clean build of both Lean and LaTeX

### 0.2 Create Working Branch

**Command:**
```bash
cd /home/ts/code/projects/papers-archive
git checkout -b neurosymbolic-linking-mechanization
```

**Deliverable:** Clean branch for all changes

---

## Phase 1: Expose Existing Handles (Paper Text Only)

**Goal:** Surface the 50+ handles that exist in Lean but are not cited in the paper.

**Effort:** 2-3 hours
**Risk:** LOW (no Lean changes)

### 1.1 Add Forcing Chain to Section 2 or 6

**File:** `content/02_compression_framework.tex` or `content/06_applications.tex`

**Current state:** No mention of AXI1, IMP1, EMB1

**Target state:** Add paragraph:

```latex
\paragraph{Why structural representation alone cannot distinguish.}
The mechanization proves that structural typing cannot distinguish 
entities with identical structural profiles (\LH{AXI1}), cannot track 
provenance or source identity (\LH{IMP1}), and cannot embed nominal 
identity information into the structural layer (\LH{EMB1}). These 
impossibility theorems establish that the information barrier is not 
a limitation of specific encodings but a fundamental property of 
structural observation.
```

**Verification:** Build PDF, check that handles render

### 1.2 Add Disclosure Separation to Section 6 or 7

**File:** `content/06_applications.tex` or `content/04a_complexity_bounds.tex`

**Current state:** Only PRIV3 is cited

**Target state:** Add paragraph:

```latex
\paragraph{Nominal versus attribute-only identity resolution.}
The mechanization establishes a formal separation between nominal 
and attribute-only identity resolution. Nominal access requires 
constant disclosure budget (\LH{PRIV1}), while tag-free attribute-only 
resolution can require up to $n-1$ primitive disclosures in the 
adversarial family (\LH{PRIV2}). The separation theorem (\LH{PRIV3}) 
shows that this gap is fundamental: identity-carrying nominal tags 
resolve collisions in one step, while attribute-only observers must 
pay an interaction cost proportional to the distinguishability 
structure of the class space.
```

**Verification:** Build PDF, check that handles render

### 1.3 Add Nominal Typing Properties to Section 6

**File:** `content/06_applications.tex`

**Current state:** No mention of NOM1, NOM2

**Target state:** Add to "Nominal, Structural, and Neuro-Symbolic Readings":

```latex
The nominal typing theorems formalize this distinction: nominal 
typing is centralized and provides constant-time localization 
(\LH{NOM1}, \LH{NOM2}), while structural typing must traverse the 
entire namespace hierarchy. This is the PL-theoretic counterpart to 
the information-theoretic coding theorems: nominal identity is 
architecturally efficient because it carries identity-resolving 
information directly.
```

### 1.4 Update Abstract

**File:** `content/abstract.tex`

**Current final sentence:**
> Beyond the coding laws themselves, the same ambiguity-fiber geometry yields exact task-sufficiency, helper-view, and factorized-representation consequences for learned and neuro-symbolic systems.

**Target state:** Add after:

```latex
The distortion floor quantifies the error that remains when 
identity-resolving bits are withheld. The mechanization proves 
that structural observation alone cannot distinguish entities 
(\LH{AXI1}, \LH{IMP1}) while nominal identity resolution succeeds 
in constant disclosure (\LH{PRIV1}). In open worlds, persistent 
certified zero-error recovery is forced off the semantic layer 
and onto a nominal identity layer.
```

### 1.5 Update Introduction Contributions

**File:** `content/01_introduction.tex`

**Add new bullet after contribution 4:**

```latex
\item \textbf{Lossless and lossy identity compression.} The paper's 
framework makes explicit a fundamental tradeoff: exact identity 
recovery from a noninjective representation is either lossless 
(pay the exact nominal bit cost $\lceil \log_2 |\pi^{-1}(u)| \rceil$ 
per fiber) or lossy (accept the distortion floor $D \ge 1 - 2^L/a$). 
The mechanization proves that structural observation cannot 
distinguish entities (\LH{AXI1}), cannot embed nominal information 
(\LH{EMB1}), and that nominal identity resolution requires constant 
disclosure (\LH{PRIV1}) while attribute-only resolution may require 
$n-1$ steps (\LH{PRIV2}).
```

### 1.6 Update Conclusion

**File:** `content/07_conclusion.tex`

**Add new paragraph after line 7:**

```latex
The paper also makes explicit the lossless/lossy identity compression 
tradeoff. A system either pays the exact nominal bit cost for 
zero-error identity recovery, or accepts the distortion floor that 
remains when those bits are withheld. Under open-world growth, 
persistent certified $D=0$ is forced off the semantic representation 
layer and onto a nominal identity layer. The mechanization proves 
that structural typing cannot distinguish (\LH{AXI1}), cannot embed 
nominal information (\LH{EMB1}), and that the disclosure gap between 
nominal and attribute-only identity resolution is fundamental 
(\LH{PRIV3}).
```

### Phase 1 Deliverables

- [ ] 02_compression_framework.tex or 06_applications.tex updated with AXI1, IMP1, EMB1
- [ ] 06_applications.tex or 04a_complexity_bounds.tex updated with PRIV1, PRIV2
- [ ] 06_applications.tex updated with NOM1, NOM2
- [ ] abstract.tex updated
- [ ] 01_introduction.tex updated
- [ ] 07_conclusion.tex updated
- [ ] PDF builds successfully
- [ ] All new handles render correctly

---

## Phase 2: Lean Mechanization - NeurosymbolicLink Structure

**Goal:** Define the NeurosymbolicLink structure and basic theorems.

**Effort:** 4-6 hours
**Risk:** MEDIUM (new Lean code, but straightforward)

### 2.1 Create New File

**File:** `proofs/AbstractClassSystem/Neurosymbolic.lean`

**Structure:**
```lean
/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: [Author Name]

Neurosymbolic Linking: formalization of neural-symbolic identity systems.

A neurosymbolic system combines:
1. A neural encoder that produces structural/semantic representations (Shape)
2. A symbolic handle that provides nominal identity (Symbol)
3. The linking that binds them together

This formalizes the pattern from GPH40: the pair (encoder, handle) 
achieves what neither can achieve alone.
-/

import AbstractClassSystem.Core
import AbstractClassSystem.Extended
import Paper1IT.GraphEntropy

namespace AbstractClassSystem.Neurosymbolic

open Ssot.GraphEntropy

/-! ## Basic Types -/

/-- Entity: the things being identified (types, objects, classes) -/
abbrev Entity := Typ

/-- Shape: structural/semantic representation (what neural encoder produces) -/
abbrev Shape := Finset AttrName

/-- Symbol: nominal identity (what symbolic handle provides) -/
abbrev Symbol := String

/-! ## NeurosymbolicLink Structure -/

/-- Neural encoder: maps entities to structural/semantic representation -/
def NeuralEncoder := Entity → Shape

/-- Symbolic handle: maps entities to nominal identity -/
def SymbolicHandle (α : Type*) := Entity → α

/-- A neurosymbolic link pairs a neural encoder with a symbolic handle.

    The key invariant: the handle must be injective for zero-error identity.
    This captures the architectural pattern of neurosymbolic systems:
    - Neural encoder compresses semantic structure
    - Symbolic handle carries identity-resolving information
    - Together they achieve zero-error identity recovery
-/
structure NeurosymbolicLink (SymbolType : Type*) [DecidableEq SymbolType] where
  /-- Neural encoder: entity → Shape (structural/semantic) -/
  encode : NeuralEncoder
  /-- Symbolic handle: entity → Symbol (nominal identity) -/
  handle : SymbolicHandle SymbolType
  /-- The handle must be injective for zero-error identity -/
  handle_injective : Function.Injective handle

/-- Convenience: neurosymbolic link with string symbols -/
abbrev NeurosymbolicSystem := NeurosymbolicLink String

/-! ## Combined Observation -/

/-- The combined observation from a neurosymbolic link -/
def NeurosymbolicLink.observe (ns : NeurosymbolicLink α _) : Entity → Shape × α :=
  fun e => (ns.encode e, ns.handle e)

end AbstractClassSystem.Neurosymbolic
```

### 2.2 Verify Structure Compiles

**Command:**
```bash
cd /home/ts/code/projects/papers-archive/docs/papers/paper1_typing_discipline/proofs
lake build AbstractClassSystem.Neurosymbolic
```

**Deliverable:** Structure compiles with no errors

### Phase 2 Deliverables

- [ ] New file created: `AbstractClassSystem/Neurosymbolic.lean`
- [ ] NeurosymbolicLink structure defined
- [ ] Basic types (Entity, Shape, Symbol) defined
- [ ] File compiles successfully
- [ ] No sorries

---

## Phase 3: Lean Mechanization - Core Theorems

**Goal:** Prove the core neurosymbolic linking theorems.

**Effort:** 4-6 hours
**Risk:** MEDIUM (proof work)

### 3.1 Theorem 1: Neurosymbolic Linking Achieves Zero-Error Identity

**File:** `proofs/AbstractClassSystem/Neurosymbolic.lean`

**Theorem:**
```lean
/-! ## Theorem 1: Neurosymbolic Linking Achieves Zero-Error Identity

    If the symbolic handle is injective, the neurosymbolic link achieves
    zero-error identity recovery (maxFiberCard = 1).
    
    This follows from GPH25/GPH26/GPH27 in GraphEntropy: an injective
    observation has maxFiberCard ≤ 1.
-/

theorem neurosymbolic_zero_error_identity 
    {α : Type*} [DecidableEq α] [Nonempty Entity]
    (ns : NeurosymbolicLink α _) :
    maxFiberCard ns.observe ≤ 1 := by
  -- The handle is injective by construction
  have h : Function.Injective ns.observe := by
    intro e1 e2 hpair
    have := congrArg Prod.snd hpair
    exact ns.handle_injective this
  -- Injective observation has maxFiberCard ≤ 1
  exact maxFiberCard_le_one_of_injective _ h
```

**Tactics:**
- Use `maxFiberCard_le_one_of_injective` from GraphEntropy.lean
- Show that the pair (encode, handle) is injective when handle is injective
- This is a 5-line proof

### 3.2 Theorem 2: Shape-Alone Does NOT Achieve Zero-Error

**File:** `proofs/AbstractClassSystem/Neurosymbolic.lean`

**Theorem:**
```lean
/-! ## Theorem 2: Shape-Alone Does NOT Achieve Zero-Error Identity

    This specializes AXI1/IMP1 to the neurosymbolic setting: shape-based
    typing cannot distinguish entities with identical shape.
    
    Formal statement: If there exist entities with same shape but different
    identity, then maxFiberCard(encode) > 1.
-/

theorem shape_alone_not_zero_error
    (encode : NeuralEncoder) 
    (hne : ∃ e1 e2 : Entity, e1 ≠ e2 ∧ encode e1 = encode e2) :
    1 < maxFiberCard encode := by
  obtain ⟨e1, e2, hne, henc⟩ := hne
  -- e1 and e2 are in the same fiber but distinct
  -- Therefore the fiber has at least 2 elements
  have h_fiber : ∀ e, e ∈ (Finset.univ : Finset Entity).filter (fun e' => encode e' = encode e) := by
    intro e
    simp [Finset.mem_filter]
    exact ⟨Finset.mem_univ e, rfl⟩
  -- Use the definition of maxFiberCard
  simp only [maxFiberCard]
  -- Show the maximum fiber size is at least 2
  have h_card : 2 ≤ (Finset.univ.filter (fun e' => encode e' = encode e1)).card := by
    -- e1 and e2 are both in this fiber and are distinct
    have h1 : e1 ∈ Finset.univ.filter (fun e' => encode e' = encode e1) := by
      simp [Finset.mem_filter]
    have h2 : e2 ∈ Finset.univ.filter (fun e' => encode e' = encode e1) := by
      simp [Finset.mem_filter, henc]
    -- Need to show these are two distinct elements
    sorry  -- TODO: complete this proof
  -- Then maxFiberCard >= 2 > 1
  sorry  -- TODO: complete this proof
```

**Tactics:**
- Use the existence of collision (hne) to show fiber has ≥ 2 elements
- Apply maxFiberCard definition
- May need helper lemmas about Finset.card

### 3.3 Theorem 3: Neurosymbolic Linking is NECESSARY

**File:** `proofs/AbstractClassSystem/Neurosymbolic.lean`

**Theorem:**
```lean
/-! ## Theorem 3: Neurosymbolic Linking is Necessary for Zero-Error

    The synthesis direction: If we require zero-error identity (D=0) and
    the neural encoder is not injective, then a symbolic handle is necessary.
    
    This is the "no free lunch" theorem for semantic identity compression.
-/

theorem neurosymbolic_necessary_for_zero_error
    (encode : NeuralEncoder)
    (hne : ∃ e1 e2 : Entity, e1 ≠ e2 ∧ encode e1 = encode e2) :
    ¬ ∃ decode : Shape → Entity, Function.LeftInverse decode encode := by
  intro ⟨decode, hinv⟩
  -- If decode is left inverse, then encode is injective
  have hinj : Function.Injective encode := Function.LeftInverse.injective hinv
  -- But we have two distinct entities with same encoding
  obtain ⟨e1, e2, hne, henc⟩ := hne
  exact hne (hinj henc)
```

**Tactics:**
- Use `Function.LeftInverse.injective`
- Contradiction with non-injectivity assumption
- This is a 5-line proof

### 3.4 Theorem 4: GPH40 Connection

**File:** `proofs/AbstractClassSystem/Neurosymbolic.lean`

**Theorem:**
```lean
/-! ## Theorem 4: GPH40 Characterizes Neurosymbolic Linking

    GPH40 (taskRecoverable_pair_iff) states that a pair (observe, side)
    recovers a task iff they jointly distinguish all task values.
    
    Specialized to identity recovery (task = id), this says:
    the neurosymbolic link recovers identity iff it distinguishes all entities.
-/

theorem neurosymbolic_identity_recovery 
    {α : Type*} [DecidableEq α] [Nonempty Entity]
    (ns : NeurosymbolicLink α _) :
    TaskRecoverable ns.observe (fun e => e) ↔
    ∀ e1 e2, ns.encode e1 = ns.encode e2 → ns.handle e1 = ns.handle e2 → e1 = e2 := by
  -- This is exactly GPH40 with task = id
  convert taskRecoverable_pair_iff (observe := ns.encode) (side := ns.handle) (task := fun e => e)
  -- Need to show ns.observe = fun e => (ns.encode e, ns.handle e)
  rfl
```

**Tactics:**
- Direct application of `taskRecoverable_pair_iff` (GPH40)
- This is a 2-line proof

### 3.5 Theorem 5: Injective Handle Implies Recoverable

**File:** `proofs/AbstractClassSystem/Neurosymbolic.lean`

**Theorem:**
```lean
/-! ## Corollary: With Injective Handle, Identity is Recoverable -/

theorem neurosymbolic_injective_implies_recoverable 
    {α : Type*} [DecidableEq α] [Nonempty Entity]
    (ns : NeurosymbolicLink α _) :
    TaskRecoverable ns.observe (fun e => e) := by
  rw [neurosymbolic_identity_recovery]
  intro e1 e2 _ heq
  exact ns.handle_injective heq
```

**Tactics:**
- Apply Theorem 4
- Use injectivity of handle
- This is a 4-line proof

### 3.6 Theorem 6: Canonical Helper-View

**File:** `proofs/AbstractClassSystem/Neurosymbolic.lean`

**Theorem:**
```lean
/-! ## Theorem 6: Neurosymbolic Linking is the Canonical Helper-View

    GPH40 establishes that (observe, side) pairs are the canonical form
    of "helper-view" that restores identity.
    
    This theorem says: for any encoder with collisions, adding an
    injective handle achieves D=0.
-/

theorem neurosymbolic_is_canonical_helper_view
    {α : Type*} [DecidableEq α]
    (encode : NeuralEncoder)
    (handle : SymbolicHandle α)
    (hinj : Function.Injective handle) :
    maxFiberCard (fun e => (encode e, handle e)) ≤ 1 := by
  apply maxFiberCard_le_one_of_injective
  intro e1 e2 hpair
  exact hinj (congrArg Prod.snd hpair)
```

**Tactics:**
- Direct application of injectivity + maxFiberCard_le_one_of_injective
- This is a 4-line proof

### Phase 3 Deliverables

- [ ] Theorem 1 (neurosymbolic_zero_error_identity) proved
- [ ] Theorem 2 (shape_alone_not_zero_error) proved
- [ ] Theorem 3 (neurosymbolic_necessary_for_zero_error) proved
- [ ] Theorem 4 (neurosymbolic_identity_recovery) proved
- [ ] Theorem 5 (neurosymbolic_injective_implies_recoverable) proved
- [ ] Theorem 6 (neurosymbolic_is_canonical_helper_view) proved
- [ ] File compiles with no sorries
- [ ] All proofs complete

---

## Phase 4: Register Handles

**Goal:** Create stable handles for the new theorems.

**Effort:** 1 hour
**Risk:** LOW

### 4.1 Add Handles to HandleAliases.lean

**File:** `proofs/HandleAliases.lean`

**Add:**
```lean
/-! ## Neurosymbolic Linking Handles (NSL namespace) -/

/-- NSL1: Neurosymbolic linking achieves zero-error identity -/
abbrev NSL1 := @neurosymbolic_zero_error_identity

/-- NSL2: Shape-alone does not achieve zero-error identity -/
abbrev NSL2 := @shape_alone_not_zero_error

/-- NSL3: Neurosymbolic linking is necessary for zero-error -/
abbrev NSL3 := @neurosymbolic_necessary_for_zero_error

/-- NSL4: GPH40 connection - identity recovery characterization -/
abbrev NSL4 := @neurosymbolic_identity_recovery

/-- NSL5: Injective handle implies identity recoverable -/
abbrev NSL5 := @neurosymbolic_injective_implies_recoverable

/-- NSL6: Neurosymbolic linking is canonical helper-view -/
abbrev NSL6 := @neurosymbolic_is_canonical_helper_view
```

### 4.2 Rebuild Handle Index

**Command:**
```bash
cd /home/ts/code/projects/papers-archive
python3 scripts/build_papers.py latex paper1_jsait
# This regenerates lean_handle_ids_auto.tex
```

**Deliverable:** New handles appear in `content/lean_handle_ids_auto.tex`

### Phase 4 Deliverables

- [ ] NSL1-6 handles registered in HandleAliases.lean
- [ ] Handle index regenerated
- [ ] Handles appear in lean_handle_ids_auto.tex

---

## Phase 5: Integrate with Paper Text

**Goal:** Cite the new NSL handles in the paper.

**Effort:** 2 hours
**Risk:** LOW

### 5.1 Add to Section 6 (Applications)

**File:** `content/06_applications.tex`

**Add new subsection:**
```latex
\subsection{Neurosymbolic Linking as the Nominal Layer}

The theorems above establish that exact identity recovery from a 
noninjective representation requires additional identity-carrying 
information. This section formalizes neurosymbolic linking as the 
architectural realization of that requirement.

\begin{definition}[Neurosymbolic link]
A \emph{neurosymbolic link} pairs a neural encoder $\pi : \mathcal{E} \to \mathcal{S}$ 
with a symbolic handle $\sigma : \mathcal{E} \to \mathcal{A}$ where $\sigma$ 
is injective. The encoder produces structural/semantic representations; 
the handle provides nominal identity.
\end{definition}

\begin{theorem}[Neurosymbolic linking achieves zero-error identity]\label{thm:neurosymbolic-zero-error}
If the symbolic handle is injective, the neurosymbolic link achieves 
zero-error identity recovery: the combined observation $(\pi(e), \sigma(e))$ 
has maxFiberCard $\le 1$.
\end{theorem}
\leanmeta{\LH{NSL1}}

\begin{proof}
The pair projection on the symbolic component is injective when the 
handle is injective. By GPH27, an injective observation has maxFiberCard $\le 1$.
\end{proof}

\begin{theorem}[Shape-alone does not achieve zero-error identity]\label{thm:shape-alone-fails}
If the neural encoder has collisions---there exist distinct entities 
$e_1 \neq e_2$ with $\pi(e_1) = \pi(e_2)$---then no shape-only decoder 
can achieve zero-error identity.
\end{theorem}
\leanmeta{\LH{NSL2}}

\begin{proof}
This is the specialization of AXI1/IMP1 to the neurosymbolic setting: 
structural observation cannot distinguish entities with identical 
structural profiles.
\end{proof}

\begin{theorem}[Neurosymbolic linking is necessary for zero-error]\label{thm:neurosymbolic-necessary}
If the neural encoder is not injective and zero-error identity is 
required, then a symbolic handle is necessary: no shape-only decoder 
can be a left inverse of the encoder.
\end{theorem}
\leanmeta{\LH{NSL3}}

\begin{proof}
A left inverse implies injectivity. If the encoder has collisions, 
it cannot be injective, so no left inverse exists.
\end{proof}

\begin{theorem}[GPH40 characterizes neurosymbolic identity recovery]\label{thm:nsl-gph40}
The neurosymbolic link recovers identity if and only if the pair 
$(\pi, \sigma)$ jointly distinguishes all entities.
\end{theorem}
\leanmeta{\LH{NSL4}}

\begin{proof}
This is GPH40 (taskRecoverable\_pair\_iff) specialized to the identity task.
\end{proof}

\paragraph{Interpretation.}
The neurosymbolic linking theorems say: the symbolic handle carries 
the identity-resolving bit budget that the neural encoder alone cannot 
provide. The handle is not decorative; it is the paid-for side 
information required by the coding theorems (GPH32, LWDC1). In 
open-world settings where barrier-freedom is not extension-stable 
(ROB2) and certification is undecidable (UND1), the symbolic handle 
is the only component that persists under world growth.
```

### 5.2 Update Abstract

**File:** `content/abstract.tex`

**Add:**
```latex
The mechanization proves that structural observation cannot distinguish 
entities (\LH{AXI1}, \LH{IMP1}) while neurosymbolic linking---pairing 
a neural encoder with an injective symbolic handle---achieves zero-error 
identity recovery (\LH{NSL1}, \LH{NSL3}).
```

### 5.3 Update Introduction

**File:** `content/01_introduction.tex`

**Add to contributions:**
```latex
\item \textbf{Neurosymbolic linking as the nominal layer.} The paper 
formalizes neurosymbolic linking---pairing a neural encoder with an 
injective symbolic handle---as the architectural realization of the 
identity-restoring nominal layer. The mechanization proves that 
neurosymbolic linking achieves zero-error identity (\LH{NSL1}), that 
shape-alone does not (\LH{NSL2}), and that neurosymbolic linking is 
necessary when the encoder has collisions (\LH{NSL3}).
```

### 5.4 Update Conclusion

**File:** `content/07_conclusion.tex`

**Add:**
```latex
The paper formalizes neurosymbolic linking as the architectural 
realization of the identity-restoring nominal layer. When a neural 
encoder produces a structural representation and a symbolic handle 
provides injective identity, the system achieves zero-error identity 
recovery (\LH{NSL1}). The symbolic handle is not philosophical 
decoration; it is the paid-for bit budget that resolves the residual 
ambiguity left by the semantic compression.
```

### Phase 5 Deliverables

- [ ] Section 6 updated with NSL theorems
- [ ] Abstract updated
- [ ] Introduction updated
- [ ] Conclusion updated
- [ ] PDF builds successfully
- [ ] All NSL handles render correctly

---

## Phase 6: Open-World Integration

**Goal:** Connect neurosymbolic linking to open-world stability.

**Effort:** 2-3 hours
**Risk:** MEDIUM (may need additional theorems)

### 6.1 Add Theorem: Persistence Under Growth

**File:** `proofs/AbstractClassSystem/Neurosymbolic.lean`

**Theorem:**
```lean
/-! ## Open-World Persistence -/

/-- A neurosymbolic link persists under world extension if the handle 
    remains injective on the extended entity set. -/
theorem neurosymbolic_persists_under_extension
    {α : Type*} [DecidableEq α]
    (ns : NeurosymbolicLink α _)
    (W W' : Finset Entity)
    (hsub : W ⊆ W')
    (hext : ∀ e ∈ W', ∃ e' ∈ W, ns.handle e = ns.handle e') :
    -- If the handle on W' is determined by handle on W
    -- Then the link persists
    maxFiberCard (ns.observe) ≤ 1 := by
  -- The handle remains injective, so maxFiberCard ≤ 1
  exact ns.zero_error_identity
```

**Note:** This may need refinement based on how we model "world extension."

### 6.2 Connect to ROB1/ROB2

**File:** `content/04a_complexity_bounds.tex`

**Add paragraph:**
```latex
\paragraph{Neurosymbolic persistence.}
The open-world instability theorems (ROB1, ROB2) show that barrier-freedom 
is not preserved under extension. However, neurosymbolic linking provides 
a persistence mechanism: the symbolic handle, being injective by 
construction, maintains zero-error identity even when the semantic 
representation develops new collisions. This is the architectural 
content of NSL1: the identity layer persists because it does not depend 
on the semantic representation's collision structure.
```

### Phase 6 Deliverables

- [ ] Persistence theorem proved (or documented as future work)
- [ ] Connection to ROB1/ROB2 in paper text
- [ ] PDF builds successfully

---

## Phase 7: Final Integration and Verification

**Goal:** Complete integration and verify everything works.

**Effort:** 2 hours
**Risk:** LOW

### 7.1 Full Lean Build

**Command:**
```bash
cd /home/ts/code/projects/papers-archive/docs/papers/paper1_typing_discipline/proofs
lake build
# Expect: Build success, 0 sorries, 399+ theorems (6 new)
```

### 7.2 Full Paper Build

**Command:**
```bash
cd /home/ts/code/projects/papers-archive
python3 scripts/build_papers.py latex paper1_jsait
# Expect: PDF builds, all handles resolve
```

### 7.3 Verify Claim Mapping

**File:** `content/claim_mapping_auto.tex`

**Expected:** All new claims (NSL1-6) appear with status "Full"

### 7.4 Verify Lean Stats

**File:** `content/lean_stats.tex`

**Expected:** 
- Lines: 9000+ (from 8952)
- Theorems: 399+ (from 393)
- Files: 31 (from 30)

### Phase 7 Deliverables

- [ ] Lean build succeeds with 0 sorries
- [ ] Paper PDF builds successfully
- [ ] All NSL handles in claim_mapping_auto.tex with status "Full"
- [ ] lean_stats.tex updated with new counts
- [ ] Manual review: all new content renders correctly

---

## Phase 8: Commit and Document

**Goal:** Commit all changes with clear documentation.

**Effort:** 1 hour
**Risk:** LOW

### 8.1 Commit Lean Changes

**Command:**
```bash
cd /home/ts/code/projects/papers-archive
git add docs/papers/paper1_typing_discipline/proofs/AbstractClassSystem/Neurosymbolic.lean
git add docs/papers/paper1_typing_discipline/proofs/HandleAliases.lean
git commit -m "lean: add NeurosymbolicLink structure and theorems (NSL1-6)

- Define NeurosymbolicLink structure pairing neural encoder with symbolic handle
- Prove neurosymbolic_zero_error_identity (NSL1): injective handle achieves D=0
- Prove shape_alone_not_zero_error (NSL2): shape collisions prevent D=0
- Prove neurosymbolic_necessary_for_zero_error (NSL3): handle required when encoder has collisions
- Prove neurosymbolic_identity_recovery (NSL4): GPH40 connection
- Prove neurosymbolic_injective_implies_recoverable (NSL5): corollary
- Prove neurosymbolic_is_canonical_helper_view (NSL6): canonical form

This mechanizes the neurosymbolic linking story for the JSAIT paper."
```

### 8.2 Commit Paper Changes

**Command:**
```bash
git add docs/papers/paper1_typing_discipline/latex_jsait/content/*.tex
git commit -m "paper1: integrate neurosymbolic linking mechanization

- Add AXI1, IMP1, EMB1 to paper (structural cannot distinguish)
- Add PRIV1, PRIV2 to paper (disclosure separation)
- Add NOM1, NOM2 to paper (nominal typing properties)
- Add new Section 6 subsection: Neurosymbolic Linking as the Nominal Layer
- Add NSL1-6 theorems with Lean backing
- Update abstract, introduction, conclusion
- Frame neurosymbolic linking as architectural realization of identity bit cost"
```

### 8.3 Update Planning Documents

**Files to update:**
- `paper1_jsait_neurosymbolic_compression_plan.md` - mark complete
- `paper1_lean_tactical_plan.md` - mark complete
- Create `paper1_jsait_neurosymbolic_completion_summary.md`

### Phase 8 Deliverables

- [ ] Lean changes committed
- [ ] Paper changes committed
- [ ] Planning documents updated
- [ ] Commit history clean and documented

---

## Execution Timeline

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 0 | Preparation | 0.5h | None |
| 1 | Expose existing handles | 2-3h | Phase 0 |
| 2 | NeurosymbolicLink structure | 2h | Phase 0 |
| 3 | Core theorems | 4-6h | Phase 2 |
| 4 | Register handles | 1h | Phase 3 |
| 5 | Integrate with paper | 2h | Phase 4 |
| 6 | Open-world integration | 2-3h | Phase 3 |
| 7 | Final verification | 2h | Phases 1-6 |
| 8 | Commit and document | 1h | Phase 7 |

**Total: 16-20 hours**

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Theorem 2 proof difficult | MEDIUM | MEDIUM | Can mark as sorry temporarily, document in paper |
| Handle registration fails | LOW | LOW | Build system well-tested |
| Paper text too long | MEDIUM | LOW | Can compress other sections |
| Open-world theorem complex | MEDIUM | LOW | Can document as future work |

---

## Success Criteria

### Minimal Success (Phases 0-5)
- [ ] Existing handles (AXI1, IMP1, EMB1, PRIV1-2, NOM1-2) exposed in paper
- [ ] NeurosymbolicLink structure defined and compiling
- [ ] NSL1, NSL3-6 proved (NSL2 can be sorry)
- [ ] Handles registered and cited in paper
- [ ] PDF builds successfully

### Full Success (Phases 0-8)
- [ ] All NSL1-6 theorems proved with no sorries
- [ ] All existing handles exposed
- [ ] Open-world connection documented
- [ ] Claim mapping shows all NSL handles as "Full"
- [ ] Lean stats updated
- [ ] Clean commit history
- [ ] Planning documents complete

---

## Rollback Plan

If issues arise:

1. **Lean build fails:** Revert to last known good commit, isolate failing theorem
2. **Paper too long:** Prioritize NSL1, NSL3, NSL4; others can be appendix
3. **Theorem 2 intractable:** Mark as sorry, document in paper as "proof in progress"
4. **Handle registration fails:** Debug build system, can proceed with manual handle definitions

---

## Post-Completion Tasks

1. **Update JSAIT submission package**
2. **Regenerate release artifacts**
3. **Update paper1_jsait_revision_plan.md**
4. **Consider arXiv update**

---

## Appendix: Theorem Dependency Graph

```
NeurosymbolicLink structure
    │
    ├─► NSL1: neurosymbolic_zero_error_identity
    │       └─► uses: maxFiberCard_le_one_of_injective (GPH25/26/27)
    │
    ├─► NSL2: shape_alone_not_zero_error
    │       └─► uses: AXI1, IMP1 (shape cannot distinguish)
    │
    ├─► NSL3: neurosymbolic_necessary_for_zero_error
    │       └─► uses: Function.LeftInverse.injective
    │
    ├─► NSL4: neurosymbolic_identity_recovery
    │       └─► uses: taskRecoverable_pair_iff (GPH40)
    │
    ├─► NSL5: neurosymbolic_injective_implies_recoverable
    │       └─► uses: NSL4
    │
    └─► NSL6: neurosymbolic_is_canonical_helper_view
            └─► uses: maxFiberCard_le_one_of_injective
```

---

## Appendix: File Change Summary

| File | Change Type | Lines Changed |
|------|-------------|---------------|
| `AbstractClassSystem/Neurosymbolic.lean` | NEW | ~200 |
| `HandleAliases.lean` | MODIFY | +20 |
| `02_compression_framework.tex` | MODIFY | +10 |
| `04a_complexity_bounds.tex` | MODIFY | +15 |
| `06_applications.tex` | MODIFY | +80 |
| `abstract.tex` | MODIFY | +5 |
| `01_introduction.tex` | MODIFY | +10 |
| `07_conclusion.tex` | MODIFY | +10 |
| `lean_handle_ids_auto.tex` | AUTO | +6 |
| `lean_stats.tex` | AUTO | +6 |
| `claim_mapping_auto.tex` | AUTO | +6 |

**Total new/modified lines: ~350**

---

## End of Plan
