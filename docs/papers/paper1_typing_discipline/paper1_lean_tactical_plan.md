# Paper 1 Lean Tactical Plan: Formalizing Neurosymbolic Compression Claims

## Goal

Ground the neurosymbolic semantic compression story in Lean 4 mechanization so that claims about:
- Open-world D=0 necessity
- Lossless/lossy identity compression
- Nominal layer as identity-restoring side information
- Security guarantees via pay-in-bits

are either already mechanized, can be mechanized from existing infrastructure, or are clearly identified as future work.

---

## Current Mechanization Inventory

### Already in Paper (Strong)

| CLAIM | HANDLES | FILE | STATUS |
|-------|---------|------|--------|
| D=0 iff injective | GPH27 | GraphEntropy.lean | Complete |
| Collision converse | LWDC1 | lwd_converse.lean | Complete |
| Maximal barrier converse | LWDC3 | lwd_converse.lean | Complete |
| Task sufficiency | GPH38, GPH39 | GraphEntropy.lean | Complete |
| Helper view sufficiency | GPH40, GPH41 | GraphEntropy.lean | Complete |
| Factorized product law | GPH42, GPH43 | GraphEntropy.lean | Complete |
| Adaptive fiber budget | GPH32, GPH33 | GraphEntropy.lean | Complete |
| Entropy bounds | GPH34, GPH35 | GraphEntropy.lean | Complete |
| Distortion floor | GPH30 | GraphEntropy.lean | Complete |
| Coarsening monotonicity | GPH28, GPH37 | GraphEntropy.lean | Complete |
| Block coding threshold | GPH13, GPH16-20 | GraphEntropy.lean | Complete |
| Open-world instability | ROB1, ROB2 | Undecidability.lean | Complete |
| Certification undecidability | UND1, UND2 | Undecidability.lean | Complete |

### Defined but Unused (Expose)

| CLAIM | HANDLES | FILE | WHAT IT PROVES |
|-------|---------|------|----------------|
| Shape cannot distinguish | AXI1 | Core.lean | Same-namespace types indistinguishable by shape |
| Shape cannot track provenance | IMP1 | Core.lean | Shape typing cannot report source/identity |
| Nominal non-embeddability | EMB1 | Extended.lean | Nominal info cannot embed into structural |
| Extension impossibility | EXT1 | Extended.lean | Shape-only extensions stay shape-bound |
| Protocol is concession | ALT1, ALT2 | Extended.lean | Shape protocol is not an alternative, it's a forced concession |
| Java forced to composition | IMP2 | Extended.lean | Concession example |
| Nominal disclosure = 1 | PRIV1 | Extended.lean | Nominal identity resolution costs 1 disclosure step |
| Attribute-only disclosure >= n-1 | PRIV2 | Extended.lean | Tag-free identity resolution costs n-1 steps |
| Identity disclosure separation | PRIV3 | Extended.lean | Formal separation between nominal and attribute-only |
| Injectivity characterization | GPH25, GPH26 | GraphEntropy.lean | maxFiberCard = 1 variants |
| Analysis has positive EV | BRG1 | Bridge.lean | Decision-theoretic justification |
| Ignorant choice has cost | BRG2 | Bridge.lean | Not analyzing has expected cost |
| Retrofit dominates | BRG3 | Bridge.lean | Retrofit cost > greenfield |

---

## Claim-by-Claim Formalization Plan

### CLAIM 1: "Lossless Identity Compression = Pay Exact Bits"

**What we want to say:**
- Zero-error identity recovery requires exactly log2(maxFiberCard) bits
- This is the lossless coding rate for identity
- No scheme can beat this without accepting D > 0

**Current mechanization:**
- GPH27 (D=0 iff injective) - in paper
- LWDC1 (collision_block_requires_bits) - in paper
- GPH32-35 (adaptive budgets, entropy bounds) - in paper

**Gap:**
- No explicit theorem saying "lossless identity compression rate = log2(maxFiberCard)"
- The pieces exist but are not unified under a single "lossless rate" statement

**Tactic: EXPOSE + MINOR BRIDGE**

Add theorem (or lemma wrapper):
```lean
theorem lossless_identity_compression_rate :
    zero_error_identity_rate π = log2 (maxFiberCard π)
```
where `zero_error_identity_rate` is defined as the minimal expected bits for D=0.

**Effort:** LOW - just a definitional wrapper around existing GPH32-35

**Claim strength after:** "The lossless identity compression rate is exactly log2(maxFiberCard), mechanized as GPH44."

---

### CLAIM 2: "Lossy Identity Compression = Accept Distortion Floor"

**What we want to say:**
- If you refuse to pay the bit budget, you get D >= 1 - 2^L/a
- This is the lossy coding regime
- Trading bits for distortion is exact

**Current mechanization:**
- GPH30 (uniformFiberErrorRate_ge_one_sub_tag_ratio) - in paper

**Gap:**
- GPH30 is for uniform source on a single fiber
- No general rate-distortion theorem for identity
- No theorem saying "lossy identity compression has this exact floor"

**Tactic: EXPOSE + DOCUMENT LIMITATION**

Expose GPH30 more prominently. Add remark in paper:
> "The distortion floor theorem GPH30 gives the exact error floor for uniform collision blocks. A full rate-distortion characterization for non-uniform sources and hierarchical distortion measures is future work."

**Effort:** LOW - no new Lean needed

**Claim strength after:** "The distortion floor is mechanized for uniform collision blocks (GPH30). General rate-distortion is future work."

---

### CLAIM 3: "Shape/Structural Typing Cannot Recover Identity"

**What we want to say:**
- Structural information alone cannot distinguish entities
- This is not a limitation of specific encodings
- It's fundamental to the information that structural typing exposes

**Current mechanization:**
- AXI1 (shape_cannot_distinguish) - DEFINED BUT NOT USED
- IMP1 (shape_provenance_impossible) - DEFINED BUT NOT USED
- EMB1 (semantic_non_embeddability) - DEFINED BUT NOT USED

**Gap:**
- These theorems exist but are invisible in the paper
- The "forcing chain" is mechanized but not exposed

**Tactic: EXPOSE**

Add to paper:
- In Section 2 or Section 6, explicitly cite AXI1, IMP1, EMB1
- Frame them as: "The mechanization proves structural typing cannot distinguish same-shape entities (AXI1), cannot track provenance (IMP1), and cannot embed nominal information (EMB1)."

**Effort:** LOW - just paper text changes

**Claim strength after:** "Structural typing cannot recover identity - mechanized as AXI1, IMP1, EMB1."

---

### CLAIM 4: "Nominal Tags Are the Architectural Fix"

**What we want to say:**
- Nominal tags restore identity
- They do so by providing the exact bit budget the theorem requires
- Neurosymbolic linking is one architectural realization of this

**Current mechanization:**
- PRIV1 (nominal_identity_disclosure_constant) - DEFINED BUT NOT USED
- PRIV2 (attribute_only_identity_disclosure_lower_bound) - DEFINED BUT NOT USED
- PRIV3 (identity_disclosure_separation) - IN PAPER
- NOM1, NOM2 (nominal typing theorems) - DEFINED BUT NOT USED

**Gap:**
- PRIV1-2 exist but only PRIV3 is cited
- No explicit theorem connecting "nominal tag" to "log2(maxFiberCard) bits"

**Tactic: EXPOSE + MINOR BRIDGE**

1. Expose PRIV1-2 in paper
2. Add bridging lemma:
```lean
theorem nominal_tag_bits_suffice :
    nominal_tag_length >= ceil (log2 (maxFiberCard π))
```
This connects the nominal tag (which PRIV1 shows costs 1 disclosure step) to the bit budget (which GPH32 computes).

**Effort:** LOW-MEDIUM - one lemma connecting existing definitions

**Claim strength after:** "Nominal tags provide the exact identity-restoring bit budget - mechanized via GPH32 + PRIV1-3."

---

### CLAIM 5: "Open-World D=0 Is Not Extension-Stable"

**What we want to say:**
- A representation that is injective today may not be tomorrow
- Barrier-freedom is not preserved under world growth
- This is mechanized

**Current mechanization:**
- ROB1 (extend_to_force_barrier) - IN PAPER
- ROB2 (barrierFreedom_not_extension_stable) - IN PAPER

**Gap:**
- None - this is already strong

**Tactic: EXPOSE MORE PROMINENTLY**

Already in paper. Just frame it more centrally in the neurosymbolic story.

**Effort:** NONE

**Claim strength after:** STRONG - "Open-world D=0 is not extension-stable (ROB1, ROB2)."

---

### CLAIM 6: "D=0 Certification Is Undecidable"

**What we want to say:**
- You cannot computably verify that a representation will remain D=0
- This is a Rice-style theorem
- Certification requires an identity layer that doesn't depend on the representation

**Current mechanization:**
- UND1 (hasBarrier_not_computable) - IN PAPER
- UND2 (barrierFree_not_computable) - IN PAPER

**Gap:**
- None - this is already strong

**Tactic: EXPOSE MORE PROMINENTLY**

Already in paper. Frame as: "Because certification is undecidable (UND1, UND2), formal D=0 guarantees require an identity layer that persists independently of the semantic representation."

**Effort:** NONE

**Claim strength after:** STRONG - "D=0 certification is undecidable (UND1, UND2)."

---

### CLAIM 7: "Persistent D=0 in Open World Must Be Nominal" (THE BIG ONE)

**What we want to say:**
- If you want D=0 that survives world growth
- And you can't certify D=0 from the representation alone
- Then you MUST carry an identity layer
- The only architectural form of this is nominal (identity-based) tagging
- Neurosymbolic linking is one realization

**Current mechanization:**
- ROB1, ROB2 (instability) - STRONG
- UND1, UND2 (undecidability) - STRONG
- AXI1, IMP1, EMB1 (shape cannot distinguish) - DEFINED BUT NOT USED
- PRIV1-3 (nominal vs attribute-only separation) - PARTIAL
- NOM1, NOM2 (nominal properties) - DEFINED BUT NOT USED

**Gap:**
- No synthesis theorem connecting these pieces
- No theorem of the form: "persistent D=0 implies nominal"

**Tactic: TWO OPTIONS**

**Option A: Conservative (Expose existing, claim is "strongly suggested")**
- Expose IMP*, AXI1, EMB1, PRIV1-2, NOM1-2 in paper
- Frame as: "The mechanization shows structural typing cannot distinguish (AXI1), cannot embed nominal (EMB1), and barrier-freedom is both unstable (ROB2) and uncertifiable (UND1). This strongly suggests that persistent D=0 requires a nominal layer, though an explicit synthesis theorem would strengthen this."
- Add to future work: "A synthesis theorem directly stating 'persistent D=0 implies nominal' would complete the formal picture."

**Option B: Add Synthesis Theorem (MEDIUM effort)**
```lean
theorem persistent_zero_error_implies_nominal_identity :
    (∀ W' : Finset Typ, W ⊆ W' → BarrierFreeWorld W') →
    (∃ nominal_layer : Typ → Symbol, 
     injective_on W nominal_layer ∧ 
     (∀ W' ⊇ W, preserves_identity nominal_layer W'))
```

This theorem would say: if barrier-freedom persists under all extensions, then there exists a nominal identity layer that is injective and persists.

**Effort:** 
- Option A: LOW (paper text only)
- Option B: MEDIUM (new theorem, but infrastructure exists)

**Claim strength after:**
- Option A: "Strongly suggested by mechanization, synthesis theorem is future work"
- Option B: "Directly mechanized"

---

### CLAIM 8: "Neurosymbolic Linking = Architectural Realization"

**What we want to say:**
- When a neural encoder produces representation U and a symbolic layer attaches handle S
- The system is doing exactly what the theorem prescribes
- S is the identity-carrying bit budget

**Current mechanization:**
- The coding theorems (GPH32-35, LWDC1-3) specify the bit budget
- PRIV1-3 show nominal vs attribute-only separation
- But no theorem explicitly about "neural encoder + symbolic handle"

**Gap:**
- No Lean theorem about neural networks (not needed)
- No Lean theorem explicitly named "neurosymbolic_linking" (not needed)

**Tactic: INTERPRETATIVE (no new Lean)**

This is an architectural interpretation of existing theorems, not a new formal claim. The paper can say:

> "Neurosymbolic linking can be read as an architectural realization of the identity-carrying nominal layer required by the theorem. The symbolic handle S is the log2(maxFiberCard) bits that GPH32 prescribes. The neural encoder U is the semantic representation that leaves residual ambiguity. The linking relation is the architectural mechanism for carrying the identity-resolving side information."

**Effort:** NONE - this is paper framing only

**Claim strength after:** "Interpretative claim grounded in GPH32-35, LWDC1-3, PRIV1-3."

---

### CLAIM 9: "Security Guarantees via Pay-in-Bits"

**What we want to say:**
- For security-critical systems, D=0 may be a hard requirement
- The theorem tells you the exact bit cost
- The distortion floor tells you the exact error if you refuse to pay
- This is not a risk assessment - it's a mathematical lower bound

**Current mechanization:**
- GPH30 (distortion floor) - IN PAPER
- LWDC1-3 (bit requirements) - IN PAPER

**Gap:**
- No theorem explicitly about "security" (not needed)
- Theorems are about D, not "security"

**Tactic: INTERPRETATIVE (no new Lean)**

Frame in paper:
> "For security-critical applications where D=0 is a hard requirement, the theorem provides an exact accounting: either the representation is injective, or nominal bits must be paid, or the distortion floor D >= 1 - 2^L/a is accepted. This is not a probabilistic risk assessment - it is a mathematical lower bound (GPH30)."

**Effort:** NONE - paper framing only

**Claim strength after:** "Security implication grounded in GPH30, LWDC1-3."

---

## Priority Summary

### Priority 1: EXPOSE (Paper text changes only, no new Lean)

| HANDLE | CLAIM SUPPORTED | WHERE TO ADD |
|--------|-----------------|--------------|
| AXI1 | Shape cannot distinguish | Section 2 or 6 |
| IMP1 | Shape cannot track provenance | Section 2 or 6 |
| EMB1 | Nominal non-embeddability | Section 6 |
| PRIV1 | Nominal disclosure = 1 | Section 6 or 7 |
| PRIV2 | Attribute-only disclosure >= n-1 | Section 6 or 7 |
| NOM1, NOM2 | Nominal typing properties | Section 6 |

### Priority 2: MINOR BRIDGES (Low effort new lemmas)

| NEW THEOREM | CONNECTS | EFFORT |
|-------------|----------|--------|
| lossless_identity_compression_rate | GPH32-35 → single statement | LOW |
| nominal_tag_bits_suffice | PRIV1 → GPH32 | LOW-MEDIUM |

### Priority 3: SYNTHESIS (Medium effort, optional)

| NEW THEOREM | WHAT IT PROVES | EFFORT |
|-------------|----------------|--------|
| persistent_zero_error_implies_nominal_identity | "Persistent D=0 implies nominal" | MEDIUM |

### Priority 4: DOCUMENT LIMITATIONS

| LIMITATION | CURRENT STATUS | HOW TO ADDRESS |
|------------|----------------|----------------|
| General rate-distortion | Only GPH30 for uniform | "Future work" in paper |
| Neural network theorems | Out of scope | Interpretative only |
| Probabilistic extensions | Deterministic only | "Future work" in paper |

---

## Recommended Execution Order

### Phase 1: Paper Text (1-2 hours)
1. Add AXI1, IMP1, EMB1 to Section 2 or 6
2. Add PRIV1-2, NOM1-2 to Section 6 or 7
3. Frame neurosymbolic linking as architectural interpretation
4. Frame security guarantees as interpretative

### Phase 2: Minor Lean Bridges (2-4 hours, optional)
1. Add `lossless_identity_compression_rate` wrapper
2. Add `nominal_tag_bits_suffice` connector lemma
3. Register new handles GPH44, GPH45

### Phase 3: Synthesis Theorem (4-8 hours, optional)
1. Define `persistent_barrier_free` predicate
2. Prove `persistent_zero_error_implies_nominal_identity`
3. Register handle NOM3 or SYN1

### Phase 4: Update Paper Claims
1. Update abstract to cite new handles
2. Update introduction contributions
3. Update conclusion

---

## Claim Strength After Each Phase

| CLAIM | AFTER PHASE 1 | AFTER PHASE 2 | AFTER PHASE 3 |
|-------|---------------|---------------|---------------|
| Lossless = pay bits | STRONG (GPH32-35) | STRONG + explicit (GPH44) | Same |
| Lossy = distortion floor | MODERATE (GPH30) | Same | Same |
| Shape cannot distinguish | STRONG (AXI1, IMP1, EMB1 exposed) | Same | Same |
| Nominal fixes it | STRONG (PRIV1-3 exposed) | STRONG + explicit (GPH45) | Same |
| Open-world unstable | STRONG (ROB1, ROB2) | Same | Same |
| Certification undecidable | STRONG (UND1, UND2) | Same | Same |
| Persistent D=0 must be nominal | SUGGESTED | SUGGESTED + connected | **DIRECT** (SYN1) |
| Neurosymbolic = architecture | INTERPRETATIVE | Same | Same |
| Security via pay-in-bits | INTERPRETATIVE | Same | Same |

---

## Minimal Viable Path

If time is limited, the **minimum** to make the neurosymbolic story credible:

1. **Expose AXI1, IMP1, EMB1** - proves structural typing cannot distinguish
2. **Expose PRIV1-2** - proves nominal vs attribute-only separation
3. **Frame neurosymbolic linking as interpretation** - connects to GPH32-35
4. **Frame "must be nominal" as strongly suggested** - honest about synthesis gap

This requires **zero new Lean** and makes the story significantly stronger.

---

## Honest Limitations to State in Paper

1. **"The distortion floor theorem GPH30 applies to uniform collision blocks. A full rate-distortion characterization for general sources is future work."**

2. **"The mechanization strongly suggests that persistent certified D=0 requires a nominal layer, but an explicit synthesis theorem directly stating this conclusion would strengthen the formal foundation."**

3. **"The neurosymbolic linking interpretation is architectural: the theorems specify the bit budget and the separation between nominal and attribute-only identity resolution, but do not directly formalize neural encoders or symbolic handles."**

---

## Conclusion

The Lean infrastructure is **already sufficient** to ground the neurosymbolic compression story at a level that exceeds typical information-theory papers. The main work is:

1. **Exposing what already exists** (IMP*, PRIV*, NOM*)
2. **Minor bridging lemmas** (optional but nice)
3. **Honest framing** of what's mechanized vs. interpretative

The "no-free-lunch" thesis is **strongly supported** by the combination of:
- ROB1, ROB2 (instability)
- UND1, UND2 (undecidability)
- AXI1, IMP1, EMB1 (structural cannot distinguish)
- PRIV1-3 (nominal vs attribute-only separation)

A synthesis theorem would make it **direct**, but the pieces already form a coherent picture.
