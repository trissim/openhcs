# Plan: No Hidden Registries Theorem (Revised)

## The Claim

**Theorem:** In any admissible system, every semantically relevant read/write effect in a valid execution trace targets declared state. Hence undeclared state is semantically inert relative to the model.

**NOT:** "If it's not declared, it doesn't exist." (Too strong - reviewers will attack this)

**INSTEAD:** "Undeclared state is semantically inaccessible in the admissible model. Any purported dependence on it is a specification violation, not an uncharged informational resource."

**Paper Placement:** Introduction (admissibility contract section)

## Why This Matters

1. **Eliminates escape hatch:** Critics might say "maybe there's a hidden registry." This theorem closes that door formally.
2. **Follows existing pattern:** Paper has `no_additional_observables` - this is the computational analogue.
3. **Defensible:** "Semantic closure" is provable and hard to attack.

## The Lean Formalization (Revised Architecture)

### Layer 1: Declared Interface

```lean
structure RegDecl where
  name : String
deriving DecidableEq

structure SystemSpec where
  declared   : Finset RegDecl
  Primitive  : Type
  effectsOf  : Primitive → List ComputationEffect
```

### Layer 2: Computation Effects

```lean
inductive ComputationEffect where
  | read_reg  (r : String)
  | write_reg (r : String)
deriving DecidableEq

def effectName : ComputationEffect → String
  | .read_reg r  => r
  | .write_reg r => r
```

### Layer 3: Derived Definitions

```lean
def declaredNames (Σ : SystemSpec) : Finset String :=
  Σ.declared.image RegDecl.name

def PrimitiveWellFormed (Σ : SystemSpec) : Prop :=
  ∀ p e, e ∈ Σ.effectsOf p → effectName e ∈ declaredNames Σ
```

### Layer 4: Execution Semantics

```lean
abbrev ExecutionTrace := List ComputationEffect

inductive Executes (Σ : SystemSpec) : List Σ.Primitive → ExecutionTrace → Prop
  | nil :
      Executes Σ [] []
  | cons :
      Executes Σ ps t →
      Executes Σ (p :: ps) (Σ.effectsOf p ++ t)
```

### Main Theorem (by induction)

```lean
theorem effects_in_valid_trace_are_declared
    (Σ : SystemSpec)
    (hWF : PrimitiveWellFormed Σ)
    {ps : List Σ.Primitive} {t : ExecutionTrace}
    (hex : Executes Σ ps t)
    {e : ComputationEffect}
    (he : e ∈ t) :
    effectName e ∈ declaredNames Σ := by
  induction hex with
  | nil =>
      simp at he
  | cons hex ih =>
      simp at he
      cases he with
      | inl hin =>
          exact hWF _ _ hin
      | inr hin =>
          exact ih hin
```

### Corollary

```lean
theorem undeclared_reg_not_in_valid_trace
    (Σ : SystemSpec)
    (hWF : PrimitiveWellFormed Σ)
    {r : String}
    (hnd : r ∉ declaredNames Σ)
    {ps : List Σ.Primitive} {t : ExecutionTrace}
    (hex : Executes Σ ps t) :
    ComputationEffect.read_reg r ∉ t ∧
    ComputationEffect.write_reg r ∉ t := by
  constructor <;> intro h
  · have hr : effectName (.read_reg r) ∈ declaredNames Σ :=
      effects_in_valid_trace_are_declared Σ hWF hex h
    simpa [effectName] using hr
  · have hw : effectName (.write_reg r) ∈ declaredNames Σ :=
      effects_in_valid_trace_are_declared Σ hWF hex h
    simpa [effectName] using hw
```

### Alternative: Typed Registry (stronger discipline)

```lean
structure RegRef (Σ : SystemSpec) where
  name : String
  declared : name ∈ declaredNames Σ

inductive ComputationEffect (Σ : SystemSpec) where
  | read_reg  (r : RegRef Σ)
  | write_reg (r : RegRef Σ)
```

Then hidden registers are literally unrepresentable in the semantics.

## Proof Strategy

### Lemma 1: Primitive Well-Formed
Every primitive operation accesses only declared registers.

**Proof:** By assumption / construction of scratch-built system.

### Lemma 2: Execution Closure (main theorem)
By induction on `Executes`: if primitives are well-formed, traces are well-formed.

### Corollary: Undeclared = Semantically Inert
If r is not declared, it cannot appear in any valid trace.

## Reviewer Pushback Strategy

### "This is obvious / hand-wavy"
**Response:** The closure property is proved by induction from primitive well-formedness to full executions. Not merely asserted.

### "This is just definitional"
**Response:** The closure theorem is not definitional - it requires inductive proof over execution traces. The typed registry version is by construction, but the trace-closure version is substantive.

### "What about timing side channels?"
**Response:** This theorem is semantic, not physical. Timing and physical leakage are excluded by the admissibility contract, not treated as legal semantic resources.

### "What about compiler bugs?"
**Response:** If compiled execution depends on undeclared state, that's a model violation - not a semantic freebie.

### "What about hardware backdoors?"
**Response:** Out of scope. The theorem applies to formal semantics of admissible systems, not physical implementations.

## Paper Text (Revised)

**Good version:**

> We formalize a declared-interface closure property: in any admissible system, every semantically relevant read/write effect in a valid execution trace targets declared state. Hence undeclared state is semantically inert relative to the model; any purported dependence on it is a specification violation rather than an uncharged informational resource.

**Bad version (do NOT use):**

> "If it's not declared, it doesn't exist."

## Integration Points

### Lean
- File: `discipline_migration.lean`
- Handle alias: HR1 (Hidden Registry 1)
- Main theorem: `effects_in_valid_trace_are_declared`
- Corollary: `undeclared_reg_not_in_valid_trace`

### Paper
- Add to 01_introduction.tex after admissibility contract paragraph
- Reference Lean theorem with `\LH{HR1}`

## Execution Order

1. Define `RegDecl` and `SystemSpec` structures
2. Define `ComputationEffect` and `effectName`
3. Define `declaredNames` and `PrimitiveWellFormed`
4. Define `Executes` inductive relation
5. Prove `effects_in_valid_trace_are_declared` (by induction)
6. Prove `undeclared_reg_not_in_valid_trace`
7. Add to HandleAliases.lean with HR1
8. Update paper text in 01_introduction.tex

## Summary

| Aspect | Old (Wrong) | New (Right) |
|--------|-------------|-------------|
| Claim | "Undeclared = non-existent" | "Undeclared = semantically inaccessible" |
| Proof | Undefined `execution_traces` | Induction on `Executes` |
| Corollary | Quantifies over ALL traces | Quantifies over VALID traces |
| Vulnerability | Attackable from both sides | Semantic closure is provable |
