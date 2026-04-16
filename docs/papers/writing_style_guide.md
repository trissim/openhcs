# Paper Writing Style Guide

This guide is for agents editing papers in this repository.

The goal is not to make the prose "fancier." The goal is to make it:

- direct
- formal
- reader-facing
- scope-accurate
- non-defensive
- easy to verify

Use this guide when revising abstracts, introductions, theorem framing, related work, and conclusions.

## Core Rule

Write declaratively.

Do not narrate what the paper is doing.
Do not announce comparisons, analogies, or objections.
Do not explain that a statement is important. Just state it clearly.

Bad:

- `The comparison is direct: exact relevance certification also asks which coordinates matter...`
- `This is the precise answer to the domain-restriction objection.`
- `The present paper does not identify such an invariant.`

Good:

- `Exact relevance certification likewise asks which coordinates matter...`
- `Domain restriction is governed by orbit-gap freedom on the restricted closure-closed domain.`
- `No such complete invariant is identified.`

## Preferred Tone

Use:

- short declarative sentences
- precise scope statements
- theorem-first exposition
- explicit definitions in the paper text
- calm confidence

Avoid:

- defensive prose
- self-reference
- signposting phrases
- hype
- apology
- meta commentary about what the paper is "trying" to do

## What To Avoid

### 1. Self-referential paper talk

Avoid phrases like:

- `this paper`
- `the present paper`
- `the present manuscript`
- `here we show`
- `identified here`
- `in the present setting`

Exceptions:

- neutral references like `Section 4 proves...` are fine
- direct artifact references like `The Lean artifact formalizes...` are fine

Preferred replacements:

- `No such complete invariant is identified.`
- `The theorem uses only closure-law invariance.`
- `The result isolates the direct closure-invariant regime.`

### 2. Signposting and announcement language

Avoid phrases like:

- `The comparison is direct`
- `The point is`
- `This is why`
- `This answers`
- `The nearest analogue is`
- `The broader analogy is`
- `The clearest comparison is`

Preferred pattern:

- state the comparison directly
- state the consequence directly

Bad:

`The broader analogy is the dichotomy tradition for satisfiability and constraint satisfaction.`

Better:

`The dichotomy tradition for satisfiability and constraint satisfaction provides the broader analogy.`

Best when possible:

`Schaefer's theorem and the finite-domain CSP dichotomy theorems provide the relevant methodological comparison.`

Avoid meta-logical narration that tells the reader a result follows from the previous one instead of stating what the result adds.

Avoid phrases like:

- `Corollary 6.4 records the next logical consequence of Proposition 6.1.`
- `The next proposition is an immediate consequence of...`
- `The following corollary gives the natural extension of...`
- `This theorem is the next step in the argument.`

Why this is weak:

- it narrates the proof graph instead of the mathematical content
- it tells the reader something they can already infer from the theorem ordering and citations
- it consumes sentence budget without clarifying scope, object, or consequence
- it is not declarative in the useful sense; it comments on the paper's structure rather than the claim

Preferred pattern:

- say what the corollary/proposition identifies
- say what changes in formulation, scope, or object level
- if the dependence matters, cite the prior result without narrating the dependency

Bad:

`Corollary 6.4 records the next logical consequence of Proposition 6.1.`

Better:

`Corollary 6.4 identifies the quotient-level consequence for sufficient-set statistics.`

Bad:

`The next proposition is an immediate consequence of Proposition 6.1.`

Better:

`The proposition shows that zero-distortion summaries refine the optimizer quotient.`

Avoid genre-labeling sentences that classify the result instead of stating its content.

Avoid phrases like:

- `This is a theorem of exact semantic equivalence.`
- `This corollary is the approximation-specialized form of...`
- `This proposition is the exact-semantics specialization of...`

Why this is weak:

- it tells the reader what kind of result this is rather than what the result says
- it adds meta-description instead of mathematical content
- it usually can be deleted with no loss of meaning

Preferred pattern:

- state the identification directly
- state the changed object directly
- name the specialization only if that information changes scope in a useful way

Bad:

`This is a theorem of exact semantic equivalence.`

Better:

`Deterministic payload sufficiency and relevance are exactly exact relevance certification for the induced decision problem.`

Bad:

`This corollary is the approximation-specialized form of Corollary 6.9.`

Better:

`Approximation enters only through the specification: exact agreement is still agreement with the admissible-output relation.`

### 3. Defensive prose

Avoid phrases like:

- `not a loophole`
- `not a technical convenience`
- `not load-bearing`
- `not merely`
- `not an encoding trick`
- `not a by-product`

Often these should be replaced by a positive declarative sentence.

Bad:

`The package itself is not load-bearing.`

Better:

`The contradiction uses only closure-law invariance.`

Bad:

`Domain restriction is not a loophole.`

Better:

`Domain restriction helps only by removing orbit gaps on the restricted closure-closed domain.`

### 4. Internal repository language

Never refer to manuscripts by repo number in reader-facing prose.

Avoid:

- `Paper 2`
- `Paper 4`
- `paper3b`

Use:

- the manuscript title
- `the companion coherence manuscript`
- `the companion decision-quotient manuscript`
- or cite the actual bibliography entry

### 5. Vague claims about importance

Avoid:

- `important`
- `crucial`
- `significant`
- `major`
- `key point`

unless the sentence truly needs emphasis and the importance is mathematically specified.

Usually the fix is to state the content more concretely.

Bad:

`This crucial result shows...`

Better:

`The theorem identifies orbit gaps as the complete obstruction criterion.`

## What To Prefer

### 1. State exact scope

Strong claims are fine when they are formally supported.

Do not weaken real results out of stylistic caution.
Instead, state the exact formal scope.

Preferred pattern:

- define the formal object
- define the exact regime
- say what is local and what is imported
- distinguish unconditional statements from model-relative ones

Example:

Bad:

`Five independent frameworks all characterize the same condition.`

Better:

`Five criterion families collapse to the same rank-1 regime in the canonical decision encoding.`

### 2. Define guardrails in the paper text

If a concept matters to the theorem, define it in the manuscript.
Do not leave the actual content only in Lean.

This especially applies to:

- admissibility classes
- bounded-pattern definability
- closure operations
- encoding assumptions
- exact semantics

Preferred pattern:

- give the finite parameters explicitly
- give the membership condition explicitly
- say what the restriction excludes

### 3. Put orienting sentences before technical subregimes

If a section jumps into a specialized regime, add one direct orienting sentence.

Good pattern:

`Binary pairwise slices are the smallest representation class in which unary collapse, dense interaction, optimizer degeneracy, and the orbit-gap obstruction already all appear.`

This is not rhetorical fluff. It tells the reader why the section exists.

### 4. Elevate the strongest positive result

Do not bury the best constructive or reduction result.

If the paper has:

- one strong impossibility theorem
- one strongest positive reduction/classification theorem

then both should appear in the front matter.

For example, if there is a compression theorem that extends a bounded-case result to a larger class, mention it in:

- the abstract
- the introduction's positive-side summary
- the conclusion

### 5. Use remarks for conceptual interpretation

If a paragraph is conceptually important but structurally optional, make it a remark.

Use a remark when the text:

- interprets a proof mechanism
- gives geometric or transport intuition
- clarifies why a witness is not arbitrary

This keeps the main theorem flow clean while preserving the insight.

## Abstract Rules

An abstract should:

- define the canonical object early
- state the main positive result
- state the main negative result
- state the exact scope of the no-go
- avoid ambiguous pronouns like `them`, `this`, `that`

Bad:

`Zero-distortion summaries, quotient entropy bounds, and support counting explain them.`

Better:

`Zero-distortion summaries, quotient entropy bounds, and support counting already force the same closure laws.`

## Introduction Rules

The introduction should:

- define the semantic object cleanly
- state the main theorem in manuscript language, not repo language
- distinguish the positive and negative sides clearly
- avoid front-loading artifact statistics
- avoid sounding like a grant proposal or manifesto

Artifact statistics belong in:

- the supplement
- an appendix
- a short closing sentence if necessary

not in the opening pages unless absolutely needed.

## Related Work Rules

Related work should:

- compare directly
- not advertise that a comparison is being made
- distinguish methodological analogy from exact equivalence
- avoid overclaiming a complete analogy when only a partial one is justified

Bad:

`The broader analogy is CSP dichotomy, and this explains why...`

Better:

`CSP dichotomy provides the relevant methodological comparison. Any successful frontier theorem must therefore use stronger structure than the direct closure-invariant regime ruled out here.`

If no complete analogue is known, say so directly:

- `No complete analogue of polymorphisms is identified.`

## Conclusion Rules

The conclusion should:

- restate the canonical object
- restate the strongest positive result
- restate the sharpest impossibility result
- end with the exact open problem

Avoid:

- broad motivational flourishes
- self-congratulation
- retrospective narration of the paper

Bad:

`The present paper therefore shows that...`

Better:

`The open problem is sharp. Simple finite lists, unconstrained quotient predicates, and admissible closure-invariant classifiers of this kind cannot characterize the boundary.`

## Standalone Paper Rules

Make the prose standalone even when the proofs import companion manuscripts.

Reader-facing prose should:

- define all needed objects locally
- state all needed theorems locally
- cite companions as provenance, not scaffolding

Good:

- `the companion coherence manuscript`
- `the companion decision-quotient manuscript`
- bibliography citations

Bad:

- `Paper 2 proves...`
- `Paper 4 shows...`

## Editing Checklist

Before finishing a writing pass, check for these patterns and remove them.

Search targets:

- `this paper`
- `present paper`
- `present manuscript`
- `the point`
- `this is why`
- `this answers`
- `the comparison`
- `the analogy`
- `not a loophole`
- `not load-bearing`
- `not merely`
- `important`
- `crucial`
- `key`
- `here`

Then verify:

- all theorem-critical notions are defined in the manuscript
- the best positive result is visible early
- the strongest negative theorem is stated with exact scope
- no internal repo numbering appears in reader-facing prose
- conceptual interpretation paragraphs are either concise or moved into remarks

## Summary Rule

Do not make the prose sound smarter.
Make it harder to misunderstand.
