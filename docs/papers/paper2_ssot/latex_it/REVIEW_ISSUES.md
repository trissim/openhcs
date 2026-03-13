# Paper 2 Review Issues

This note collects the issues I found while reading the main Paper 2 manuscript source that is actually used by `coherence_it.tex`.

Scope:
- main paper source under `docs/papers/paper2_ssot/latex_it/`
- focus on rendered main PDF content, theorem/story consistency, and reviewer-risk points
- based on direct reading after the recent transitivity/equality edits
- review order should always follow `coherence_it.tex` include order, not numeric filename prefixes

Status at time of note:
- Lean build for Paper 2 proofs passes
- LaTeX build for `paper2_it` passes
- equality/transitivity prose has already been softened from an `iff` claim to a sufficient-condition route

Planning status convention for the numbered points below:
- unless explicitly marked otherwise, the **decision is closed** and the item is **implementation pending**
- this file is now a decisions-settled execution plan, not an unresolved brainstorm list

## Main Story Check

The main story still works:
- deterministic partial views induce a confusability graph
- exact recovery becomes graph colorability
- repeated composition becomes strong powers
- asymptotic rate becomes Shannon capacity with theta-style upper bounds
- transitivity gives a route to the cluster-graph equality regime
- meet-witnessing and fiber coherence are stronger sufficient conditions

The main remaining issues are mostly about manuscript structure, section flow, repetition, and a few reference/mechanization hygiene problems.

## High-Priority Issues

### 1. Section-order / transition mismatch
- `content/07_graph.tex:139` says: "The next subsection makes that lift explicit."
- But the next included file in `coherence_it.tex` is `content/08_affine.tex`, not `content/09_capacity.tex`.
- This creates a real reading break in the main PDF.

User check:
- the quoted passage is **not coherent in the current PDF order**
- the sentence promises a move to strong powers / asymptotic rates, but the reader is immediately taken into affine determination instead

Reader-experience explanation:
- in the current order, the reader climbs the main graph theorem staircase and then gets pulled sideways into a specialization before the graph-capacity arc reaches its payoff
- if the order is changed to graph -> capacity -> theta -> equality -> affine, the reader experiences one continuous escalation: one-shot object -> asymptotic growth -> upper theory -> equality/sharpness -> specialization
- the benefit is narrative momentum, not a mathematical change

Why it matters:
- the graph section naturally points to asymptotic capacity
- the reader is instead moved into the affine matroid section
- this makes the paper feel out of order even if each section is individually fine

Dry-run outcome:
- after walking the manuscript in actual PDF/include order, I would reorder the main arc to:
  - graph -> capacity -> theta -> equality -> affine
- then I would update the roadmap language in `content/01_introduction.tex` and the closing transition in `content/07_graph.tex` to match that order

Why this is the endpoint I would choose:
- it preserves the reader's sense of escalation through the main theorem arc
- it lets equality land immediately after the upper-theory setup that motivates it
- it makes affine read as a rewarding dual/specialized payoff rather than as an interruption in the main climb

Status:
- decision closed
- implementation pending

### 2. Introduction section-order description does not match actual manuscript order
- `content/01_introduction.tex:40` says Sections `graph-characterization` through `equality` contain the main graph-capacity arc, and then Section `affine` comes after that.
- Actual include order in `coherence_it.tex` places `content/08_affine.tex` before `content/09_capacity.tex`, `content/09_theta.tex`, and `content/10_equality.tex`.

Why it matters:
- the introduction promises one structure
- the paper body delivers another
- reviewers notice this kind of mismatch quickly

Action item:
- once Point 1 is resolved, update `content/01_introduction.tex` so the roadmap follows the actual PDF order exactly

Status:
- decision closed
- implementation pending

### 3. Broken cross-reference in evaluation section
- `content/05_evaluation.tex:3` uses `Corollary~\ref{ssot-iff}`
- the actual label is `thm:ssot-iff` in `content/04_requirements.tex:101`

Why it matters:
- broken refs are easy reviewer bait
- this section is already short, so every visible issue stands out more

Action item:
- fix `\ref{ssot-iff}` to `\ref{thm:ssot-iff}`

Status:
- decision closed
- implementation pending

## Equality / Reviewer-Concern Status

### 4. Equality/transitivity prose is much safer now, but still needs disciplined framing
Current state:
- `content/10_equality.tex` now states the mechanized implication
  - transitivity of confusability implies component-cluster collapse
- `content/01_introduction.tex`, `content/abstract.tex`, and `content/09_conclusion.tex` have been softened to match this
- transitivity is explicit in the main manuscript in the abstract, introduction, capacity section, equality section, related work, conclusion, and Lean appendix; it is present often enough to anchor the story, but it is still concentrated rather than repeated everywhere

Why this is mostly good:
- it no longer overdepends on the stronger informal `iff` wording in the prose
- it still preserves the paper's intended story

Remaining caution:
- `content/10_equality.tex:11` still says: "That equality is no longer confined to the abstract cluster-graph subclass."
- This is probably acceptable, but it is rhetorically hot if skimmed out of context

Action item:
- keep transitivity explicit in the current strategic locations
- tighten `content/10_equality.tex:11` so the transitivity condition is named in the same sentence as the export claim

Status:
- decision closed
- implementation pending

### 5. Equality prose should sound diplomatically confident, not defensive
Current formal situation:
- the Lean source currently contains both:
  - the forward route used in the manuscript
  - a stronger formal equivalence theorem `confusableTransitive_iff_clusterCollapse`

Why this matters:
- the manuscript should not sound apologetic or timid
- it should state the mechanized route with confidence while avoiding unnecessary overclaiming

Action item:
- rewrite any sentence that sounds like "we are backing off" into positive language of the form:
  - transitivity gives the model-side export to the cluster-graph equality case
  - meet-witnessing and fiber coherence are structural sufficient conditions
- keep the tone confident, direct, and mathematical

Status:
- decision closed
- implementation pending

## Medium-Priority Structural Issues

### 6. The affine section interrupts the strongest narrative arc
- `content/08_affine.tex` is understandable and mathematically useful
- but in the current include order it interrupts the graph -> capacity -> theta -> equality progression

Why it matters:
- the graph-capacity story is the strongest part of the manuscript
- cutting away to affine before capacity makes the argument feel less cumulative

Reader-experience explanation:
- affine is easier to absorb after the reader already understands the main graph-capacity climb
- putting it earlier does not simplify the paper; it asks the reader to absorb a specialization before the general arc has paid off

Dry-run outcome:
- I would not move affine earlier
- I would move affine later, after equality

Why this is the endpoint I would choose:
- putting affine earlier does not simplify the paper for the reader; it asks for a specialization before the general arc has paid off
- moving affine later preserves the strongest narrative sequence and makes the matroid section feel like an earned second lens on the same model

Implementation note under the chosen reordering:
- the end of `content/07_graph.tex` should point directly to asymptotic rates / strong powers
- the opening of `content/08_affine.tex` should then frame affine duality as a later specialized second lens on the same confusability structure

Status:
- decision closed
- implementation pending

### 7. The paper repeats the unit-rate / integrity / manual-cost story many times
Repeated across:
- `content/01_introduction.tex`
- `content/02_foundations.tex`
- `content/03_ssot.tex`
- `content/12_rate_corollaries.tex`
- `content/04_requirements.tex`
- `content/09_conclusion.tex`

Why it matters:
- repetition can make the paper feel like two papers layered together
- the graph-capacity arc is the main technical novelty, but the integrity/rate framing sometimes takes over too often

Action item:
- compress repeated threshold/integrity/rate statements and let later sections point back instead of restating the same pitch

Status:
- decision closed
- implementation pending

### 8. `content/05_evaluation.tex` is too thin
Current content:
- 5 lines total
- mostly just points to the appendix

Why it matters:
- as a standalone section in the main PDF, it reads underdeveloped
- reviewers may ask why it exists as a section rather than a paragraph or note

Recommended action:
- move the useful sentence-level payoff into `content/04_requirements.tex`
- make the main text point explicitly to `content/12_appendix_classification.tex`
- do **not** leave `content/05_evaluation.tex` as a standalone ultra-thin section if it stays this short

Status:
- decision closed
- implementation pending

### 9. `content/07_empirical.tex` is also too thin
Current content:
- 6 lines total
- acts only as a supplement pointer

Why it matters:
- same issue as evaluation: it feels like placeholder scaffolding in the main paper

Recommended action:
- treat this the same way as Point 8
- either merge its payoff sentence into the conclusion or fold it into the requirements/evaluation bridge with an appendix or supplement pointer

Status:
- decision closed
- implementation pending

### 10. `content/09_theta.tex` is very compressed relative to its conceptual importance
Current state:
- the section is clear, but short and summary-like
- it carries a lot of conceptual load because the abstract and introduction emphasize the upper theory

Why it matters:
- readers may feel the upper-theory section is underspecified compared with the space spent on framing elsewhere

Action item:
- add one short explanatory paragraph or one compact summary theorem block if space allows

Status:
- decision closed
- implementation pending

## Low-to-Medium Priority Issues

### 11. The conclusion is accurate but somewhat repetitive
- `content/09_conclusion.tex` mostly works
- but it repeats earlier framing rather than ending with a sharper punchline

Action item:
- when collapsing Points 8 and 9, move the strongest reward/payout language into the conclusion
- trim duplicated intro-style exposition and end on: model-generated graph class, capacity arc, equality route, realizability consequence, and case-study payoff

Status:
- decision closed
- implementation pending

### 12. Host-system classification language is rhetorically strong relative to the amount of main-text support
Examples:
- `content/05_evaluation.tex:3`
- `content/12_appendix_classification.tex:28`
- phrases like "apply this criterion uniformly" or "the same theorem-level coordinates apply uniformly"

Why it matters:
- the table is useful, but the main text support is brief
- reviewers may push on whether this is a theorem application or a stylized interpretive classification

Current decision:
- keep this language as-is
- rationale: the classification is intended as a mathematical application of the criterion under the stated host-model assumptions, not as a tentative anecdotal survey

Status:
- decision closed
- implementation pending only if later reviewer feedback forces revision

### 13. The manuscript has two tightly linked components that need to read as one system
Component A:
- graph-capacity theory for deterministic partial views

Component B:
- unit-rate structural integrity / realizability / host classification

Clarification:
- these are **not** separate papers or competing arcs
- they are two components of one dynamic multi-fact system story

Reader-experience explanation:
- the issue is not conceptual correctness; it is transition legibility
- a reader can still experience a topic shift from graph-capacity structure to unit-rate realizability/host interpretation unless the manuscript explicitly signals that these are two regimes of the same model
- the needed change is therefore a framing change, not a change in mathematical content

Action item:
- improve transitions so the paper reads as: graph-theoretic failure structure in the nontrivial regime, and realizability/integrity structure at the unit-rate boundary of the same model

Status:
- decision closed
- implementation pending

## Mechanization / Coverage Hygiene

### 14. Some claims in the auto mapping remain unmapped
In `content/claim_mapping_auto.tex`:
- `cor:integrity-threshold` is unmapped
- `cor:matroid-capacity-bounds` is unmapped

Why it matters:
- the paper strongly advertises mechanized support
- unmapped claims may prompt questions even if they are simple corollaries from mapped results

Action item:
- fix the unmapped claims by adding the missing derived claim mappings
- if that proves impossible in the current build pipeline, add an explicit note in the mechanization appendix explaining that these are paper-level corollaries from formalized results

Status:
- decision closed
- implementation pending

### 15. Lean-handle citations in some summary sentences are broad rather than tightly local
Examples:
- `content/09_conclusion.tex`
- some intro summary lines

Why it matters:
- not fatal, but precise handle bundles help trust

Action item:
- tighten broad handle bundles where easy, especially in summary sentences and the conclusion

Status:
- decision closed
- implementation pending

## Files That Actually Affect the Main PDF

Main manuscript inputs from `coherence_it.tex`:
- `content/abstract.tex`
- `content/01_introduction.tex`
- `content/06_figures.tex`
- `content/02_foundations.tex`
- `content/06_converse.tex`
- `content/07_graph.tex`
- `content/08_affine.tex`
- `content/09_capacity.tex`
- `content/09_theta.tex`
- `content/10_equality.tex`
- `content/03_ssot.tex`
- `content/12_rate_corollaries.tex`
- `content/04_requirements.tex`
- `content/05_evaluation.tex`
- `content/07_empirical.tex`
- `content/08_related.tex`
- `content/09_conclusion.tex`
- appendix: `content/11_lean_proofs.tex`, `content/12_appendix_classification.tex`

Files not currently used in the main PDF body:
- `content/10_rebuttals.tex`
- `content/11_affine.tex`
- supplement files under `supplementA/`
- auto-generated support files unless explicitly input elsewhere

## Recommended Review Order

If reviewing by hand, I would check in this order:
1. `coherence_it.tex` include order
2. `content/01_introduction.tex`
3. `content/07_graph.tex`
4. `content/08_affine.tex`
5. `content/09_capacity.tex`
6. `content/09_theta.tex`
7. `content/10_equality.tex`
8. `content/03_ssot.tex`
9. `content/12_rate_corollaries.tex`
10. `content/04_requirements.tex`
11. `content/05_evaluation.tex`
12. `content/07_empirical.tex`
13. `content/09_conclusion.tex`
14. `content/claim_mapping_auto.tex`

This is intentionally the **PDF/include order**, not the numeric filename order.

## Suggested Immediate Fix Set

If time is limited, the highest-value near-term fixes are:
1. Fix the section-order / transition mismatch
2. Fix the broken `thm:ssot-iff` reference
3. Tighten the hottest equality sentence in `content/10_equality.tex`
4. Collapse or relocate the ultra-thin `content/05_evaluation.tex` and `content/07_empirical.tex` material
5. Fix unmapped claims and tighten broad Lean-handle citations
