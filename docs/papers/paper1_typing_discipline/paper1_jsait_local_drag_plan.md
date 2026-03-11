# Paper 1 Local Drag Plan

This note isolates places in the current JSAIT draft where the paper risks feeling like
"I am now in a different paper for a page." The goal is not to cut the mathematics. The goal is
to keep the content while reducing tonal and structural whiplash.

## Main principle

Each flagged passage should be revised by one of the following moves:

1. **Compress** repeated explanatory material while keeping the theorem statement.
2. **Reclassify** the passage as a consequence of the main residual-ambiguity story.
3. **Merge** nearby statements that currently read like separate mini-results.
4. **Reorder** so the reader sees the coding or representation-sufficiency relevance before the formal detour.

The standard to aim for is:

- readers should feel they are still inside the same paper's spine,
- not like they opened a side note on computability, PL semantics, or query design.

## 1. `latex_jsait/content/02_compression_framework.tex`

### Why it drags

The opening is now better, but the section still slows down after the coding-model setup because it stacks too much framework material before returning to the main theorem arc.

### Most jarring sub-block

The tail after the information barrier:

- `cor:provenance-barrier`
- quotient viewpoint remark
- fixed-axis completeness
- fixed-axis incompleteness
- observational quotient
- adding-information corollary

This cluster reads like a compact framework paper on observation/quotients/axes rather than like the minimal model section of a compression paper.

### Keep

- coding model induced by a representation
- notation bridge between `(C,U)` and `(V,C(V),U)`
- admissibility contract
- information barrier theorem
- class-identity corollary

### Likely changes

- compress the post-barrier cluster into one short paragraph plus at most one representative theorem/corollary;
- explicitly mark the axis-specialized restatements as supporting secondary sections only;
- remove any sentence that repeats "equal representation implies indistinguishability" in a new dialect without adding downstream value.

### Target feeling after revision

"This section gave me the minimal model and the one barrier theorem I need."

Not:

"Now I am reading a framework note about observational quotients and axis completeness."

## 2. `latex_jsait/content/03_matroid_structure.tex`

### Why it drags

This section is now conceptually tied to Arc 2, which is good. But once it moves into the unrestricted counterexample / structured-axis distinction / matroid remarks, it still feels denser and more self-contained than the surrounding paper.

### Most jarring sub-block

The structured-axis subsection:

- unrestricted non-equicardinality counterexample
- structured axis model definition
- coherence remark
- restricted matroid theorem
- relation-to-general-invariant remark
- VC-style dimension remark

All of this is individually reasonable, but together it still feels like a mini-paper on query structure.

### Keep

- attribute-only lower bound theorem
- minimum distinguishing number as the query-side analogue of residual coding burden
- restricted matroid theorem (since it is real and mechanized)

### Likely changes

- shorten the unrestricted-vs-structured setup language;
- keep the counterexample, but in one compact sentence instead of a dramatic pivot;
- keep only one of the two current interpretive remarks after the matroid theorem unless both are doing real work;
- make the closing comparison table do more narrative work so the section lands back in the main story.

### Target feeling after revision

"This section gives the operational cost of restoring representation sufficiency through primitive observables."

Not:

"Now I am reading a separate paper on matroids and structured query families."

## 3. `latex_jsait/content/04a_complexity_bounds.tex`

### Why it drags

The opening is much better than before, but the Rice-style block still feels like a clean little undecidability note nested inside the paper.

### Most jarring sub-block

- function-level barrier predicate definition
- Rice-style non-computability theorem
- proof
- certification interpretation remark
- final corollary

This is exactly the kind of thing that makes a reader feel they have entered a different theorem culture for a page.

### Keep

- open-world instability theorem
- Rice-style non-computability theorem

### Likely changes

- compress the proof and commentary around the Rice theorem;
- possibly merge the certification interpretation and final corollary into one tighter paragraph;
- keep explicit connection to residual ambiguity under world growth in the very first and very last sentence of the section.

### Target feeling after revision

"This is a robustness consequence of the main ambiguity story."

Not:

"Now I am reading a separate computability paper."

## 4. `latex_jsait/content/06b_extensions.tex`

### Why it drags

This section does not derail the paper the way Sections 2--4 can, but it can still feel like a generic future-work appendix if the prose becomes too broad.

### Most jarring sub-block

Any extension paragraph that starts sounding like a different research agenda:

- full noisy coding theory
- full lossy semantic theory
- full privacy formalism
- practical audit methods for large learned models

### Keep

- noisy/lossy/privacy outlooks
- learned-representation audit paragraph
- rate-distortion-perception positioning

### Likely changes

- keep each extension tied to the same primitive: residual ambiguity left by the representation;
- avoid long method-style lists unless they directly answer a reviewer concern;
- keep this section visibly subordinate to the theoremic body.

### Target feeling after revision

"These are disciplined next steps for the same theory."

Not:

"Now I am reading a survey of three future papers."

## 5. `latex_jsait/content/06_applications.tex`

### Why it mostly works now

This section used to be one of the biggest sources of drag. It is now much stronger because it contains labeled results rather than only interpretation.

### Residual risk

The nominal/structural and neuro-symbolic subsection can still tip into a different-paper feeling if the prose becomes too grand or too cross-disciplinary.

### Keep

- task sufficiency theorem family
- helper-view results
- factorized/product law
- learned-representation interpretation

### Likely changes if needed

- keep the nominal/structural paragraph short and derivative of the theoremic arc;
- avoid letting the neuro-symbolic discussion become a new manifesto;
- make sure every interpretive paragraph still points back to a theorem just stated above it.

### Target feeling after revision

"This is the paper's second theoremic arc."

Not:

"Now I am reading a conceptual applications essay."

## Priority order for de-dragging

If doing one more tightening pass, the order should be:

1. `02_compression_framework.tex`
2. `03_matroid_structure.tex`
3. `04a_complexity_bounds.tex`
4. `06b_extensions.tex`
5. only then `06_applications.tex`, and only if it starts to sound too grand

## Standard for success

After the pass, a reader should experience the paper as:

1. exact residual-coding laws,
2. exact representation-sufficiency laws,
3. then query and robustness as controlled consequences,
4. with no point where the paper feels like it has temporarily become a different discipline's note.
