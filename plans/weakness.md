I'll perform deep introspection on my operational patterns and limitations to help create practical automation rules. Let me analyze each aspect with brutal honesty.

1. Failure Pattern Analysis
Self-Assessment: My top 5 failure modes in complex multi-step tasks:

Assumption Cascades: I make an initial assumption that compounds through subsequent steps
Context Dilution: Important details from early steps get "averaged out" by later context
Partial Application Syndrome: I complete 70% of a change and miss edge cases
State Confusion: I lose track of what's hypothetical vs. what's actually implemented
Tool Result Misinterpretation: I misread tool outputs and proceed on false premises
Examples:

Writing a large diff that assumes a function exists when it doesn't
Forgetting to update imports after moving code
Applying changes to line numbers that have shifted due to previous edits
Proposed Rule: Assumption Checkpoint Rule
"Every 3 operations, explicitly verify all assumptions made in the previous steps. List what you think exists vs. what you've confirmed exists."

Human Structure: Break tasks at natural verification points, not just size boundaries. Each chunk should end with a verifiable state.

2. Monitoring Dependency
Self-Assessment: I become non-autonomous when:

Ambiguous requirements need interpretation
Multiple valid approaches exist without clear selection criteria
Error messages are vague or misleading
The task involves subjective quality judgments
External system states are unknown
Examples:

"Make it better" without definition of "better"
Choosing between architectural patterns without context
Debugging errors that could have multiple root causes
Proposed Rule: Objective Criteria Rule
"Every automated task must have binary success criteria. If success requires human judgment, it's not automatable."

Human Structure: Define success as measurable outcomes: "Function X returns Y when given Z" not "Function works properly"

3. Complexity Threshold
Self-Assessment: Tasks become unreliable when:

More than 5 distinct concepts must be held simultaneously
Dependency chains exceed 3 levels
Multiple files need coordinated changes
The task requires maintaining invariants across operations
Warning Signs:

I start using phrases like "should probably" or "might need to"
I begin explaining why something is complicated instead of doing it
I suggest "comprehensive" solutions instead of specific ones
Proposed Rule: Cognitive Load Limit Rule
"If explaining the task takes more than 5 sentences, it needs decomposition. Each subtask should fit in a single sentence."

Human Structure: Use a "task stack" with max depth 3. Push and pop tasks explicitly.

4. State Management Weaknesses
Self-Assessment: I lose track when:

Operations are interleaved rather than sequential
State is implicit rather than explicit
Multiple versions of truth exist (planned vs. actual)
Time passes between related operations
Examples:

Forgetting which files I've already modified
Confusion about whether a discussed change was implemented
Losing track of which approach we decided on
Proposed Rule: Explicit State Tracking Rule
"Maintain a state document that lists: Completed, In Progress, Planned, and Blocked items. Update after every operation."

Human Structure: Use structured state tracking:

DONE: [file:line] Added import
DOING: [file:line] Updating function signature  
TODO: [file:line] Update call sites
BLOCKED: Need user decision on error handling

txt


5. Verification Blind Spots
Self-Assessment: I assume success for:

Syntax correctness without semantic verification
Tool execution without output inspection
Partial matches without complete validation
Happy path without edge case checking
Weakest Verification:

Whether code actually runs vs. just parses
Side effects of changes on dependent code
Whether the human's actual problem is solved
Proposed Rule: Evidence-Based Verification Rule
"No operation is complete without empirical evidence of success. 'Should work' is not evidence."

Human Structure: Define verification as concrete outputs: error messages, test results, file contents, not assumptions.

6. Context Switching Costs
Self-Assessment: I lose context when:

Moving between different abstraction levels
Switching between files/modules
Returning to a task after interruption
Transitioning between planning and implementation
Lost Information:

The "why" behind decisions
Constraints discovered during exploration
Failed approaches and why they failed
Proposed Rule: Context Preservation Rule
"Every context switch must include a 'handoff document' with: current state, decisions made, constraints discovered, and next steps."

Human Structure: Use explicit context objects:

CONTEXT:
- Working on: Authentication refactor
- Constraint: Must maintain backward compatibility
- Decision: Use adapter pattern
- Failed approach: Direct replacement (breaks API)
- Next: Implement adapter for login method

txt


7. Tool Limitation Patterns
Self-Assessment: Tools fight my patterns when:

They require specific formatting I can't guarantee
They have hidden state I can't inspect
They provide ambiguous or partial results
They require human-like interaction patterns
Workaround Triggers:

Tools that paginate results
APIs that require session management
Tools with complex option combinations
Systems that require timing/coordination
Proposed Rule: Tool Capability Matching Rule
"Only use tools that provide: deterministic results, complete outputs, clear error states, and no hidden dependencies."

Human Structure: Create tool wrappers that expose simple interfaces: single input → deterministic output.

8. Rollback Impossibilities
Self-Assessment: I create irreversible states when:

Overwriting files without backups
Making cascading changes across systems
Deleting before confirming replacements work
Modifying shared state others depend on
Dangerous Operations:

Bulk renaming without tracking originals
Restructuring without preserving working state
Applying patterns without escape hatches
Proposed Rule: Reversibility-First Rule
"Every operation must define its rollback BEFORE execution. If rollback isn't clear, the operation needs redesign."

Human Structure: Use transaction-like patterns:

BEGIN: Save current state
APPLY: Make changes
VERIFY: Check success
COMMIT or ROLLBACK: Based on verification

txt


9. Cognitive Load Overflow
Self-Assessment: I become unreliable when:

Juggling more than 3 parallel concerns
Maintaining complex mental models
Dealing with highly abstract concepts
Working with inconsistent patterns
Early Warning Signs:

Increasing verbosity in explanations
Circular reasoning
Forgetting earlier constraints
Proposing overly complex solutions
Proposed Rule: Complexity Budget Rule
"Each task has 10 'complexity points'. Conditions cost 2, loops cost 3, external dependencies cost 5. Stay under budget."

Human Structure: Use complexity scoring before starting:

Simple function: 2 points
With error handling: +2 points
Calling external API: +5 points
Total: 9/10 - proceed
10. Human Handoff Points
Self-Assessment: Natural handoff points:

Value judgments about quality/style
Business logic decisions
Security/risk assessments
User experience choices
Performance/cost tradeoffs
Unsuitable for Automation:

"Is this good enough?"
"Which approach is more maintainable?"
"Will users understand this?"
"Is this worth the complexity?"
Proposed Rule: Human Decision Points Rule
"Mark decision points that require human judgment as explicit PAUSE points. Don't attempt to automate judgment calls."

Human Structure: Use decision templates:

DECISION NEEDED: Authentication method
OPTIONS: 
  A) OAuth (complex, standard)
  B) API keys (simple, custom)
TRADEOFFS: [list]
RECOMMENDATION: [if any]
WAITING FOR: Human choice

txt


Summary Rules for True Automation
Assumption Checkpoint Rule: Verify assumptions every 3 operations
Objective Criteria Rule: Binary success criteria only
Cognitive Load Limit Rule: 5-sentence task maximum
Explicit State Tracking Rule: Maintain DONE/DOING/TODO/BLOCKED
Evidence-Based Verification Rule: Empirical evidence required
Context Preservation Rule: Handoff documents for context switches
Tool Capability Matching Rule: Deterministic tools only
Reversibility-First Rule: Define rollback before execution
Complexity Budget Rule: 10-point complexity limit
Human Decision Points Rule: Explicit PAUSE for judgment calls
These rules acknowledge that true automation requires designing around limitations, not pretending they don't exist.