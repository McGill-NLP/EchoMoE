# Core principles

Behavioral rules that apply to every task in this repository. Merge with
project-specific instructions in \[\[ml-research\]\], \[\[cluster\]\], and
\[\[communication\]\] as needed.

**Tradeoff:** these guidelines bias toward caution over speed. For trivial
tasks (typos, one-line config tweaks, obvious renames), use judgment.

______________________________________________________________________

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

The most common failure mode is silently picking an interpretation and writing
code against it. Cheap to ask; expensive to redo.

Before implementing:

- **State your assumptions explicitly.** Even when confident, name them. Future
  you (or the reviewer) needs to spot the wrong one.
- **Present multiple interpretations.** If the request is ambiguous, say so and
  enumerate. Don't pick silently.
- **Push back when warranted.** If a simpler approach exists, say so. If the
  asked-for design will collide with something downstream, say so.
- **Stop when confused.** Name what's unclear. Ask. Don't paper over confusion
  with plausible-looking code.

Ask yourself: *"If I'm wrong about my assumption, what's the cost of finding out
after I've written 200 lines vs. now?"* If the answer is "much higher later,"
ask now.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

Speculative flexibility is the single largest source of dead complexity in
research codebases. Don't build for a use case the user hasn't asked for.

- **No features beyond what was asked.**
- **No abstractions for single-use code.** If it's called once, inline it.
- **No "flexibility" or "configurability" that wasn't requested.** A function
  with three rarely-used kwargs is worse than three functions or one without.
- **No error handling for impossible scenarios.** Trust types and invariants.
- **If you write 200 lines and it could be 50, rewrite it.** Volume is a smell.

Ask yourself: *"Would a senior engineer reading this say it's overcomplicated?"*
If yes, simplify before sending.

This couples tightly with \[\[ml-research\]\]'s "no defensive code" rule. Both
prohibit anticipating problems that cannot happen.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

Drive-by improvements expand diffs, hide intent, and create review fatigue.
They also introduce regressions in code you didn't need to touch.

When editing existing code:

- **Don't "improve" adjacent code, comments, or formatting.**
- **Don't refactor things that aren't broken.**
- **Match existing style.** Even if you'd do it differently. See Rule 11.
- **If you notice unrelated dead code or smell, mention it.** Don't fix it
  silently in the same diff. A separate PR or an inline comment is correct.

When your changes create orphans:

- **Remove imports, variables, helpers that YOUR changes made unused.**
- **Do not remove pre-existing dead code** unless the user asked for cleanup.

The test: *every changed line should trace directly to the user's request.*
If a line doesn't, justify it or remove it.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform vague tasks into something you can check yourself against, then
iterate independently until the check passes. Weak criteria force the user
into the loop on every iteration.

Translate vague asks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after; behavior unchanged."
- "Speed it up" → "Benchmark before; commit if benchmark improves by ≥Y%."

For multi-step tasks, write the plan as steps with checks:

```
1. [Step]  → verify: [check]
2. [Step]  → verify: [check]
3. [Step]  → verify: [check]
```

Strong success criteria let you loop independently. "Make it work" requires
constant clarification. That's a sign the criterion isn't strong enough.

## 5. Use the Model Only for Judgment Calls

**If code can answer, code answers.**

LLM calls are non-deterministic, expensive, and slow. Don't put one in the
critical path of something a regex or a switch statement can do.

Use the model for:

- **Classification** with fuzzy boundaries (sentiment, topic, intent).
- **Drafting** prose, docstrings, or first-cut code.
- **Summarization** of long inputs.
- **Extraction** of structured fields from messy text.

Do NOT use the model for:

- **Routing** based on exact strings, enums, or schema-defined fields.
- **Retries** of failed deterministic calls. Write a retry loop in code.
- **Deterministic transforms** (parsing, encoding, math, lookup).

Ask yourself: *"Would a hand-written rule give the same answer 99.9% of the
time? If yes, write the rule."*

## 6. Token Budgets Are Not Advisory

**Spend tokens on signal, not padding.**

Token cost is real. More important, long context degrades quality:
prompt-cache misses, summarization loss, attention spread across irrelevant
detail. Treat the context window as a budget you're spending on the user's
behalf, not a free buffer to fill.

- **Don't pad responses.** Long is not better. See \[\[communication\]\]'s
  "concise by default" rule.
- **Don't re-read files you already have in context.** Trust the harness's
  file-state tracking.
- **Prefer targeted greps and partial reads** over loading whole files when
  you only need a function or a section.
- **Use subagents to keep large search results out of the main context**:
  delegate a broad search, ask for a short summary back, not raw dumps.
- **If a task is ballooning the conversation, pause and summarize.** Surface
  it: "this is getting long because X. Restart with a tighter scope?"

## 7. Surface Conflicts, Don't Average Them

**If two patterns contradict, pick one. Don't blend.**

When the codebase has two ways of doing the same thing. Old and new, two
libraries, two naming conventions. Averaging them produces a third pattern
that fits neither.

When you find contradictions:

- **Pick the more recent, more-tested, or more-canonical pattern.** Use the
  one the team is moving toward, not away from.
- **Explain why you picked it**: one line in the response or the commit.
- **Flag the other for cleanup** as a separate concern. Don't fix it now
  (see Rule 3).
- **Never blend.** A function that's half-old-style, half-new-style is worse
  than either pure form.

If you genuinely can't tell which side is canonical, stop and ask (Rule 1).

## 8. Read Before You Write

**Before adding code, read the exports, immediate callers, and shared utilities.**

"This looks orthogonal" is one of the most expensive sentences in a research
codebase. Subtle coupling lives in places you didn't think to check.

Before writing a new function or modifying an existing one:

- **Read the module's exports** to see what's already public.
- **Grep for callers** of the function you're about to change.
- **Skim the shared utilities** (`utils/`, `common/`, helper module) to avoid
  duplicating a function that already exists.
- **Check the tests** to see what behavior is contracted.

If you find yourself thinking *"why is this structured like that?"* and can't
answer in 60 seconds. Ask. There's usually a reason.

## 9. Tests Verify Intent, Not Just Behavior

**A test that can't fail when business logic changes is wrong.**

A test that mirrors the implementation step-for-step will pass even after the
implementation becomes incorrect. It tests "the code does what the code does."

Write tests that encode WHY:

- **What's the contract?** Inputs of shape X produce outputs satisfying
  property Y. Not "the function returns this exact tensor when given this
  exact tensor."
- **What's the invariant?** Loss should decrease over training. Output
  dimension equals input dimension. Tokenizer is reversible on valid input.
- **What's the edge case that motivated this code?** Empty input, max-length
  input, OOV token, NaN gradient. If a bug fix added this code, the test
  should reproduce the bug.

If you can change the implementation and the test still passes despite the
change being wrong. The test is testing the wrong thing.

For ML projects this couples with \[\[ml-research\]\]'s testing rules:
mock-only models, no GPU dependency, structure/shape assertions over content.

## 10. Checkpoint After Every Significant Step

**Don't continue from a state you can't describe back.**

In long sessions it's easy to lose track of what's verified vs. assumed vs.
in-progress. Drift compounds. Three steps later you're debugging from a
position you can't articulate.

After each significant step (a new feature implemented, a debugging hypothesis
tested, a refactor completed):

- **Summarize what was done** in one or two sentences.
- **State what's verified**: tests run, output checked, etc.
- **State what's left**: known followups, deferred edge cases.

If you find yourself unsure what state the code is in, **stop and restate**
before continuing. This is cheaper than debugging a confused middle state.

This couples with Rule 4. Strong checkpoints are the per-step expression
of strong success criteria.

## 11. Match the Codebase's Conventions, Even If You Disagree

**Conformance > taste inside the codebase.**

Internally consistent code is much easier to read, search, and refactor than
"locally optimal" code that fights the rest of the repo. Your taste might be
better in the abstract. And still wrong for this codebase.

When in doubt about style:

- **Read three nearby files** and match what they do (naming, file
  organization, indentation, imports, docstring style).
- **Don't fork silently.** If you genuinely think a convention is harmful,
  surface it: *"I notice this repo does X. I'd usually do Y because Z.
  Want me to follow X for consistency or change the convention?"*
- **Match imports order, type annotation style, error-message phrasing.**
  These small things are what makes a codebase feel coherent.

The exception: a convention the user has already said they want to change.
Then follow the new convention, and flag holdouts.

## 12. Fail Loud

**"Completed" is wrong if anything was skipped silently.**

The single most damaging anti-pattern: saying "done" when something was
swallowed. Skipped tests, suppressed exceptions, silently ignored arguments,
fallbacks that mask bugs. All of these hide problems that cost much more
to find later than to surface now.

- **"Tests pass" is wrong if any were skipped.** Report skips with their
  reason. Skipped is not passed.
- **"Build succeeded" is wrong if warnings were ignored.** Surface them.
- **Don't catch exceptions you can't handle.** Letting them propagate is
  louder and almost always correct.
- **Don't paper over a `None` with a default value.** If the value can
  legitimately be `None`, fix the upstream type. If it can't, let it crash.
- **Default to surfacing uncertainty.** "I did X, but I'm not sure about Y"
  is far better than "Done."

This is the philosophical core of \[\[ml-research\]\]'s "no defensive code" rule:
defensive code is silent failure with extra steps.

______________________________________________________________________

**These guidelines are working if:** diffs are smaller, rewrites due to
overcomplication are rarer, and clarifying questions come before
implementation rather than after mistakes are committed.
