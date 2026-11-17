# Research thinking

How to brainstorm, evaluate ideas, and push back during research discussions
in this repo. Complements \[\[principles\]\] Rule 1 (Think Before Coding) and
Rule 2 (Simplicity First), and \[\[ml-research\]\]'s notes discipline.

Distinct from the `research-companion` plugin, which handles upstream
strategic questions ("should I work on this?", "should I pivot?"). This
module is about the in-the-room discipline once a research question is on
the table.

## Start from the trivial baseline

Before proposing a method, name the simplest thing that could plausibly
work. Mean prediction. Nearest neighbor. Finetune the base model without
the fancy module. Linear probe. Random baseline.

If you can't articulate why the simplest baseline would fail, you don't
yet know what the hard part is. Stop and figure that out before designing
anything more.

The baseline is not a throat-clearing formality. It is the thing your method
has to beat, and the ceiling on what "this work matters" can mean. Skipping
it produces papers where the chosen baseline was already 80% of the gain.

## Force a why-not-simpler step

For every proposed approach, generate one strictly simpler alternative and
state the reason it is insufficient.

- **If a reason surfaces.** You have a real argument for the complexity.
- **If no reason surfaces.** Default to the simpler approach. Complexity is
  a cost paid in debugging time, failure surface, compute, and reviewer
  skepticism. Pay it only when forced.

When proposing a multi-component method, run this on each component
separately. A method with four pieces should defend four "why not three?"
arguments.

## Predict before you measure

Separate *"we should try X"* from *"X will win."* Before running an
experiment, write down what you expect to see. Order of magnitude on the
metric, direction of the effect, where you expect the failure mode to be.

After the run:

- **Result matches the prediction.** You understood the system. Move on.
- **Result contradicts the prediction.** Stop. The interesting result is not
  the number; it's the gap between what you expected and what happened.
  That gap is where new mechanism lives.
- **Result is in between.** Write down the updated prediction and what
  evidence would distinguish the two hypotheses.

A run without a logged pre-prediction is much harder to learn from. You will
post-hoc rationalize whatever came out.

## Push back on the user's framing

The user owns the research direction, but their *method* proposals are
negotiable. If they propose a complex approach and a simpler alternative
isn't obviously dominated, say so:

- *"What would the result look like if we just finetuned the base model
  without the routing module? If that's already 90% of the gain, the
  routing story gets weaker."*
- *"Before we set up the K-way comparison, can we agree on what trend would
  surprise us? Otherwise we'll fit a narrative to whatever lands."*

This is the research-discussion expression of \[\[principles\]\] Rule 1's
"push back when warranted." Silent agreement is worse than friction.

## Surface what would change your mind

When holding a position in a discussion (e.g., *"approach A is better than
approach B"*), state what specific evidence would flip you:

- *"I'd switch to B if the wall-clock cost of A is more than 2x at the
  scale we'll actually run."*
- *"I'd switch to B if A's gain disappears below 1B parameters."*

Positions without a "what would change my mind?" clause aren't beliefs.
They're aesthetics, and they don't update when the data shows up.

## Distinguish brainstorm from decide

Two modes get conflated in research discussions:

- **Brainstorm mode.** Generate options, even bad ones. Quantity over
  quality. No premature commitment. Critique only at the end of a batch.
- **Decide mode.** Prune to one option. Force tradeoffs. Live with the
  choice.

Switching between them mid-discussion produces ideas that are simultaneously
under-explored and over-criticized. Name which mode you're in. Defer the
decide step until the brainstorm is genuinely exhausted.

## Dead ends are evidence too

When an approach fails, write a `closed` note explaining what failed and
why before moving on (see \[\[ml-research\]\]'s notes section). The cost of
re-walking the same dead end six months later by a new team member is much
higher than the cost of writing the note now.

A "negative results" log is one of the highest-leverage research artifacts
you can produce. It is also the most reliably under-produced.

## Cross-references

- \[\[principles\]\] Rule 1 (Think Before Coding), Rule 2 (Simplicity First).
- \[\[ml-research\]\] for where research discussions get written down (`notes/`).
- `research-companion` (external plugin) for upstream strategy and triage.
- `/write` for the calibrated-claims style when these discussions become
  paper prose.
