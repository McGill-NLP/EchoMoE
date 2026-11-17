# CLAUDE.md

Behavioral guidelines for this research repository. Composed from modular
files in `agents/`. Each module covers one concern and can be swapped
per-project.

The `agents/*.md` modules are tool-agnostic and shared with Codex CLI via
`AGENTS.md` at the repo root. This file (`CLAUDE.md`) adds Claude Code
specific surfaces (skills, slash commands) on top. See
[`AGENT_SETUP.md`](./AGENT_SETUP.md) for the cross-tool layout.

## Modules

@agents/principles.md
@agents/communication.md
@agents/research-thinking.md
@agents/ml-research.md
@agents/cluster.md

## Custom commands

The `.claude/commands/` directory provides slash commands:

- `/commit`. Run pre-commit hooks, code-review subagent, then create a
  conventional commit. Use this rather than manual `git commit` when you
  want the review pass; not mandatory.

## Skills

The `.claude/skills/` directory ships research-specific skills with the repo.
no install step. Each skill auto-loads when Claude detects a matching task.

Available skills:

| Command          | When to use                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| `/review`        | Code review: bugs, style, correctness                                            |
| `/latex`         | Writing, editing, or debugging LaTeX                                             |
| `/bash`          | Writing or debugging shell scripts                                               |
| `/slurm`         | Job scripts, resource allocation, debugging failed jobs                          |
| `/pytorch-debug` | Dtype, device, shape, gradient, OOM errors                                       |
| `/jax-debug`     | Tracing, jit/vmap/pmap, PRNG keys, NaN/Inf, recompiles, OOM                      |
| `/distributed`   | DDP/FSDP/DeepSpeed/torchrun, NCCL hangs, multi-node SLURM                        |
| `/experiment`    | Planning, scaffolding, launching, profiling, evaluating, sweeping ML experiments |
| `/git`           | Branching, rebasing, cherry-picking, conflicts                                   |
| `/paper`         | Discovering literature, finding implementations, summarizing papers              |
| `/write`         | Drafting paper prose: abstract, intro, related work, methods, results, captions  |
| `/plot`          | Paper-quality matplotlib / seaborn figures                                       |

**Proactively invoke these** when the task matches. Don't wait for the user
to type the command explicitly. Match the skill to the context (debugging
a tensor error → `/pytorch-debug`; writing a figure → `/plot`).

## Model tiers

Default session model is **Opus** (`model: opus` in `.claude/settings.json`) --
the workhorse for execution. The bounded helper in `.claude/agents/` uses
Sonnet for mechanical work:

| Agent              | Model  | Use for                                                                  |
| ------------------ | ------ | ------------------------------------------------------------------------ |
| `trivial-executor` | Sonnet | Typos, one-line tweaks, running a known command, simple mechanical edits |

Route only trivial mechanical work to `trivial-executor`; keep design,
debugging, evaluation analysis, and implementation on the Opus main loop.
There is no auto-router.

## Complementary plugin: research-companion

[`research-companion`](https://github.com/andrehuang/research-companion)
handles upstream research-direction work this template doesn't cover.

- *"What should I work on? Is this idea worth pursuing? Should I pivot?"*
  use `research-companion:research-companion` (strategy / triage).
- *"How do I find papers / set up / run / debug / plot this?"*
  use this template's skills (execution).
