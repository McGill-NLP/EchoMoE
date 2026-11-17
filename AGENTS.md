# AGENTS.md

Entry point for OpenAI Codex CLI and opencode CLI (and any other agent
that respects the `AGENTS.md` convention). The behavioral rules below are
tool-neutral and shared with Claude Code, which loads them through
`CLAUDE.md`.

Codex walks the directory hierarchy and concatenates every `AGENTS.md` it
finds along the path (global `~/.codex/AGENTS.md` first, then repo root,
then subdirectories). The combined prompt is capped at 32 KiB by default
(`project_doc_max_bytes`); the modules below total well under that.

## Read these first

Before responding, load the modules below. Each one covers one concern and
applies to every task in this repository. Codex does not expand file
references automatically; treat each filename as an instruction to open
and read that file before doing work that involves it.

1. [`agents/principles.md`](./agents/principles.md). 12 core principles:
   think before coding, simplicity first, surgical changes, goal-driven
   execution, model only for judgment, token discipline, flag conflicts,
   read before write, tests verify intent, checkpoint often, match
   conventions, fail loud.
2. [`agents/communication.md`](./agents/communication.md). Concise by
   default, top-down planning, when to ask clarifying questions, multi-agent
   task decomposition, debugging by data-flow trace.
3. [`agents/research-thinking.md`](./agents/research-thinking.md). Start
   from the trivial baseline, force a why-not-simpler step, predict before
   you measure, push back on framing, state what would change your mind,
   brainstorm vs decide, dead ends are evidence too.
4. [`agents/ml-research.md`](./agents/ml-research.md). Never inspect data,
   TDD discipline,
   self-verification (tests + lint + types), no defensive code, comment
   discipline, reproducibility (seeds, env capture, provenance logging),
   notes & research discussions.
5. [`agents/cluster.md`](./agents/cluster.md). `uv` mandate for Python
   environments, pre-flight checks before `sbatch`, storage discipline
   (`$HOME` vs `$SCRATCH` vs `$SLURM_TMPDIR`), job script structure, time
   limits.

When the rules cross-reference `[[principles]]`, `[[ml-research]]`, etc.,
that means the corresponding `agents/<name>.md` file.

## Skills

This project ships skills at `.agents/skills/` (Codex's canonical
project-scoped skill directory; also discovered by opencode). Each skill
has a `SKILL.md` with `name` and `description` frontmatter; both Codex
and opencode can implicitly select a skill when your task matches the
description. Codex invokes explicitly with `$skill-name` or via
`/skills`; opencode uses `/skill-name`.

Skill index (matched by description, not the slash name):

| Skill           | When to use                                                                       |
| --------------- | --------------------------------------------------------------------------------- |
| `review`        | Code review: bugs, style, correctness.                                            |
| `latex`         | Writing, editing, or debugging LaTeX.                                             |
| `bash`          | Writing or debugging shell scripts.                                               |
| `slurm`         | Job scripts, resource allocation, debugging failed jobs.                          |
| `pytorch-debug` | Dtype, device, shape, gradient, OOM errors.                                       |
| `jax-debug`     | Tracing, jit/vmap/pmap, PRNG keys, NaN/Inf, recompiles, OOM.                      |
| `distributed`   | DDP/FSDP/DeepSpeed/torchrun, NCCL hangs, multi-node SLURM.                        |
| `experiment`    | Planning, scaffolding, launching, profiling, evaluating, sweeping ML experiments. |
| `git`           | Branching, rebasing, cherry-picking, conflicts.                                   |
| `paper`         | Discovering literature, finding implementations, summarizing papers.              |
| `write`         | Drafting paper prose: abstract, intro, related work, methods, results, captions.  |
| `plot`          | Paper-quality matplotlib / seaborn figures.                                       |

The same files are exposed to Claude Code via a symlink at
`.claude/skills/`, so edits in one place propagate to both tools.

## Slash commands

opencode reads `.claude/commands/` via the
`.opencode/commands -> ../.claude/commands` symlink, so `/commit` works
natively. Codex CLI does not currently support project-
defined custom slash commands; the workflows still apply, run them
step-by-step:

- **`/commit` equivalent**: run pre-commit hooks (`pre-commit run --all-files`),
  do a self-review pass, then write a conventional-style commit message.
  See `.claude/commands/commit.md` for the recipe.

## Tool-specific config

- **Codex CLI.** `.codex/config.toml` sets the approval policy, sandbox
  mode, model, and environment defaults. `.codex/.rules` lists commands
  that are forbidden or require explicit user approval (`sbatch`,
  `scancel`, `rm -rf`, force push, etc.).
- **opencode.** `.opencode/opencode.jsonc` sets the model and a
  `permission.bash` glob map mirroring the Codex deny posture
  (`"allow" | "ask" | "deny"`, last-match-wins).

Read the relevant config before deviating.
