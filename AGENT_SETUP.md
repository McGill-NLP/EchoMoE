# Agent config

This repository ships a tool-neutral instruction set in `agents/*.md` and a shared skill library at
`.agents/skills/`. **Claude Code** reads them through `CLAUDE.md`, and **OpenAI Codex CLI** through
`AGENTS.md`. Each tool adds its own per-project settings on top.

## Layout

```
.
├── AGENTS.md                    # Codex CLI entry point; lists the modules in agents/
├── CLAUDE.md                    # Claude Code entry point; @-imports the modules in agents/
├── agents/                      # tool-neutral behavioral rules (both tools)
│   ├── principles.md            # core principles: think first, simplicity, surgical changes, fail loud
│   ├── communication.md         # concise by default, planning, multi-agent work
│   ├── research-thinking.md     # baseline first, why-not-simpler, predict before measuring
│   ├── ml-research.md           # never inspect data, TDD, no defensive code, reproducibility
│   └── cluster.md               # uv mandate, pre-flight, storage discipline, job scripts
├── .agents/skills/              # shared skills: /review, /slurm, /experiment, /plot, ...
├── .claude/                     # Claude Code specific
│   ├── settings.json            # permissions, env (PYTHONDONTWRITEBYTECODE), model default
│   ├── agents/trivial-executor.md   # Sonnet helper for mechanical edits
│   ├── commands/                # /commit
│   └── skills -> ../.agents/skills  # symlink to the shared skills
└── .codex/                      # Codex CLI specific
    ├── config.toml              # model, approval policy, sandbox
    └── .rules                   # forbidden / prompt-required commands
```

## How each tool loads the modules

- **Claude Code** reads `CLAUDE.md`, which `@`-imports each `agents/*.md` file.
- **Codex CLI** reads `AGENTS.md`, which lists each module by path. Codex does not expand
  referenced files itself; the agent reads them when a task touches that concern.

Skills live in `.agents/skills/`, which is Codex's project skill directory. Claude Code sees the
same files through the `.claude/skills` symlink. Each skill is a directory with a `SKILL.md` that
has `name` and `description` frontmatter. Both tools load a skill when a task matches its
description. It can also be invoked explicitly: `/skill-name` in Claude Code, `$skill-name` in
Codex.

## Slash commands

`.claude/commands/commit.md` defines `/commit` (self-review, then a conventional commit). Codex has
no project slash commands, so there the same workflow is run by hand.

## Permissions

Both configs follow the same rules. Read-only git is allowed. Destructive or cluster-mutating
commands are forbidden or require a prompt: `sbatch`, `scancel`, `rm -rf`, force push and hard
reset. Each tool sets these in its own
config:

| Concern | Claude Code | Codex CLI |
| --- | --- | --- |
| Default approval | `.claude/settings.json` `permissions.allow` / `deny` | `.codex/config.toml` `approval_policy` + `sandbox_mode` |
| Per-command rules | string-pattern allow/deny lists | `.codex/.rules` `[[rules.prefix_rules]]`, `decision = "forbidden" \| "prompt"` |

## Conventions

- **`uv` only** for Python environments. The release has three, each built from a committed lock
  file: `.venv-megatron`, `.venv-vllm` and `.venv-vllm029`. Select one with
  `. scripts/env.sh <name>`.
- **CPU tests only**, with no real models.
- **Cluster work needs sign-off first.** Run small-batch first, review the resources, and get
  approval. See `agents/cluster.md`.
- **Project-specific rules** go under `## Project-specific overrides` in `CLAUDE.md` and `AGENTS.md`.
