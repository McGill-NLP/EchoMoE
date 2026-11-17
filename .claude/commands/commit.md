---
description: Run pre-commit checks, code review, and create a git commit
allowed-tools: Bash(git:*), Bash(pre-commit:*), Bash(ruff:*), Bash(pyright:*)
---

## Context

- Git status: !`git status`
- Git diff (staged + unstaged): !`git diff HEAD`
- Branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -5`

## Instructions

### Step 1: Pre-commit checks

Run `pre-commit run --all-files`. If it fails, fix issues and re-run until clean.

### Step 2: Code review

Spawn a Task subagent (general-purpose) as a strict code reviewer.
The reviewer receives the full diff from context above and evaluates:

- Is there unnecessary code being committed?
- Could this code be simpler? Could any abstraction, helper, or indirection be removed?
- Is there defensive code? (`or ""`, `or []`, `if x is not None` when x cannot be None, try/except around code that cannot fail, fallback defaults that mask bugs)
- Is there anything here that doesn't belong (debug prints, .env files, credentials)?
- Are there unnecessary comments, docstrings on obvious code, or over-engineering?

The reviewer is STRICT. Every suggestion must be addressed. You are NOT allowed
to ignore reviewer feedback. Output: "LGTM" or a numbered list of required changes.

### Step 3: Address feedback

If the reviewer flags ANY issues, fix ALL of them and re-run steps 1-2.
You must NOT skip or dismiss any reviewer suggestion.

### Step 4: Commit

- Stage relevant files (not .env, credentials, etc.)

- Commit message format: conventional commit, lowercase description.

  - `type: short description` or `type(scope): short description`
  - Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `exp` (for experiments).
  - Example: `feat: add sequential block placement for dynamic machine`
  - Example: `fix(tokenizer): handle empty input on packed batches`

- End with a `Co-Authored-By:` trailer identifying the model that wrote
  the commit. Pick the row that matches your runtime:

  | Runtime            | Trailer                                                              |
  | ------------------ | -------------------------------------------------------------------- |
  | Claude Code        | `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`            |
  | Codex CLI          | `Co-Authored-By: GPT-5.5 (OpenAI Codex) <noreply@openai.com>`        |
  | opencode + GPT-5.5 | `Co-Authored-By: GPT-5.5 (opencode) <noreply@openai.com>`            |
  | opencode + Claude  | `Co-Authored-By: Claude Opus 4.7 (opencode) <noreply@anthropic.com>` |

  If your model isn't listed, fill in
  `<Model name> (<runtime>) <noreply@<provider>.com>`.

- Use HEREDOC for the commit message:

```
git commit -m "$(cat <<'EOF'
type: short description here

Co-Authored-By: <pick from table above>
EOF
)"
```

- Do NOT push to the remote unless the user explicitly asks
