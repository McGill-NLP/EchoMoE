# ML / research operations

These rules govern how to work with data, tests, and specs in this research repo.
They complement (do not replace) the core principles.

## Never inspect data

ML pipelines in this repo may contain sensitive, large-scale, or model-generated data
that can crash the session or trip content policies when surfaced.

- **Do NOT** `cat`, `head`, `tail`, `Read`, or otherwise view dataset / prompt /
  completion files.
- **Do NOT** print samples in tests. Assert on structure, shape, length, dtype.
- **Do NOT** log raw text during training or evaluation.
- **Do** trust the pipeline based on metadata (row counts, column names, dtypes,
  sequence lengths).
- **Do** use opaque fixtures in tests (mock data loaders, assert shapes not values).

If you must debug data, inspect **metadata only**, never content.

## Test-driven development

Write tests BEFORE implementing any new feature or algorithm. Don't ask first.

1. Write test cases that encode expected behavior.
2. Run them. Confirm they fail (red).
3. Implement.
4. Run them. Confirm they pass (green).
5. Refactor if needed.

File naming: `<module>_test.py` in `tests/` mirroring the source layout.

**Model testing policy:**

- **NEVER run real models in tests**; always mock.
- **NEVER require GPU/VRAM for tests**; mock anything that does.
- Use `unittest.mock` for model loading and inference.
- Environment-specific skips (e.g. `SLURM_JOB_ID`) are acceptable.

This couples with \[\[principles\]\] Rule 9: tests encode intent, not just behavior.

## Self-verification (don't trust your own diff)

After making changes:

1. Run relevant tests (`pytest tests/path/to_test.py`).
2. Run linter (`ruff check --fix`).
3. Run type checker (`pyright`).
4. If anything fails, fix the root cause. Do not suppress.

**Never dismiss any failure as "pre-existing."** Every file in this repo is
maintained by us. If a test failure, type error, or lint error appears (even
in code you didn't touch this session), fix it. Skipping or `--ignore`ing a
failing test is the same as hiding a bug. If a test is genuinely obsolete,
delete it and explain why. If behavior changed, update the test to match.

## No defensive code

See \[\[principles\]\] Rule 12 (fail loud) and Rule 2 (simplicity). Defensive code
is silent failure with extra steps.

## Comments & docstrings

Default to **no comments**. A comment exists to communicate something the code
can't. If a future reader could derive the comment by reading the function name
and body, the comment is noise.

Use plain technical English in code prose. Keep public docstrings short and
error messages direct. Avoid em dashes, metaphors such as "load-bearing," and
schema phrases such as "must both be null."

**Write a comment only when the *why* is non-obvious:**

- A hidden constraint (e.g., "must run before tokenizer init; torch sets the
  RNG state").
- A subtle invariant (e.g., "input is sorted by length; downstream packing
  relies on this").
- A workaround for a specific bug, with the source (e.g.,
  "transformers#28473: `attention_mask` must be `int64` for fp16 paths").
- Behavior that would surprise a careful reader.

**Don't write:**

- Comments that restate the code ("# increment counter" above `counter += 1`).
- Section headers in short functions ("# --- parse args ---").
- References to the current task, caller, or issue ID ("# added for the new
  SFT flow", "# called by train.py"). These rot as the codebase evolves and
  belong in the PR / commit message, not the source.

**Docstrings:**

- **Public API** (anything imported across modules, or part of a library
  surface): one short docstring describing purpose and any non-obvious arg
  semantics. One line is usually enough.
- **Internal helpers, private methods, test functions:** no docstring required.
  A good name beats a bad docstring.
- **No multi-paragraph docstrings.** No boilerplate `Args:` / `Returns:`
  sections that just restate type hints. Types belong in annotations, not
  prose.
- **Match the codebase's existing docstring style** (Google / NumPy /
  one-liner). See \[\[principles\]\] Rule 11.

The test: *if you delete the comment or docstring, does a reader lose
information they can't recover from the code itself?* If no, delete it.

## Reproducibility

A result that can't be reproduced isn't a result. Reproducibility isn't a
post-hoc "we'll release the code" promise. It's a set of disciplines
applied while the work is happening.

**Seed discipline:**

- **Set every seed** at process start: Python (`random.seed`), NumPy
  (`np.random.seed`), PyTorch (`torch.manual_seed`,
  `torch.cuda.manual_seed_all`), and any framework-specific RNG
  (`transformers.set_seed`, `jax.random.PRNGKey`).
- **Log the seed alongside the run.** A run without a logged seed is
  unreproducible by definition.
- **Run with ≥3 seeds** for any result you'll put in a paper. Single-seed
  numbers are noise. Never report them as headline results.
- For JAX: thread keys explicitly; never reuse a key. See `/jax-debug`.

**Environment capture:**

- **Commit the canonical lockfiles.** This repo uses separate compiled locks
  (`requirements-vllm.lock`, `requirements-megatron.lock`) instead of one
  `uv.lock`; keep them in sync with `pyproject.toml` so future installs resolve
  the same versions.
- **Don't `uv pip install` ad-hoc into a shared env.** If you need a new
  package, `uv add <pkg>` so it's tracked in `pyproject.toml`.
- **Pin the CUDA environment** in the job script. Source `scripts/env.sh`
  rather than mutating the environment ad hoc.

**Provenance logging at job start:**

Every training / eval job must log, before any real work:

```bash
echo "git_commit=$(git rev-parse HEAD)"
echo "git_dirty=$(git status --porcelain | wc -l)"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
nvidia-smi
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"
```

Reportable jobs must fail when the tree is dirty. Don't strip this check out.

**Config discipline:**

- **Never edit a config in place during a run.** Change the checked-in recipe
  before launch or use an explicit CLI override.
- **Save one resolved `config.yaml` authority** in the run directory. Do not
  copy the source recipe or add config hashes.

**Experiment tracking:**

- Local config, metrics, results, logs, checkpoints, command, Git commit, and
  hardware/container context are the required provenance.
- W&B is an optional metric mirror. A disabled or failed tracker must not
  change local files or resume behavior. W&B IDs and config hashes are not
  required evidence.
- Keep reportable outputs under the documented run directory on durable
  project storage. Scratch alone is not durable evidence.

**The reproducibility test:**

Before publishing or sharing a result, ask:

> *"If a collaborator clones this repo at commit X and runs
> `<documented command>`, do they get the same number?"*

If the answer requires hidden state (uncommitted changes, env vars set
in your shell, files outside the repo), fix it before claiming the result.

## Notes & research discussions

Durable, multi-paragraph thinking goes in `notes/`, not in the chat history,
not in code comments, and not in auto-memory. The directory is for:

- **Research discussions**: open questions, tradeoffs, dead-end paths with
  the reason they failed.
- **Design proposals**: extensions to the current code, before they become
  specs or PRs.
- **Parking-lot ideas**: anything worth not losing that isn't being acted on
  now.
- **Decision records**: short writeups of non-obvious technical choices and
  the alternatives considered.

**When to write a note** (vs other capture mechanisms):

- During a research discussion, if the user says *"that's interesting, we
  should think about it"* or *"park this for later"*: write a `parked` note.
  Don't trust it to memory.
- After a design decision: write a `closed` decision-record note with the
  alternatives and the reason for the chosen path.
- After a dead end: write a `closed` note explaining what failed and why,
  so the team doesn't re-walk the path.

**Conventions** (see `notes/README.md` for the full spec):

- Filename: `notes/YYYY-MM-DD-short-slug.md`.
- Status field: `open` / `closed` / `parked` / `superseded`.
- Link liberally to commits, local run directories, optional tracker runs,
  other notes, and paper sections.

**What does NOT go in `notes/`:** per-run experiment results (use local
`results.json`, and optionally a tracker), bug fixes (commit messages), code documentation (docstrings), personal scratch (keep outside
the repo).

## Self-improvement

When you hit a repeated error, surprising convention, or non-obvious gotcha that
better instructions would have prevented, suggest adding it to the relevant
`agents/*.md` module. Phrase as: *"I hit \[X\]. Should I add a rule to
`agents/<module>.md` to prevent this?"*

Keeps the template evolving as a living document.
