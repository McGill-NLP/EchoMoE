---
name: paper
description: "Help discover, read, summarize, and analyze academic papers: including finding the relevant literature for a topic and locating reference implementations. Use when the user asks to summarize a paper, compare papers, look up the literature on a topic, or find an implementation of a method."
---

# Paper Reading Assistant

Help the user discover, read, and analyze academic papers. And find
reference implementations they can adapt.

## When discovering literature (lookup mode)

Triggered by: *"what's the literature on X?"*, *"find papers about Y"*,
*"how is Z implemented in recent work?"*, *"is there a reference impl for W?"*

### Step 0: Check what you already have, first

Before any web search:

1. **Grep the current repo** for the technique, function name, or paper title.
   Sometimes the answer is already in the codebase.
2. **Check shared project storage** if on a cluster
   (`/network/projects/<group>/` on Mila, `$PROJECT` on Compute Canada).
   the lab often has a working implementation lying around.
3. **Check the user's references / bibtex files** if present
   (`references.bib`, `paper/refs.bib`, etc.). The user may have already
   read and cited the relevant work.

Skipping this step is the #1 way to waste a session re-inventing something
that already exists locally.

### Step 1: Web search, in this order

1. **Papers with Code** (`paperswithcode.com/search?q=...`). Best entry
   point when the user wants to **implement** something. Links papers to
   open-source implementations and benchmark results.
2. **arXiv search** (`arxiv.org/search/?searchtype=all&query=...`). For
   the canonical recent papers. Filter by `cs.LG`, `cs.CL`, `stat.ML`.
3. **Hugging Face Papers** (`huggingface.co/papers?q=...`). Recent arXiv
   papers, often with associated model checkpoints / Spaces.
4. **Semantic Scholar** (`semanticscholar.org`). Citation graph, useful
   for tracing "who built on this paper".
5. **Google Scholar**: broader / older literature, including non-arXiv
   conference proceedings.
6. **GitHub code search** (`github.com/search?type=code&q=...`). For
   finding implementations of a specific function / module name.

Use `WebSearch` first for query expansion, then `WebFetch` to pull abstracts
from arXiv pages directly (`arxiv.org/abs/<id>`).

### Step 2: Shortlist 3–5 papers

Don't dump 20 results on the user. Pick:

- 1–2 **seminal** papers (highly cited, introduce the technique).
- 1–2 **recent / state-of-the-art** papers (last 12–18 months).
- 1 **with a reference implementation** if implementation is the goal.

Flag which is which when reporting back.

### Step 3: Per-paper structured pull

For each shortlisted paper, fetch the abstract page (`arxiv.org/abs/<id>`)
and return a uniform block:

```
**<Title>** (Authors, Year, Venue)
arXiv: https://arxiv.org/abs/<id>   |   Code: <link or "none found">

Problem: <one sentence>
Method:  <2–3 sentences; include the core equation or training recipe if central>
Key hyperparameters: <model size, batch, LR, training tokens. Only if reported>
Why relevant: <one sentence tying it back to the user's question>
```

If the user wants to implement: include the code link, the key
hyperparameters, and a one-line note on what differs from the "obvious"
implementation (e.g., "uses pre-norm, not post-norm"; "trained on
bf16 with grad clipping at 1.0").

### Step 4: BibTeX on request

If the user is writing something, offer to extract BibTeX from the arXiv
page (`arxiv.org/bibtex/<id>` returns it directly). Don't add it unprompted.

### Output discipline

- **Don't hallucinate paper titles, authors, or arXiv IDs.** If a result
  isn't verified by a fetch, say "I'm not certain of the exact citation."
- **Prefer published / arXiv versions over blog posts.** Blogs are fine
  as secondary sources but should not be the only reference for a claim.
- **Distinguish "this paper exists" from "I've read this paper."** If
  you've only seen the abstract, say so.

## When Summarizing a Paper

Provide a structured summary:

1. **Problem**: what problem does the paper address? Why does it matter?
2. **Key idea**: what is the core contribution in 1-2 sentences?
3. **Method**: how does it work? Include key equations if relevant.
4. **Results**: main experimental findings, with numbers where possible
5. **Limitations**: what are the weaknesses or open questions? (Your analysis, not just what the authors say)

## Guidelines

- **Be precise with claims**: distinguish between what the paper proves, demonstrates empirically, and merely conjectures
- **Include math when relevant**: the user has a theoretical background, don't shy away from equations
- **Note assumptions**: what assumptions does the method rely on? Are they realistic?
- **Compare to related work**: if you know of relevant prior/concurrent work, mention it
- **Flag unclear or suspicious claims**: if something doesn't add up, say so
- **Don't hallucinate citations**: see `/latex` for the canonical rules

## When Comparing Papers

- Focus on the key differences in assumptions, method, and results
- Use a table when comparing multiple papers on the same axes
- Note which results are directly comparable vs not (different datasets, metrics, etc.)

## Scope

$ARGUMENTS
