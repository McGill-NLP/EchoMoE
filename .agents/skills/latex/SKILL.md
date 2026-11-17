---
name: latex
description: Help with LaTeX writing, formatting, and debugging. Use when the user asks about LaTeX, papers, equations, TikZ, Beamer, or BibTeX.
---

# LaTeX Assistant

Help the user write, edit, or debug LaTeX documents.

## Capabilities

- **Writing**: Draft sections, paragraphs, captions, abstracts
- **Math**: Typeset equations, align environments, theorem blocks
- **Figures & Tables**: TikZ diagrams, table formatting, float placement
- **Beamer**: Slide decks for talks and presentations
- **BibTeX**: Citation formatting, .bib file management
- **Debugging**: Fix compilation errors, bad references, spacing issues

## Guidelines

- **Read before writing.** Before editing a LaTeX project, read the preamble and any custom .sty/.cls files to understand the document structure. What packages are loaded, what custom environments and commands exist, and what conventions the author follows.
- **Use existing macros.** If the document defines `\newcommand`, `\newenvironment`, or similar, use them. Don't redefine or duplicate existing macros. If a useful macro doesn't exist, suggest defining one rather than repeating raw LaTeX.
- **Respect the template.** Conference/journal templates (NeurIPS, ICML, ACL, etc.) have strict formatting rules. Don't override their spacing, fonts, or layout unless explicitly asked.
- **Match existing style.** Follow the conventions already in the document (e.g. `\textbf` vs `\bfseries`, `equation` vs `align`, `\cref` vs `\ref`).
- **Use standard packages.** Don't introduce obscure dependencies.
- **Math clarity over compactness.** Use `\left( \right)` only when needed. Prefer readable notation.
- **For compilation errors.** Explain the root cause, don't just give the fix.

## Citation hallucination: high priority

LLM-generated text frequently fabricates citations: real-sounding authors,
plausible titles, well-formed arXiv IDs that point to nothing or to an
unrelated paper. **This is the single most damaging failure mode in paper
writing**. It survives skimming, gets caught only at peer review or
post-publication, and reflects on the author, not the assistant.

**Hard rules:**

1. **Never invent a citation.** Don't write a `\cite{...}` key, fabricate
   author/year, or generate a BibTeX entry from memory. If a reference is
   needed and not in `references.bib`, write a visible placeholder like
   `\cite{TODO-need-cite-for-X}` or `[CITATION NEEDED: X]` and tell the user.
2. **Cite only what's already in the project's `.bib` file** unless the user
   asks you to add a new entry. And if they do, verify it first (see below).
3. **Verify before adding.** When asked to add a new reference:
   - Fetch the arXiv abstract page (`arxiv.org/abs/<id>`) or the publisher
     page to confirm the title and authors exist as claimed.
   - Pull BibTeX from `arxiv.org/bibtex/<id>` rather than typing it by hand.
   - If you can't verify (no network, paywalled venue, ambiguous match),
     stop and ask. Do not generate an entry from training data.
4. **Cross-check existing `\cite{key}` uses.** When editing prose, don't
   trust that a citation key in the surrounding text actually points to the
   right paper. If a claim is load-bearing, grep `references.bib` for the
   key and confirm the title matches the claim being supported.
5. **Distinguish "this paper exists" from "I've read this paper."** If you
   only know the abstract, don't paraphrase specific results or numbers.

**Common hallucination patterns to flag in user-supplied drafts:**

- Plausible author lists that are actually mis-combinations (Author A of paper
  X + Author B of paper Y as co-authors of a non-existent paper Z).
- Real authors paired with a title close to (but not exactly matching) a
  paper they actually wrote.
- arXiv IDs with valid format (`YYMM.NNNNN`) but no paper at that ID.
- "Et al." citations where the lead author never wrote on the topic.

If you spot one, surface it: *"I can't verify this citation. The arXiv ID
2305.12345 doesn't resolve. Could be a typo or a hallucination from an earlier
draft."*

See `/paper` for finding the correct citation when one is needed.

## Scope

$ARGUMENTS
