---
name: review
description: Review code for bugs, style issues, and correctness. Use when the user asks for a code review, review, or CR.
---

# Code Review

Review the provided code (file, diff, or selection) systematically.

## Review Checklist

1. **Correctness**: logic errors, off-by-ones, edge cases, wrong assumptions
2. **Bugs**: null/None handling, uninitialized variables, race conditions
3. **Types & shapes**: tensor dtype/device mismatches, shape errors, implicit casts
4. **Security**: injection, hardcoded secrets, unsafe deserialization
5. **Style**: naming, dead code, unnecessary complexity, consistency with surrounding code
6. **Performance**: unnecessary copies, repeated computation, memory leaks

## Output Format

For each issue found:

- **File and line**: where the issue is
- **Severity**: critical / warning / nit
- **What**: one-line description
- **Why**: brief explanation
- **Fix**: suggested change (code snippet if helpful)

If no issues are found, say so. Don't invent problems.

## Scope

$ARGUMENTS
