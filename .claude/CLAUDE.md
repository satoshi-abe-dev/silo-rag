# cae-rag project operating rules

These apply on top of the workspace-wide rules (`/Users/satoshi/myFolder/.claude/CLAUDE.md`).

## Document language

- A document that only needs to exist in one language (this file included) is written in English, following the `meeting-minutes` project's convention.
- This doesn't apply to documents that are deliberately bilingual pairs, such as `README_ja.md` / `README_en.md` — see the README section below.

## Branching

- No direct commits or pushes to `main`. Work happens on a branch, opened as a PR via `gh pr create`.
- Merge a PR only after the user has explicitly approved it (and the checklist below is satisfied).

## Commit / PR language

- Commit messages and PR titles/descriptions are written in English going forward. (Past Japanese commit history is not rewritten.)
- Conversational replies to the user stay in Japanese as before.

## Pre-merge checklist

Before merging a PR into `main`, confirm:

- `.gitignore` correctly excludes `config.toml` / `data/chroma_db/` etc.
- The diff stays within the intended scope (nothing beyond the corresponding issue/request has crept in).
- Tests pass (`pytest`).
- No destructive operations (force push, history rewrite, etc.) are included.
- An independent review (e.g. the `codex` CLI, a different vendor's AI) has been run, with any findings fixed and re-reviewed.

## Changing config/permission files

- Get the user's confirmation before changing `config.toml`, this directory's `CLAUDE.md`, or permission settings (`settings.json`, etc.).

## README translation pairing

- If you edit `README_ja.md`, apply the same change (wording, ordering, etc.) to `README_en.md` too. Never leave a change to just one side.
- The same applies in the other direction, if `README_en.md` is edited first.

## Translation quality

- When translating Japanese into English (docs, commit messages, etc.), prefer natural, idiomatic English phrasing over a literal, word-for-word translation. Preserve the exact meaning and technical accuracy, but rephrase sentence structure and word order the way a native English writer would.
