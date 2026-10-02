# Worked example: how one question becomes an answer

[日本語](worked_example_ja.md) | English

This follows a question through each processing step, using **real values**. It helps to read
[the README's "Terms: BM25 and vector search"](../README_en.md#terms-bm25-and-vector-search) first.

## Setup

- The data is synthetic (60 dummy in-house project reports).
- Query writing, reranking, and answer generation use `qwen2.5-7b-instruct`; embeddings use
  `text-embedding-nomic-embed-text-v1.5` (both in LM Studio).
- The values were taken in October 2026 with throwaway scripts (not included in the repository). They step through the
  inside of `search()` one stage at a time, and the script checked that the result matched the real `search()`.
  **Steps that involve an LLM vary a little from run to run, even at temperature 0** (the query written for the same
  question differed in wording and spacing between runs). Read the values as "one particular run".

## The overall flow

| Step | Input → output |
| --- | --- |
| 0. Ingestion (once, beforehand) | report file → one chunk per heading → one vector (768 numbers) per chunk |
| 1. Query writing (LLM) | question → search query (only the words for what you want to find) |
| 2. BM25 | query → scores for chunks whose words match (top 20) |
| 3. Vector search | query → scores for chunks whose meaning is close (top 20) |
| 4. Blending | scale both scores to 0–1 and add them half and half → top 20 |
| 5. Reranking (LLM) | the 20 candidates → ordered by relevance to the question → top 5 |
| 6. Answer generation (LLM) | the top 5 chunks → an answer with citations |

Steps 2–5 are the inside of `search()` in retrieval (node C). Step 1 is done by the agent's (node G) `plan` node before the
query is handed to `search()` (plain mode has no such step; the question goes straight to `search()`). Step 6 is answer
generation (node D).

## 0. Ingestion: one report becomes chunks and vectors

Take `RPT-018.md` (Planning Dept., marketing initiative, subject "budget-planning process review"). The metadata at the top
of the file is separated from the body, which is split by heading into chunks.

Metadata (attached to every chunk): `report_id=RPT-018`, `dept=経営企画部` (Planning Dept.),
`project_type=マーケティング施策` (marketing initiative), `subject=予算策定プロセス見直し` (budget-planning process review),
`method=アジャイル（スクラム）` (agile/scrum), `date=2024-06-03`, and so on.

| Chunk ID | Text (beginning; translated) |
| --- | --- |
| `RPT-018::プロジェクト目的` (Project purpose) | [Project purpose] Review the budget-planning process for marketing initiatives to achieve more efficient and flexible budget management. |
| `RPT-018::対象領域・テーマ` (Scope and theme) | [Scope and theme] The budget-planning process as a whole, for marketing initiatives. |
| `RPT-018::与件` (Constraints) | [Constraints] The current budget-planning process needs a detailed review; past projects repeatedly overran their budgets. |
| `RPT-018::教訓・つまずいたポイント` (Lessons) | [Lessons and stumbling points] The overrun was noticed late: by the project's halfway point, 120% of the total budget had been used. … |
| (4 more) | execution plan, main resources, adopted method, results summary |

This report becomes 8 chunks. The embedding model turns each chunk into **a list of 768 numbers** (a vector), stored in
ChromaDB. For example, the first four numbers of the vector for `RPT-018::プロジェクト目的` are
`[0.010, 0.018, -0.132, 0.031]` (the other 764 are omitted). Sentences with close meanings get close lists of numbers.
BM25, on the other hand, uses the chunk text itself (recomputed over all chunks on every search).

## Example 1: a question that works (QA-005)

> **Question** (translated): Are there past in-house cases of a marketing initiative that reviewed its budget-planning
> process? Please also tell me the conditions for carrying it out and what to watch out for.
>
> **Gold report**: RPT-018 (Planning Dept.)

### 1. Query writing

| | Content | Pieces used for search |
| --- | --- | --- |
| Question | (above) | 58 |
| Query | 予算策定プロセス見直し マーケティング施策 社内事例 実施条件 注意点 | 26 |

"Are there…?" and "please tell me" are gone; the words for what is being looked for remain.
(Pieces are counted as in the search preprocessing: Japanese is cut into overlapping two-character pieces — "予算策定" →
"予算", "算策", "策定" — and an ASCII word counts as one piece.)

### 2–4. BM25, vector search, blending

| Rank | BM25 (score) | Vector search (score = negative distance) | Blended (total = BM25 part + vector part) |
| --- | --- | --- | --- |
| 1 | RPT-018::対象領域・テーマ (123.8) | RPT-018::プロジェクト目的 (-0.353) | RPT-018::プロジェクト目的 (0.98 = mean of 0.96 and 1.00) |
| 2 | RPT-018::プロジェクト目的 (122.4) | RPT-018::与件 (-0.410) | RPT-018::対象領域・テーマ (0.74) |
| 3 | RPT-016::対象領域・テーマ (105.6) | RPT-056::プロジェクト目的 (-0.410) | RPT-018::与件 (0.40) |
| 4 | RPT-018::与件 (97.8) | RPT-035::プロジェクト目的 (-0.415) | RPT-035::プロジェクト目的 (0.36) |
| 5 | RPT-025::プロジェクト目的 (96.8) | RPT-044::実施条件 (-0.419) | RPT-025::プロジェクト目的 (0.33) |

- A BM25 score is an absolute value set by word matches, so it cannot simply be added to a vector score. Vector scores
  are larger when the distance is smaller. So each is **scaled so that the minimum and maximum among its top 20 become 0
  and 1**, then the two are added half and half.
- The blended first place is a chunk that was second in BM25 and first in vector search. Chunks that score high on both rise.

### 5. Reranking (LLM)

The LLM reorders the blended top 20 by relevance to the question and picks the top 5. (The reranking LLM receives the
**rewritten query**; the answer-generating LLM receives the original question.)

1. RPT-018::プロジェクト目的 2. RPT-018::与件 3. RPT-015::対象領域・テーマ 4. RPT-035::プロジェクト目的
5. RPT-016::対象領域・テーマ

`RPT-015`, which was not in the blended top 5, entered at third place (the LLM picked it from the 20 candidates).

### 6. Answer generation (LLM)

The answer is written from the top 5 chunks only. Inline references such as "(RPT-018)" are written by the LLM. Separately,
a list of sources is built mechanically from the metadata of the top 5 chunks (deduplicated by report and section).

> (translated) As an in-house case that reviewed the budget-planning process for a marketing initiative, the Planning Dept.
> ran "Review of the budget-planning process for marketing initiatives" (RPT-018). That project called for a detailed
> review, because past projects had repeatedly overrun their budgets (RPT-018). … (rest omitted)

The source list has five entries: two sections of RPT-018 (Planning), RPT-015 (Planning), RPT-035 (Planning), and RPT-016
(Product Development). The gold report, RPT-018, is included.

## Example 2: a question that doesn't work (QA-001)

> **Question** (translated): This is the Planning Dept. We are considering a marketing initiative on a theme similar to
> an existing-product renewal. Are there past marketing cases from other departments we could learn from? Please tell us
> especially about pitfalls to watch out for.
>
> **Gold report**: RPT-041 (Product Development Dept.)

### 1. Query writing

| | Content | Pieces |
| --- | --- | --- |
| Question | (above) | 101 |
| Query (7B) | 経営企画部既存商品リニューアルマーケティング施策落とし穴 | 27 |
| (for reference) Query (32B) | 既存商品リニューアル マーケティング施策 落とし穴 | 20 |

The 7B model kept the asker's own department, "経営企画部" (Planning Dept.), in the query (the prompt tells it not to).
The 32B model did not. In another run of the same 7B model on the same question, the query came out as
"経営企画部 商品リニューアル マーケティング施策 落とし穴", with spaces and without "既存" (existing) — "経営企画部"
was kept both times.

### 2–4. BM25, vector search, blending

In BM25, **the gold report RPT-041 moved from third place to first once the question was rewritten** (question as is:
3rd; 7B query: 1st; 32B query: 1st).

| Rank | BM25 (score) | Vector search (score) | Blended (total = BM25 part + vector part) |
| --- | --- | --- | --- |
| 1 | **RPT-041::成果サマリー** (117.8) | RPT-017::推進体制 (-0.299) | RPT-017::推進体制 (0.50 = mean of 0.01 and 1.00) |
| 2 | RPT-018::対象領域・テーマ (112.7) | RPT-042::推進体制 (-0.309) | **RPT-041::成果サマリー** (0.50 = mean of 1.00 and 0.00) |
| 3 | RPT-016::対象領域・テーマ (111.9) | RPT-005::主要リソース (-0.342) | RPT-042::推進体制 (0.48) |
| 4 | RPT-027::成果サマリー (109.9) | RPT-048::推進体制 (-0.345) | RPT-005::主要リソース (0.41) |
| 5 | RPT-025::主要リソース (108.8) | RPT-019::推進体制 (-0.358) | RPT-048::推進体制 (0.37) |

The vector-search top is filled with "推進体制" (project structure) chunks that have little to do with the question, and
**RPT-041 is not among them**. In the blended list, RPT-041 stays in second place.

### 5. Reranking (LLM)

After reranking, the top 5 are as follows, and **RPT-041 is gone**.

1. RPT-018::対象領域・テーマ 2. RPT-025::主要リソース 3. RPT-019::推進体制 4. RPT-048::推進体制
5. RPT-023::対象領域・テーマ

The 7B model's reranking dropped the gold chunk, which was second among the candidates, from the top 5.

### 6. Answer generation (LLM)

> (translated) No matching case was found (RPT-018 and RPT-025 concern the Planning Dept.'s budget-planning process, …).
> … (rest omitted)

For this question, **retrieval (collecting candidates) succeeded and reranking failed**. It is an example where looking
only at the query rewrite does not reveal the cause.

## At which step does the rewrite help? (7B, 15 questions)

Rewriting the question into a search query raised the 7B model's overall hit rate from 0.60 to 0.93. To see where that
comes from, I counted per step over all 15 questions. Numbers are "questions (of 15) where the gold report was included".

**The retrieval steps (before reranking; no LLM involved)**

| Step | Question as is | 7B query | 32B query |
| --- | --- | --- | --- |
| BM25 alone, top 5 | 11 | **15** | **15** |
| Vector search alone, top 5 | 5 | 5 | 7 |
| Right after blending, top 5 | 10 | 12 | 11 |
| Blended top 20 (the reranking candidates) | 15 | 15 | 15 |

- **BM25 improved a lot with the rewrite** (11 → 15). But a rewrite does more than drop extra words: it also changes the
  choice of words and the spacing (the 7B query for QA-001, for instance, drops "既存"). Whether "removing the extra
  words" is the reason is unconfirmed — I did not run an experiment that isolates it.
- **Vector search was unchanged in total** (5 → 5) and is weak to begin with (5–7 questions). Per question, though, two
  questions gained a hit and two lost one, so they merely swapped places. Vector rankings and scores are also used in the
  blending, so they can affect the blended result even when the top-5 hit count does not change.
- **The 20 reranking candidates already contained the gold report for all 15 questions, even before rewriting.** So
  misses do not happen in collecting candidates; they happen in choosing the top 5 from them.

**The final top 5 (after reranking), swapping the query used to collect candidates and the query used to rerank**

| | Rerank: question as is | Rerank: rewritten |
| --- | --- | --- |
| **Candidates: question as is** (plain mode) | 9 | 12 |
| **Candidates: rewritten** | **15** | 14 (the agent's default) |

- The rewritten query is handed to BM25 and vector search at the same time, so this result cannot be called "BM25's effect
  alone".
- Collecting candidates with the rewritten query helps a lot (9 → 15). Using the rewritten query only for reranking also
  helps (9 → 12). Rewriting both is not additive (14).
- The best combination was "collect candidates with the rewritten query, rerank with the original question" (15). With it,
  Example 2 (QA-001) keeps the gold report in the top 5. Perhaps the original question conveys the asker's intent (wanting
  other departments' cases, for instance) to the reranking LLM, but I haven't checked. **It is a one-question difference,
  so I did not adopt this combination.**

## Caveats

- **7B only.** (For 32B, only the BM25 and vector-search steps were measured; the reranking LLM was not run.)
- **15 questions, one run each.** LLM output varies a little from run to run; a one-question difference can't be called
  real.
- The "why the rewrite helps" statements go no further than what the per-step numbers show. For example, "a short query
  is easier for the LLM to judge during reranking" is still unconfirmed.
