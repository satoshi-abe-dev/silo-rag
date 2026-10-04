# Evaluation details for the agent mode (nodes G and H)

The tables, numbers, question sets and measuring method behind the evaluation in the "Node G" section of the [README](../README_en.md). The README carries only the summary and the conclusion. The terms used here ("plain mode", "agent mode", "search-query writing") are explained in the [Node G](../README_en.md#node-g-the-langgraph-agent) section of the README.

## Evaluation

- The tables below are from one evaluation set (15 questions, 5 of them cross-department), run once per variant. **These 15 questions are the ones I was looking at when I designed the search-query writing, so the design probably fits them especially well** (the 45-question re-measurement is further down). 7B = `qwen2.5-7b-instruct`, 32B = `qwen2.5-coder-32b-instruct-mlx` (both in LM Studio, 4-bit)
- "Query writing" is `first_query = "rewrite"`; "1 search" is `max_attempts = 1`
- "LLM calls" are the chat calls up to the answer (embeddings and judge excluded). Measured before "skip grading at the cap", so each question that reached the cap actually costs one call fewer
- The judge uses the same model as the answerer, so **judge scores are only comparable between rows of the same model**

**7B**

| Variant | hit_rate | Cross-dept | Same-dept | MRR | Citation rate | judge | Searches | LLM calls | Sec/question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plain | 0.60 | 0.40 | 0.70 | 0.49 | 0.60 | 3.67 | 1.0 | 2.0 | 10.8 |
| G strict | 0.73 | 0.40 | 0.90 | 0.61 | 0.60 | 3.73 | 1.9 | 6.4 | 17.8 |
| G lenient | 0.67 | 0.40 | 0.80 | 0.54 | 0.60 | 3.80 | 1.4 | 4.5 | 14.6 |
| G strict + query writing | 0.80 | 0.60 | 0.90 | 0.66 | 0.80 | 3.73 | 2.4 | 9.0 | 19.5 |
| G lenient + query writing | 0.87 | 0.60 | 1.00 | 0.69 | 0.80 | 3.60 | 1.6 | 6.2 | 14.9 |
| **G lenient + query writing + 1 search** | **0.93** | **0.80** | 1.00 | 0.72 | **0.87** | 3.53 | 1.0 | 4.0 | **12.0** |
| LangChain stock ([node H](../README_en.md#node-h-langchain-integration)) | **0.93** | **0.80** | 1.00 | 0.54 | **0.87** | 3.73 | 1.3 | 3.3 | 18.6 |

**32B**

| Variant | hit_rate | Cross-dept | Same-dept | MRR | Citation rate | judge | Searches | LLM calls | Sec/question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plain | 0.87 | 1.00 | 0.80 | 0.71 | 0.87 | 3.60 | 1.0 | 2.0 | 50.4 |
| G strict | 0.87 | 1.00 | 0.80 | 0.72 | 0.80 | 3.33 | 2.5 | 8.2 | 88.9 |
| G lenient | 0.87 | 1.00 | 0.80 | 0.74 | 0.73 | 3.27 | 1.1 | 3.5 | 55.8 |
| G lenient + query writing | 0.87 | 0.60 | 1.00 | 0.73 | 0.80 | 3.00 | 1.2 | 4.8 | 63.8 |
| G lenient + query writing + 1 search | 0.93 | 0.80 | 1.00 | 0.72 | 0.87 | 3.33 | 1.0 | 4.0 | 57.2 |

I didn't measure the 32B model with the stock LangChain agent (LM Studio's model list shows no tool-support icon for it).

**What the results show**

- **On the 15 questions, what helped looked like not "searching again" but "writing a search query from the question" (7B).** With the question used as is, adding re-search only moved hit_rate from 0.60 to 0.67–0.73. Having the LLM write the first query took it to 0.80–0.93, and **limiting it to a single search (0.93) did no worse**. Cross-department questions also rose from 0.40 to 0.80. The step-by-step breakdown of where it helps is in the [worked example](worked_example_en.md) (also from the 15 questions). **However, re-measured on 45 questions, this effect could not be confirmed** (see "Re-measured on 45 questions" below)
- **This breakdown was prompted by losing to the stock LangChain agent.** It reached 0.93 without searching more. Looking into it, the LLM wrote a keyword-style query from the question before every search, while my G used the question as is on the first search. My original hypothesis ("re-search makes up for it") was only half right
- **With 32B, none of the tweaks shows a clear effect.** Plain mode was already at 0.87; query writing + 1 search reached 0.93 (one question). Cross-department dropped from 1.00 to 0.80 (one question), and to 0.60 once re-search was added
- **Conclusion: plain mode stays the default answer mode; the agent looked better on small models with the 15 questions, but on 45 questions no difference from plain mode could be confirmed.** The agent's defaults are `first_query = "rewrite"` with `max_attempts = 1`, the best-measured on the 15 questions (the "query writing + 1 search" rows). Re-measured on 7B with 45 questions, these defaults behave the same as plain mode with search-query writing turned on, and the difference is within chance (see "Re-measured on 45 questions" below). The re-search loop is available by raising `max_attempts`, but it showed no benefit on top of query writing with either model. With larger models, use plain mode
  - After making these the defaults, I re-ran both models; per-question hit/miss and metrics such as hit_rate matched. LLM calls dropped from the 4.0 in the table to 3.0 because grading is skipped at the cap

**Re-measured on 45 questions (plain mode, without and with search-query writing)**

I turned on `rewrite_query` in plain mode's `search()`, and ran the agent's defaults (`first_query = "rewrite"`, `max_attempts = 1`) on 7B, with more questions: 45, one run per condition.

| Model and variant | hit_rate | Cross-dept | Same-dept | MRR | Citation rate | judge | LLM calls | Sec/question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7B, no query writing | 0.71 | 0.60 | 0.77 | 0.50 | 0.64 | 3.80 | 2.0 | 10.8 |
| 7B, query writing | 0.76 | 0.60 | 0.83 | 0.52 | 0.67 | 3.73 | 3.0 | 10.8 |
| 7B, agent defaults (query writing + 1 search) | 0.76 | 0.60 | 0.83 | 0.52 | 0.71 | 3.78 | 3.0 | 11.7 |
| 32B, no query writing | 0.76 | 0.47 | 0.90 | 0.57 | 0.69 | 3.11 | 2.0 | 49.3 |
| 32B, query writing | 0.80 | 0.60 | 0.90 | 0.61 | 0.73 | 2.93 | 3.0 | 50.5 |

- **Questions**: 45 (15 cross-department, 30 same-department), built mechanically from the report specifications (no LLM). Same reports and same construction as the 15-question evaluation; 13 questions have the same wording as the 15, and 32 are new
- **Counting hit_rate changes question by question, the difference is within chance** (sign test)
  - 7B: 7 improved, 5 got worse (p=0.77)
  - 32B: 6 improved, 4 got worse (p=0.75). Cross-department went from 0.47 to 0.60, but 4 improved and 2 got worse (p=0.69)
- **Restricted to the 13 questions shared with the 15, there is an improvement; restricted to the 32 new ones, there is none.** Improved / got worse: 7B 4 / 0 on the 13 and 3 / 5 on the 32 new ones; 32B 2 / 0 on the 13 and 4 / 4 on the 32 new ones. The large improvement seen on the 15 questions probably looked large because the design was tuned to them
- **Cost**: LLM calls go from 2.0 to 3.0 per question. The elapsed time is about the same (7B 10.8 to 10.8 s, 32B 49.3 to 50.5 s)
- **The agent's defaults behave the same as plain mode with search-query writing turned on (7B).** The retrieved reports are identical, in order, for 44 of 45 questions, and the hit/miss outcome is identical for all 45 (one question differed slightly because of LLM variation). With one search, no grading or re-search happens, so it is just "write the query, search, answer". So its difference from plain mode is the same as that of search-query writing above, within chance (7 better, 5 worse, p=0.77). I did not measure 32B because the structure is the same. The re-search loop (raising `max_attempts`) was not measured on 45 questions
- **Conclusion**: since no effect could be confirmed, `[retrieval] rewrite_query` stays off

> ⚠️ **Every result is a single run per condition. A one-question difference (0.07 on 15 questions, 0.02 on 45) can't be called real; read the results as a trend on these question sets.**
>
> - Temperature 0 does not guarantee an exact repeat. What varies is answer generation (citation rate, judge) and the stock LangChain agent's search (its tool query is written by the LLM at temperature 0.2 each time)
> - The stock agent's variation is something I observed across several runs I stopped partway; only the last run's result is saved, so the numbers don't back it up
> - That retrieval reproduces doesn't mean a different question set would give the same result. In fact, the search-query writing I chose on the 15 questions showed no effect on 45 questions (32 of them new)
> - I also tried Gemma 4 26B (a thinking model), but it took about 174 s per question, so I cut it off midway and left it out of the comparison

## Problems found along the way

- **Three findings from the independent review (codex)**: CI didn't install the extra libraries for the agent mode (LangGraph etc.); a retry pushed out all earlier evidence; the grading prompt had no conversation history. Each was reproduced with a mock, fixed, and given a regression test
- **Three findings from running a real LLM (7B)** (mock-based tests alone didn't surface these): grading was so strict it retried up to the cap every time; the queries written for the second search included the asker's own department; interleaving the retries' results pushed out a gold report the original query had found (fixed by the `select` node, which reranks everything collected against the original question)
- **A design mistake found by the evaluation**: the original hypothesis that "re-search helps" was only half right. And the next conclusion, "writing a search query helps", was too strong because it was tuned to the 15 questions (found by re-measuring on 45)
