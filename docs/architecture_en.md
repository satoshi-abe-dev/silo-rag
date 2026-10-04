# How it works: the nodes and the search

[日本語](architecture_ja.md) | English

This document explains the eight nodes (units of work, one file each) in the [README](../README_en.md#architecture) diagram, and how the search (node C) works. It is easier to follow after a look at that diagram.

What this document covers:

- What each node does and which AI models it uses
- Which functions of the other nodes the optional nodes G and H call
- The two kinds of search (BM25 keyword search and vector search, which looks for meaning) and how they are combined

## What each node does, and the models it uses

The model names (`llm_model` and so on) are set under `[ai]` in `config.toml`.

| Node | File | Role | Models used |
| --- | --- | --- | --- |
| A | `src/silo_rag/datagen.py` | - Writes the demo's synthetic retrospective reports (house style and file format vary by department; an outcome chart is embedded)<br>- Writes the evaluation questions and their correct answers | `llm_model` (report body generation) |
| B | `src/silo_rag/ingest.py` | - Splits each report by heading (into chunks; five formats supported)<br>- Turns embedded images into text with an AI that can read images (a VLM)<br>- Turns each chunk into a list of numbers (a vector) and stores it in ChromaDB | - `embed_model` (chunk vectors)<br>- `vlm_model` (image descriptions) |
| C | `src/silo_rag/retrieval.py` | - Searches by combining keyword search (BM25) and semantic search (vector search)<br>- Reranks the candidates with an LLM<br>- Uses the conversation history to turn references such as "that" into concrete words<br>- (Optional) writes a search query from the question before searching (`[retrieval] rewrite_query`, off by default) | - `embed_model` (question vector)<br>- `llm_model` (reranking, resolving references, writing the search query) |
| D | `src/silo_rag/generation.py` | - Writes the answer, grounded in the chunks found<br>- Attaches citations (report ID, section, department)<br>- Uses the conversation history to resolve references | `llm_model` (answer generation) |
| E | `src/silo_rag/eval.py` | - Measures retrieval accuracy (Recall@k, MRR)<br>- Measures answer quality (scoring by an LLM, the rate of citing the correct report)<br>- For what the metrics mean, see the [evaluation](agent_evaluation_en.md#how-to-read-the-tables) | - Every model used by C and D<br>- `llm_model` (scoring) |
| F | `src/silo_rag/app.py` | - The screen (a Streamlit chat page)<br>- Filtering by department and project type, showing the citations | Every model used by C and D (on every question) |
| G (optional) | `src/silo_rag/agent.py` | - Runs "write a search query → search → grade → write a query from a different angle → search again" with LangGraph<br>- Details: [the agent mode's design](agent_en.md#node-g-the-langgraph-agent) | Every model used by C and D |
| H (optional) | `src/silo_rag/langchain_adapter.py` | - Makes the existing search usable as a LangChain component (a Retriever)<br>- Answers with LangChain's stock agent, to compare it with G<br>- Details: [the agent mode's design](agent_en.md#node-h-langchain-integration) | Every model used by C and D |

If `vlm_model` isn't loaded, only B's image descriptions are skipped (a warning is logged; everything else works).

## Which functions of C and D G and H call

G and H are not inside C or D; they are separate nodes that call the public functions of C and D.

- **G** (`agent.py`): calls C's `search` (search), `plan_query` (write a search query) and `rerank` (rerank), and D's `answer_question` (write the answer)
- **H** (`langchain_adapter.py`): calls C's `search`, and D's `Answer` (the answer type) and `build_citations` (make the citations)
- Both are imported only when used. Plain mode works without LangGraph or LangChain installed

## How the search works: BM25 and vector search

C combines two searches of different kinds.

- **BM25 (keyword search)**: a **formula** that scores each document by how often, and how distinctively, the question's words appear in it. It is not an AI model and involves no training
  - The more often a word appears, the higher the score (with diminishing returns). A word found in almost every document (the Japanese equivalents of "is", "please") barely counts; a rare word (say, "budget planning") counts a lot. Long documents are discounted a little
  - Good at: searches where **the words themselves match** (part numbers, proper nouns, technical terms)
  - Weak at: treating a paraphrase ("budget" vs. "cost estimate") as the same thing. Words unrelated to the search raise the score of any document that happens to contain them, which can shift the ranking (a word absent from the index scores zero and changes nothing)
- **Vector search (semantic search)**: an embedding model (an AI model) turns each sentence into a list of numbers, and distance measures whether the **meaning is close**. Handles paraphrases well
- **Combining the scores**: each score is **normalized** to 0–1, then the two are added using `vector_weight` (0.5 by default), and an LLM reranks the top candidates
  - Normalization: the best score becomes 1, the worst 0, and the rest are rescaled proportionally. Exceptions: if the highest and lowest scores are almost equal, all become 1; for keyword search, if the highest score is 0 or below, all become 0 (checked first)
  - A calculation with real numbers is in the [worked example](worked_example_en.md)
- **In this project**: preprocessing is deliberately simple (no morphological analyzer). Japanese text is cut into overlapping **two-character pieces** (e.g., "予算策定" → "予算", "算策", "策定"), while ASCII words and IDs (such as `RPT-014`) are **kept whole and lowercased** (`rpt-014`). It uses `BM25Plus` from `rank_bm25`; `BM25Okapi` can give a negative weight (IDF) to a word found in more than half the documents, which genuinely happens on a small corpus, so it was replaced

## Related documents

- [Worked example: how one question becomes an answer](worked_example_en.md): follows each step of this search with real values
- [The agent mode's design](agent_en.md): nodes G and H in detail
- [How it was developed](development_process_en.md): the dependencies between nodes (the DAG) and the development process
