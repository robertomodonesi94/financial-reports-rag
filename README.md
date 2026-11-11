# finrag: a RAG assistant and agent over company financial reports, evaluated on FinanceBench

Question answering over **annual and quarterly reports of listed companies** (10-K, 10-Q, 8-K, earnings releases).
It retrieves passages, answers with numbered citations that point to a **document and a page**, **abstains when the
reports do not contain the answer**, and includes a tool-using agent (search, document lookup, calculator), a FastAPI
service and a reproducible evaluation harness.

Evaluation uses the open sample of **FinanceBench** (150 questions written and checked by financial analysts, with the
evidence page for each answer; [Islam et al., 2023](https://arxiv.org/abs/2311.11944)), plus unanswerable questions written by hand.
Questions need information extraction from long PDFs, tables and numerical reasoning: a hard, realistic test for a small model.

> **Status:** the code and the test suite are complete. Numbers in the *Results* section are filled in only after running
> the evaluation (see below). Nothing in this README is a measured result unless it is in that table.

## What is in it

| Part | What it does |
|---|---|
| `financebench.py` | Downloads the FinanceBench questions, the document list and the PDFs; page-level text extraction (PyMuPDF) |
| `chunking.py`, `index.py` | Word windows **inside each page** (every chunk keeps its page number); sentence-transformers embeddings or a dependency-free hashing baseline; exact cosine index |
| `retrieve.py` | Dense search, BM25 and reciprocal rank fusion (hybrid); optional restriction to some documents |
| `rag.py` | Prompt with numbered sources (report + page), citation parsing and validation, abstention detection |
| `agent.py`, `tools.py` | Agent that emits one JSON action per turn: `list_documents`, `search_corpus`, `get_document_info`, `calculate` (safe arithmetic) |
| `llm.py` | One interface, three backends: local Hugging Face model in 4-bit, any OpenAI-compatible server, scripted fake for tests |
| `eval/` | Gold dataset format, metrics (document and page hit, number match, abstention...), draft-question generator, comparison of configurations |
| `api.py` | FastAPI service: `GET /health`, `POST /ask` |

```
FinanceBench ──► fetch: questions + PDFs ──► pages ──► chunks (with page) ──► embeddings ──► VectorIndex ─┐
                                                                                                         ├─► Retriever (dense + BM25 + RRF,
question (+ optional report) ────────────────────────────────────────────────────────────────────────────┘    all reports or one)
                                   ┌──────────────────────────────────────────────────────────────────────────────┘
                                   ▼
                    RAG: prompt with [1]..[k] sources (report, page) ──► LLM ──► answer + validated citations / abstention
                    Agent: LLM ⇄ tools (list, search, document info, calculator) ──► final answer + supported sources
```

## Quick start on Google Colab (free GPU)

Open `notebooks/colab_run.ipynb`, or:

```bash
pip install -q -e ".[ml]"
python -m finrag fetch                 # FinanceBench questions + the ~84 reports they need (--max-docs 20 for a quick run)
python -m finrag ingest                # read pages, chunk, embed, index
python -m finrag make-gold             # FinanceBench questions on the indexed reports -> data/eval/gold.jsonl
python -m finrag ask "What is the FY2018 capital expenditure amount (in USD millions) for 3M?" --doc 3M_2018_10K
python -m finrag ask --mode agent "By how much did 3M's capital expenditure change between FY2017 and FY2018, in percent?"
```

Your own PDFs work too: copy them into `data/pdf/` and run `ingest` (the file name becomes the document id).

The default model is a 7B instruct model loaded in 4 bit (it fits a free Colab T4). It is a config value:
`llm_model: mistralai/Mistral-7B-Instruct-v0.3` or `Qwen/Qwen2.5-7B-Instruct`. **Check the model licence before you use or publish results.**

## Local run without GPU or downloads

```bash
pip install -e ".[api,dev]"
pytest                      # 44 tests: retrieval, page-aware chunking, citations, agent loop, metrics, API (fake LLM, hashing embedder)
```

Set `embedder: hashing` and `llm_backend: fake` to exercise the whole pipeline offline.

## Evaluation

The gold set is a JSONL file (`eval/dataset.py` documents the format):

```json
{"id": "financebench_id_03029", "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? ...",
 "answer": "$1577.00", "doc_ids": ["3M_2018_10K"], "pages": [["3M_2018_10K", 59]], "type": "factual", "source": "financebench"}
{"id": "u01", "question": "What was Boeing's employee turnover rate in Italy in FY2022?", "answer": "",
 "doc_ids": ["BOEING_2022_10K"], "type": "unanswerable", "source": "mine"}
```

`make-gold` converts the FinanceBench questions; the unanswerable questions are written by hand.
`eval-generate` drafts extra questions from random chunks: **every draft must be reviewed and corrected by hand** (`reviewed: true`).

```bash
python -m finrag eval --limit 40                       # 40 answerable questions at random (fixed seed) + all unanswerable
python -m finrag eval --mode agent --limit 20 --variant "hybrid, per_doc"
```

Variants: `dense, shared` · `hybrid, shared` · `hybrid, per_doc`. *Shared* searches all reports at once (a company-wide assistant);
*per_doc* searches only the report the question is about (FinanceBench's single-document setting).

Metrics (cheap and transparent; read the answers too):

| Metric | Meaning |
|---|---|
| `doc_hit`, `page_hit` | a retrieved chunk comes from the gold report / lies on the gold evidence page |
| `number_match` | the answer contains the gold figure within 1% (lenient on million/billion and ratio/percent) |
| `f1` | token overlap with the gold answer (screening only: it under-scores correct paraphrases) |
| `citation_precision`, `invalid_citation_rate` | cited reports that are gold reports; answers citing sources that do not exist |
| `false_abstention`, `abstention_accuracy` | abstains on an answerable question; abstains on an unanswerable one |

### Results

_To be filled in after running the evaluation. Report the number of questions, the model, the embedder, the chunk size and the date.
Include at least: dense vs hybrid, shared vs per-document search, and a failure analysis with three real examples._

| variant | doc_hit | page_hit | number_match | citation_precision | false_abstention | abstention_accuracy |
|---|---|---|---|---|---|---|
| dense, shared | | | | | | |
| hybrid, shared | | | | | | |
| hybrid, per_doc | | | | | | |

## API and Docker

```bash
uvicorn finrag.api:app_factory --factory --port 8080       # FINRAG_CONFIG=configs/default.yaml
curl -s localhost:8080/ask -H 'content-type: application/json' \
  -d '{"question": "What was the FY2018 capital expenditure?", "doc_ids": ["3M_2018_10K"]}'
```

The image does not need a GPU: with `llm_backend: openai_compat` it calls an external LLM server (vLLM, Ollama, llama.cpp...).

```bash
docker build -t finrag .
docker run -p 8080:8080 -v "$PWD/data:/app/data" -e FINRAG_CONFIG=configs/docker.yaml finrag
```

If the index was built with a sentence-transformers model, build the image with `--build-arg EXTRAS=api,ml` (large) or rebuild the index with `embedder: hashing`.

## Design decisions

- **Page-aware chunks.** Chunks never cross a page, so retrieval can be scored against the evidence page and every answer points to a page a human can open.
- **Hybrid retrieval.** Line items, tickers and figures ("capital expenditure", "PP&E", "1,577") are matched lexically; embeddings alone often miss them. Fusion is rank based, so scores of different scales need no calibration.
- **Shared vs per-document search.** Searching one report is easier; a company-wide assistant must first find the right report. Both are measured.
- **Citations are checked, not trusted.** The model cites `[n]`; the code maps each `n` back to a report and page and flags numbers that do not exist. The agent's cited ids are compared with the ids it actually saw in tool results.
- **Abstention is a measured behaviour.** A system that never says "I don't know" looks good on answerable questions and is dangerous in finance, so the harness scores it separately.
- **Safe calculator.** Ratios and growth rates go through an AST whitelist, never `eval`; the model is told not to do arithmetic in its head.
- **A deterministic hashing embedder and a fake LLM** make the whole code path testable in CI without downloads.

## Limitations

- PDF text extraction flattens tables: row labels and figures lose their alignment, so table questions are the hardest.
- Exact cosine search over tens of thousands of chunks; a vector database is not needed at this scale.
- `number_match` checks only the first figure of the gold answer; `f1` under-scores paraphrases. Read the answers.
- A 7B model in 4 bit follows the agent's JSON protocol imperfectly: failures are logged as `invalid` steps.

## Data and licences

Questions, answers and evidence pages: the open FinanceBench sample ([patronus-ai/financebench](https://github.com/patronus-ai/financebench),
paper [arXiv:2311.11944](https://arxiv.org/abs/2311.11944)); cite it if you use it, and check its licence on the
dataset card before any non-personal use. The reports are public company filings. This repository contains **no questions, report text or PDFs**: they are
downloaded at run time into `data/` (git-ignored). Code: MIT (see `LICENSE`). Models: check each model's licence.
