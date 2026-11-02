# Technical decisions and open issues

This log is for the job-portfolio write-up. Update it at the end of each phase.

## Phase 1 — project skeleton and LLM connectivity (2026-10-29)

### Product goal for this phase

Prove that a local Python project can load configuration safely and call a real
Chat Completions API. Do not build document upload, retrieval, or UI yet.

### Decisions

| ID | Decision | Why |
| --- | --- | --- |
| D1 | Incremental delivery, skeleton first | Reduces debugging surface for a beginner; each phase has a runnable checkpoint. |
| D2 | OpenAI-compatible HTTP API (`openai` SDK) | One client can talk to OpenAI, DeepSeek, DashScope compatible mode, Moonshot, etc. RAG comes later. |
| D3 | Secrets only in environment / `.env` | Keys must not live in source code or git. `.env` is gitignored; `.env.example` is the template. |
| D4 | Model name is configuration, not a constant in business code | The model must match what the key can actually call. Hardcoding a famous model name would fake success. |
| D5 | `src/` layout + `scripts/check_llm.py` | Keeps importable library code separate from a one-shot connectivity probe. |
| D6 | Offline unit tests + optional live probe | pytest can pass without network. Live API check is explicit so quota/auth failures are visible. |
| D7 | Python 3.13 venv from Anaconda `python` | Machine has 3.13 (conda) and 3.14 (`py`). 3.13 is the safer default for current wheels. |

### Not in this phase (intentionally)

- Document upload / parsing
- Chunking, embeddings, vector store
- Citation-grounded answering
- Streamlit UI
- Evaluation set

### Open issues

| ID | Issue | Impact | Next check |
| --- | --- | --- | --- |
| O1 | No `.env` / API key on this machine | Live probe was run and **did not succeed**. Actual output: `[FAIL] config: Missing required environment variables: OPENAI_API_KEY, OPENAI_MODEL.` This is expected, not a fake pass. | Copy `.env.example` to `.env`, fill a real key, rerun `python scripts\check_llm.py` |
| O2 | Provider not chosen yet (OpenAI vs DeepSeek vs DashScope, etc.) | `OPENAI_BASE_URL` and `OPENAI_MODEL` must match that provider | Confirm which key the user actually has |
| O3 | Live probe result unknown until a real key is used | Do not claim the model API works | Record the exact error body if the first live call fails |

### Learning notes (what each piece is for)

- **RAG (later)**: retrieve policy text first, then ask the model to answer only from retrieved snippets.
- **LLM API (now)**: the generation step. Without a working API, later RAG has nothing to call.
- **`.env`**: local secret store. The program reads it; git does not.
- **venv**: isolated package install so this project does not break Anaconda base.

## Phase 2 — local parse and keyword retrieval (2026-10-29)

### Product goal for this phase

Build a citeable knowledge base from local TXT/PDF files and answer “which
snippet is relevant?” without calling a model. This is the retrieve half of RAG.

### Decisions

| ID | Decision | Why |
| --- | --- | --- |
| D8 | Split modules: `parsing` / `chunking` / `retrieve` / `ingest` | Each step can be tested and later replaced (e.g. swap keyword search for vectors). |
| D9 | `pypdf` for PDF text | Pure-Python, no native build, enough for digital PDFs. Not for scans/OCR. |
| D10 | Chunk size 400 characters, overlap 80 | ~one Chinese policy clause; 20% overlap keeps boundary keywords in two chunks. |
| D11 | Chunk inside a page, never across pages | A hit’s `page_number` stays truthful for citations. |
| D12 | Character windows + snap to `。！？；` / newlines | Chinese has no spaces; token-length chunking would be the wrong unit. |
| D13 | Keyword TF×IDF + phrase bonus, stdlib only | Interpretable (“matched 年假, tf=2”). No embedding API, so phase 2 stays offline. |
| D14 | Chinese bigrams + Latin words as tokens | Simple, no jieba model download; misses some word boundaries but is reproducible. |
| D15 | Demo files prefixed `DEMO_` and labeled 虚构 | Portfolio samples must not be mistaken for a real employer’s handbook. |
| D16 | ASCII PDF fixture via a tiny writer | Helvetica cannot embed Chinese; Chinese policies stay `.txt`. PDF parser is still tested with page metadata. |
| D17 | Ingest warnings instead of aborting the corpus | One damaged PDF should not hide the other policies. |

### Known limits

- Scanned PDFs and image-only pages yield no text (no OCR).
- Encrypted PDFs fail with a clear `DocumentParseError`.
- Keyword search does not understand synonyms (年假 vs 年休假) or paraphrase.
- Long chunks can rank a bit higher because they contain more term repeats; there is no BM25 length normalization yet.
- Live LLM check from phase 1 is still **not** successful (no API key).

### Follow-ups (not this phase)

- Replace or add vector retrieval (embeddings + similarity).
- Grounded generation: send top chunks to the LLM with a “do not invent policy” rule.
- OCR for scans; DOCX support.
- Evaluation set for retrieval hit-rate and citation accuracy.

### Keyword search vs future vector search

| | This phase (keywords) | Later (vectors) |
| --- | --- | --- |
| Match unit | Tokens that literally appear | Embedding similarity of meaning |
| Needs network | No | Usually yes (embedding API) or a local model |
| Explainability | High: list matched terms and TF-IDF | Lower: a cosine score, harder to show “why” |
| Synonyms | Weak | Stronger |
| Role in RAG | Cheap, testable baseline | Better recall for natural questions |
