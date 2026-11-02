# HR Knowledge Copilot

Enterprise HR / admin policy Q&A assistant (portfolio project).

**Current phase: 2 — local document parse + keyword retrieval (offline).**  
RAG answers, Streamlit, login, and evaluation are not implemented yet.  
Phase 1 LLM live check still needs a real API key (not configured on this machine).

## What this phase proves

1. TXT and PDF files can be parsed with filename / page / paragraph metadata.
2. Text is split into overlapping chunks that stay citeable.
3. A keyword search baseline returns the top 3 chunks with an explainable score.
4. The whole path runs offline; it does not call the model API.

## Windows setup

```bat
cd C:\Users\28460\Documents\Projects\hr-knowledge-copilot
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set a real `OPENAI_API_KEY`, plus the `OPENAI_BASE_URL` and `OPENAI_MODEL` that belong to **that** key.

## Commands

Offline tests (no network, no key required):

```powershell
cd C:\Users\28460\Documents\Projects\hr-knowledge-copilot
.\.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Expected: all tests passed.

Offline search (demo data, no API key):

```powershell
python scripts\search_offline.py "年假如何申请"
python scripts\search_offline.py "报销需要发票吗"
python scripts\search_offline.py "出差住宿标准"
```

Expected: three ranked chunks with source filename, page/paragraph, score, and snippet.  
Empty question should print `[FAIL] query: Empty question...`

Live API check:

```bat
python scripts\check_llm.py
```

Expected if `.env` is valid: `[OK] LLM call succeeded` and a short reply.  
Expected if the key is missing: `[FAIL] config: Missing required environment variables...`

## Layout

| Path | Role |
| --- | --- |
| `src/hr_copilot/config.py` | Load and validate env settings |
| `src/hr_copilot/llm.py` | Call the model API (phase 1; still unused in phase 2) |
| `src/hr_copilot/parsing.py` | TXT / PDF extraction |
| `src/hr_copilot/chunking.py` | Overlapping windows + citations |
| `src/hr_copilot/retrieve.py` | Keyword TF-IDF baseline |
| `src/hr_copilot/ingest.py` | Folder → chunks |
| `scripts/check_llm.py` | Live connectivity probe |
| `scripts/search_offline.py` | Offline search entry |
| `data/demo/` | Fictional policy samples |
| `tests/` | Offline unit tests |
| `docs/DECISIONS.md` | Product/tech decision log |
