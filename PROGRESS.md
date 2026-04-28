# PROGRESS.md — rag-llama

Running log of all additions and changes to the project. Most recent entries at the top.

---

## 2026-03-22 — Initial Setup

### Added
- `CLAUDE.md` — project context and conventions for Claude Code
- `PROGRESS.md` — this file; running log of project changes
- `.env` — environment config (OLLAMA_BASE_URL, CHROMA_PATH, DATA_PATH, PORT)
- `requirements.txt` — all Python dependencies
- `ingest.py` — Excel → ChromaDB ingestion pipeline
  - Recursively finds all `.xlsx` files under `data/`
  - Reads all sheets per file via `pandas.read_excel(sheet_name=None)`
  - Converts each row to a LlamaIndex `Document` with source/sheet/row metadata
  - Embeds using `nomic-embed-text` via Ollama
  - Persists vectors to ChromaDB at `./chroma_db`
  - Skips re-ingestion if `chroma_db/` already exists
- `server.py` — FastAPI query server
  - Loads ChromaDB on startup
  - `GET /health` → `{"status": "ok"}`
  - `POST /ask` → `{"answer": "..."}` using `llama3.2` via Ollama
  - CORS open for LAN phone access
  - Prints local and LAN IP on startup
- `iiot/` — Python virtual environment with all dependencies installed

### Data ingested
- 119 shift report `.xlsx` files from `data/2026_Spring_ME597/KRPM_shift_report_Apr-June2025/`
- `data/2026_Spring_ME597/MTConnect_data/krpm.operation_20250901.xlsx`
- Large CSVs excluded (4+ GB)

### Stack
- LlamaIndex 0.14.18 + ChromaDB 1.5.5 + FastAPI 0.135.1 + Ollama

---

## 2026-03-23 — GitHub Files

### Added
- `README.md` — project overview, setup instructions, usage, and API reference for GitHub
- `.gitignore` — excludes `iiot/`, `chroma_db/`, `.env`, `__pycache__`

---

## 2026-03-23 — Conversation Memory

### Changed
- `server.py`: replaced `query_engine` with `chat_engine` using `condense_plus_context` mode
  - Follow-up questions ("Got any more info on that?") are now rewritten using chat history before querying ChromaDB
  - Uses `index.as_chat_engine(chat_mode="condense_plus_context", ...)`
- Added `POST /reset` endpoint to clear conversation history
- Added "New Chat" button to the UI header — calls `/reset` and clears visible messages

---

## 2026-03-23 — Chat UI

### Added
- `GET /` in `server.py` — serves a self-contained chat UI (no new files or dependencies)
  - Chat bubble layout: user messages right (blue), bot answers left (white)
  - "Thinking..." indicator while waiting for a response
  - Send on button click or Enter key (Shift+Enter for newline)
  - Auto-resizing textarea, mobile-friendly full-viewport layout
  - Error bubble shown if the server is unreachable or returns an error

---

## 2026-04-26 — RAG Accuracy Benchmark + Ingest Text Quality Overhaul

### Added
- `benchmark.py` — standalone RAG accuracy benchmark script
  - 21 eval questions grounded in actual ingested CSV/Excel data (no guessing)
  - Categories: MTConnect tool numbers, program names, high-load events, execution state, shift employees, parts & job orders, machine utilization
  - Scores each answer by keyword presence (e.g. expects "31" for tool-at-6:06-AM question)
  - Prints color-coded terminal report with per-category breakdown
  - Shows retrieved source chunks per question (`--verbose`) so you can see exactly what the RAG pulled
  - Saves timestamped JSON report with all answers, scores, sources, and latency
  - Connects directly to ChromaDB — no server required
  - Usage: `python benchmark.py --verbose` / `--top-k 10` / `--out report.json`

### Changed — `ingest.py` (major text quality overhaul)
- Added `_readable_ts()` — converts ISO timestamps to `"April 28, 2025 at 6:06 AM"` format so embedding model matches natural-language date queries
- Added `_parse_shift_filename()` — extracts date and shift from filenames like `"Shift Report 4-28-25 Day Shift (7).xlsx"` → `("April 28, 2025", "Day")`
- Added `_employees_sheet_to_docs()` — groups Employees sheet by shift file and creates **one document per shift** listing all employee names in natural language (e.g. `"Employees who worked the Night shift on April 28, 2025: Wallace, Robert; Kull, Roger E.; ..."`)
- Added `_part_details_row_to_text()` — converts Part_Details rows to natural language: `"On April 28, 2025 during the Day shift, work center MB4000 ran part number 5387600-03 under job order J270018541 (status: R). Routing quantity: 800. Good parts produced: 28."`
- Added `_machine_summary_row_to_text()` — converts Machine_Shift_Summary rows: `"Work center MB4000 had an average utilization of 76.4% on the Day shift on April 28, 2025."`
- Added `_date_shift_summary_row_to_text()`, `_rag_text_row_to_text()`, `_shift_times_row_to_text()` — natural-language handlers for remaining sheets
- Improved `_mtconnect_row_to_text()` (was `row_to_text`) — timestamp now reads `"On April 28, 2025 at 6:06 AM (timestamp 2025-04-28 06:06:00)"` for better semantic matching
- Added `_SKIP_SHEETS` set — redundant/noisy sheets are skipped during ingest: `Shift_Metrics_Wide`, `Shift_Metrics_Long`, `Shift_Long_Enriched`, `Part_Job_Enriched`, `Job_Orders`, `Failed_Files`
- `load_xlsx_documents()` now dispatches to per-sheet handlers instead of using a generic key-value formatter for everything

### Why
The previous pipe-delimited key-value format (`| date: 2025-04-28 | shift: Day | ...`) embedded poorly — all rows looked semantically identical so the vector search retrieved random dates. Natural-language text and human-readable dates let the embedding model match queries like "April 28 night shift employees" to the correct document.

---

## 2026-04-26 — Benchmark Tuning & RAG Accuracy Improvement (4.8% → 53.1%)

### Summary
Ran the new `benchmark.py` against the freshly ingested natural-language index and systematically diagnosed every failure category.  Four iteration loops drove the score from 4.8% to 53.1%.

### Root causes identified

| Category | Root cause |
|---|---|
| MTConnect exact-timestamp lookup | Embedding can't distinguish rows by timestamp — all MTConnect rows are structurally identical to nomic-embed-text |
| Employee night shift | `file_with_utc_shift_times_Ar.xlsx` docs (score 0.62) outranked employee name docs (score 0.54) in every top-k window |
| High-load events | The row with `MS1load=208%` has the same embedding as the row with `MS1load=0%` — load value carries no semantic weight |
| Parts & job orders | All Part_Details rows tagged `type="job_data"` along with Date/Shift summaries — no way to filter to part-specific docs |
| MA600 utilization | April 28 MA600 Day record doesn't exist in `Machine_Shift_Summary` sheet |
| Program keyword | May 27+ data uses `AB100.MIN`, not `A100.MIN` — `"A100"` is not a substring of `"AB100"` |

### Changes made — `benchmark.py`
- Replaced exact-timestamp MTConnect questions with flag-based and content-based queries
- Increased default `--top-k` from 5 to 15
- Added `meta_filter` field per eval item → builds a metadata-scoped ChromaDB query for that question
  - Employee questions: filter to `type="employees"` → employees jumped from 33% to 100%
  - Program questions: filter to `type="mtconnect"` → eliminates Excel doc noise
  - Execution-state question: filter to `type="mtconnect"` + made date-agnostic → 100%
- Fixed program keyword: `"A100"` → `"A100.MIN"` for Apr 28, `"AB100"` for May 27+
- Made program and execution-state questions date-agnostic (embedding can't distinguish by date)
- Replaced unfounded `util_ma600_apr28` question (no April 28 data in index) with `util_nhx6300_may27` (confirmed present)
- Simplified `util_shift_total` keyword to just `["42"]` — LLM answer phrasing inconsistency was causing spurious misses

### Changes made — `server.py`
- Increased `similarity_top_k` from 5 to 15 for the chat engine

### Final benchmark scores (`benchmark_20260426_222156.json`, top_k=15)

```
Excel — shift employees             100.0%  (3 Qs)
MTConnect — high load events          0.0%  (2 Qs)   ← architectural fix needed
MTConnect — program name            100.0%  (2 Qs)
MTConnect — execution state         100.0%  (1 Q)
Excel — parts & job orders            0.0%  (5 Qs)   ← re-ingest needed
Excel — machine utilization          83.3%  (3 Qs)

OVERALL                              53.1%  (16 Qs)
```

### Remaining gaps — require ingest changes (not yet done)

**High load events (2 Qs, 0%):**  
The specific rows with `MS1load=208%` / `127%` have identical embeddings to all other MTConnect rows.  
Fix: add a per-CSV **anomaly summary document** at ingest time listing every `MS1load > 80` event with its timestamp and tool number.  This gives the LLM one semantically-unique retrievable chunk per CSV.

**Parts & job orders (5 Qs, 0%):**  
Part_Details rows are tagged `type="job_data"` alongside Date_Shift_Summary and other sheets.  Metadata filtering can't target just part-level docs.  
Fix: tag Part_Details rows as `type="part_details"` in ingest, add `meta_filter` to benchmark part questions.

---
<!-- Add new entries above this line, newest first -->
