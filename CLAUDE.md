# CLAUDE.md — rag-llama

## Project Overview
RAG (Retrieval-Augmented Generation) application over IIoT/manufacturing data.
Queries are served via FastAPI so they can be called from a phone on the local network.

## Stack
- **LlamaIndex** — RAG pipeline (indexing + querying), `chat_mode="condense_plus_context"`, top-k=15
- **Ollama** — local LLM (`llama3.2`) and embedding model (`nomic-embed-text`)
- **ChromaDB** — persistent vector store at `./chroma_db`, collection name `iiot_rag`
- **FastAPI** — REST API server on `0.0.0.0:8000` with built-in chat UI at `/`
- **Pandas + openpyxl** — Excel ingestion
- **librosa + scipy** — WAV audio feature extraction (audio_processing.py)

## Virtual Environment
All dependencies live in the `iiot/` venv.
```bash
source iiot/bin/activate
```

## Key Files
| File | Purpose |
|------|---------|
| `ingest.py` | Loads `.md` KB, `.xlsx`, sampled `.csv` (+ all flagged events), and `.wav` files from `data/`, embeds into ChromaDB |
| `server.py` | FastAPI server — query routing (parts / employee / audio / general), exposes `/ask`, `/health`, `/reset`, chat UI at `/` |
| `benchmark.py` | RAG accuracy benchmark — 16 eval questions grounded in real CSV/Excel data, scores by keyword match, saves JSON report |
| `audio_processing.py` | Full 10-feature WAV pipeline — load/normalize → extract features → interpret → RAG chunk |
| `mb4000_troubleshooting.md` | Diagnostic knowledge base for MB4000 CNC machine — ingested automatically |
| `.env` | Runtime config (OLLAMA_BASE_URL, CHROMA_PATH, DATA_PATH, KB_PATH, CSV_SAMPLE_EVERY, PORT) |
| `requirements.txt` | Python dependencies |
| `PROGRESS.md` | Running log of changes and additions |

## Data
- `data/Sounds/250428/` — MB6000 WAV recordings for Apr 28–29, 2025 (902 files)
- `data/Sounds/250527/` — MB6000 WAV recordings for May 27–28, 2025 (1261 files)
- `data/pivoted_metrics_250428-0513_filtered (1).csv` — MTConnect data Apr 28–May 13 (sampled)
- `data/pivoted_metrics_250527-0614_filtered (1).csv` — MTConnect data May 27–Jun 14 (sampled)
- `data/Job_enriched_ar.xlsx` and `data/file_with_utc_shift_times_Ar.xlsx` — enriched job/shift data
- `data/2026_Spring_ME597/` — original shift reports and MTConnect xlsx (legacy)
- `chroma_db/` is the persisted vector DB — delete it to force re-ingestion

## Running the App
```bash
source iiot/bin/activate

# First time (or after new data): ingest
python ingest.py

# Start the server
python server.py

# Benchmark RAG accuracy (no server needed)
python benchmark.py --verbose
```

## API Endpoints
| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/` | — | Chat UI (HTML, mobile-friendly) |
| GET | `/health` | — | `{"status": "ok"}` |
| POST | `/ask` | `{"question": "..."}` | `{"answer": "..."}` |
| POST | `/reset` | — | `{"status": "conversation cleared"}` |

## Environment Variables (`.env`)
```
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PATH=./chroma_db
DATA_PATH=./data
KB_PATH=./
CSV_SAMPLE_EVERY=60
PORT=8000
```

## Prerequisites
Ollama must be running with both models pulled:
```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Audio Pipeline (audio_processing.py)
10 features extracted per WAV file:
| Feature | What it detects |
|---------|----------------|
| RMS | Overall energy / loudness |
| Kurtosis | Impulsive spikes — bearing/gear faults |
| Crest Factor | Peak-to-average ratio — impact events |
| Dominant Frequency | Main frequency in Hz |
| Spectral Centroid | Centre of mass of spectrum — rises with wear/friction |
| Spectral Bandwidth | Spread of energy — wide = broadband fault |
| Low-band Energy Ratio | Fraction of energy in 80–500 Hz |
| Harmonic Ratio | Tonal vs noisy — drops with looseness |
| Zero-Crossing Rate | High-frequency chattering |
| MFCC Delta | Rate of spectral change — unstable operation |

Severity levels: `normal` | `warning` | `fault`
Fault flag: kurtosis > 4.0 AND crest factor > 5.0

## Ingest Pipeline (ingest.py)
Processes four data types in order:
1. `.md` / `.txt` files from `KB_PATH` — knowledge base (split on `---` separators)
2. `.xlsx` files from `DATA_PATH` — sheet-specific natural-language text (see below)
3. `.csv` files from `DATA_PATH` — sampled every `CSV_SAMPLE_EVERY` rows **plus** all rows where `MS1load > 80%` or `Mestop == TRIGGERED` are always included regardless of the sample interval
4. After CSV sampling — one `[PROGRAM-SUMMARY]` doc per CSV (full unsampled count of each CNC program) plus a global summary; both stored as `type="mtconnect"` so they surface in program-frequency queries
5. `.wav` files from `DATA_PATH` — processed through `audio_processing.py`

### Metadata types
Every document stored in ChromaDB carries a `type` metadata field used for filtered retrieval:

| Type | Source |
|------|--------|
| `knowledge_base` | `.md` / `.txt` KB files |
| `employees` | `Employees` sheet — one doc per shift |
| `part_details` | `Part_Details` sheet — one doc per row |
| `machine_summary` | `Machine_Shift_Summary` sheet |
| `shift_summary` | `Date_Shift_Summary` sheet |
| `rag_text` | `RAG_Text` sheet |
| `shift_times` | `Sheet1` in shift-times Excel |
| `mtconnect` | CSV rows + program summary docs |
| `audio` | WAV recordings |

MTConnect CSV docs also carry a `flagged` field (`"true"` / `"false"`) indicating whether the row had a high-load or E-stop event.

### Excel Sheet Handling
Each sheet gets a dedicated text generator. Redundant sheets are skipped.

| Sheet | Handler | `type` | Output format |
|-------|---------|--------|---------------|
| `Employees` | `_employees_sheet_to_docs` | `employees` | One doc per shift listing all employee names in plain English |
| `Part_Details` | `_part_details_row_to_text` | `part_details` | `"On April 28, 2025 during the Day shift, work center MB4000 ran part 5387600-03..."` |
| `Machine_Shift_Summary` | `_machine_summary_row_to_text` | `machine_summary` | `"Work center MB4000 had 76.4% utilization on the Day shift on April 28, 2025."` |
| `Date_Shift_Summary` | `_date_shift_summary_row_to_text` | `shift_summary` | Shift-level utilization + quantity summary |
| `RAG_Text` | `_rag_text_row_to_text` | `rag_text` | Uses the pre-formatted `text` column directly |
| `Sheet1` (shift times file) | `_shift_times_row_to_text` | `shift_times` | `"On April 28, 2025, the Day shift ran from 10:00 AM to 8:30 PM at work center 5001."` |
| Skipped | — | — | `Shift_Metrics_Wide`, `Shift_Metrics_Long`, `Shift_Long_Enriched`, `Part_Job_Enriched`, `Job_Orders`, `Failed_Files` |

Timestamps are stored as both ISO (`2025-04-28 06:06:00`) and human-readable (`April 28, 2025 at 6:06 AM`) so embedding queries on either format work.

## Server Query Routing (server.py)
The `/ask` endpoint dispatches to one of four engines before falling back to the general chat engine:

| Keyword match | Engine | ChromaDB filter | Top-k |
|---------------|--------|-----------------|-------|
| "part number", "job order", "good parts", etc. | `parts_engine` | `type=part_details` | 15 |
| "employee", "who worked", "staff", etc. | `employee_engine` | `type=employees` | 15 |
| "audio", "fault", "kurtosis", "rms", etc. | `audio_engine` | `type=audio` | 20 |
| everything else | `chat_engine` | none | 15 |

`parts_engine`, `audio_engine`, and `chat_engine` use a custom QA prompt that requires the LLM to explicitly cite exact values (part numbers, job orders, program names) rather than using pronouns.

## Benchmark (benchmark.py)
16 questions across 7 categories. Each question can specify:
- `meta_filter` — single dict or list of dicts for ChromaDB `ExactMatchFilter`
- `use_prompt` — set `False` for employee questions (list-style answers work better with the default LlamaIndex prompt)

To run:
```bash
python benchmark.py --verbose          # full answers + retrieved chunks
python benchmark.py --top-k 20        # change retrieval depth
python benchmark.py --out report.json  # custom output file
```

After any change to `ingest.py`, delete `chroma_db/` and re-ingest before benchmarking.

## Coding Conventions
- Keep ingestion and serving as separate scripts (`ingest.py` / `server.py`)
- All config via `.env` — no hardcoded paths or URLs
- Print progress during ingestion so the user can follow along
- CORS is open (`*`) — this is intentional for LAN phone access
- ChromaDB collection name: `iiot_rag`
