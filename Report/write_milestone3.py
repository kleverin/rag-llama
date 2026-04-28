import json

with open(r"k:\Kevin\rag-llama\Report\Project Template.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── 3.1 Project Solution (cell 41) ─────────────────────────────────────────
nb['cells'][41]['source'] = [
"""---

### System Overview

The final project solution is a fully local Retrieval-Augmented Generation (RAG) diagnostic assistant deployed as a web application accessible from any device on the factory floor local area network (LAN). The system ingests historical manufacturing data — including MTConnect machine logs, job and shift Excel reports, employee records, and acoustic sensor WAV files — embeds all of it into a persistent vector database, and exposes a conversational chat interface where operators and engineers can ask plain-language questions about machine performance, production history, and fault events.

<!-- CONTENT TO INSERT: Screenshot of the chat UI showing the boot/loading screen and the main chat interface, on both desktop and mobile views -->

### How It Addresses the Problem

The core problem identified at the start of this project was that Kirby Risk had large volumes of machine-generated data stored across dozens of individual Excel files with no automated tool to query or interpret them. Manual analysis was time-consuming and impractical at scale.

The system directly addresses this by:

1. **Unifying all data sources** — MTConnect logs, shift reports, job orders, employee records, and acoustic sensor data are all ingested into a single searchable vector database. A question like "what happened on April 28th during the night shift" retrieves context from all relevant sources simultaneously.
2. **Natural language interface** — No SQL knowledge or file navigation is required. Operators type a question the same way they would ask a colleague.
3. **Local and private deployment** — The entire system — the LLM, the embedding model, and the vector database — runs on a single machine inside the company network. No manufacturing data is ever sent to an external server or cloud service.
4. **Persistent session context** — The system maintains conversation history within a session, allowing follow-up questions such as "what about the previous week?" or "which operator was on shift then?"

### System Architecture

The pipeline consists of four stages:

| Stage | Component | Technology |
|-------|-----------|------------|
| **Ingestion** | Load and chunk all data sources | `ingest.py`, Pandas, librosa |
| **Embedding** | Convert text chunks to vector representations | `nomic-embed-text` via Ollama |
| **Storage** | Persist vectors for retrieval | ChromaDB (`./chroma_db`) |
| **Query** | Retrieve relevant chunks and generate answer | `llama3.2` via Ollama + LlamaIndex RAG (top-k = 5) |

<!-- CONTENT TO INSERT: System architecture flowchart with the following stages connected by arrows:
  [Data Sources: .xlsx / .csv / .wav / .md]
        down-arrow
  [ingest.py — chunking and feature extraction]
        down-arrow
  [nomic-embed-text (Ollama) — embedding generation]
        down-arrow
  [ChromaDB — persistent vector store (chroma_db/)]
        up-arrow (top-5 retrieval)
  [LlamaIndex RAG engine — condense_plus_context mode]
        down-arrow (augmented prompt)
  [llama3.2 LLM (Ollama)]
        down-arrow
  [FastAPI server — server.py — port 8000]
        down-arrow
  [Chat UI — browser or phone at http://<ip>:8000]
-->

<!-- CONTENT TO INSERT: Screenshots or a short screen recording of 3 to 4 live example queries and their responses from the deployed chat UI. Suggested queries:
  1. "Were there any high spindle load events on April 28th?"
  2. "Who were the employees on the night shift during May 27th?"
  3. "What does a bearing fault sound like on the MB4000?"
  4. "How many parts were completed in the last week of May?"
-->

### Limitations

| Limitation | Description |
|-----------|-------------|
| **Data discontinuity** | The MTConnect logs, shift reports, job orders, and audio files were collected independently and do not share a consistent timestamp format or a common join key. The system can retrieve documents from each source but cannot automatically correlate them the way a relational database would, so cross-source answers are sometimes incomplete. |
| **Audio features are per-file summaries** | The acoustic sensor pipeline extracts aggregate statistics — RMS, kurtosis, crest factor, and others — across each entire WAV file. A fault that lasts only a few seconds inside a longer recording gets averaged with the surrounding normal operation, potentially masking the event. Windowed feature extraction would be needed to detect and pinpoint within-file anomalies. |
| **Response latency** | Even on a laptop with 32 GB RAM, generating a response takes 15 to 45 seconds depending on query complexity. This is a fundamental property of running a 3-billion-parameter model locally on CPU and cannot be resolved through software tuning alone. |
| **Retrieval limited to top-5 chunks** | The system retrieves only the 5 most semantically similar text chunks per query. For questions that span a long time range or require aggregating many records, important context may not make it into the top 5 and the answer will be incomplete. |
| **No real-time ingestion** | New shift reports or sensor recordings must be manually placed in the data directory and `ingest.py` must be re-run to update the knowledge base. There is no automated pipeline that picks up new files as they are created. |

---"""
]

# ── 3.2 Anticipated Benefits and Adoption (cell 43) ────────────────────────
nb['cells'][43]['source'] = [
"""---

### Immediate Benefits

The system provides three immediate operational improvements over the current manual workflow:

1. **Instant historical lookup** — An operator or engineer can ask about a specific date or event and receive a consolidated answer in under a minute, instead of manually searching through dozens of Excel files across different directories.
2. **Cross-source querying** — Questions that require linking shift data, employee records, and machine logs — such as "who was on shift when the last spindle overload occurred?" — are answered in a single conversational query.
3. **Accessible on any device** — Because the interface is a web app served on the LAN, operators can use it from a phone on the floor or from a laptop in the office without installing any software.

### Path to Permanent Adoption

During a visit to the Kirby Risk plant, company representative Eric described the AI capability he ultimately wants: a system that can **automatically recommend production schedules** when an urgent new job is received. Given a new part's design specifications and the current machine queue, the AI would estimate machining time, identify which machines are appropriate, and suggest when and where to slot the new job — all derived automatically from the part data and historical production records.

Our project is the foundational first step toward that vision. It demonstrates that:
- Historical job and machining data can be structured, embedded, and made conversationally queryable
- A fully local, private deployment is technically feasible without cloud infrastructure
- A simple chat interface is a practical interaction model for floor operators

The full scheduling system Eric described would require extending the current work with the following additions:

| Required Extension | Description |
|-------------------|-------------|
| **Machining time estimator** | A model trained on historical `Job_Orders` and `Part_Details` records to predict how long a new part will take on each machine type |
| **Real-time machine availability** | A live MTConnect feed rather than historical logs, so the system knows the current state of every machine before making a recommendation |
| **Constraint satisfaction layer** | Logic that enforces hard scheduling constraints — shift hours, machine capacity, tool availability — alongside the LLM recommendations to ensure the output is actually feasible |

### Challenges for Permanent Adoption

| Challenge | Description |
|-----------|-------------|
| **Dedicated hardware** | For production use, the system should run on a dedicated on-site server rather than a personal laptop. A machine with a discrete GPU would reduce response latency from 30 to 45 seconds down to under 5 seconds. |
| **Ongoing data ingestion** | A process must be established to automatically ingest new shift reports and MTConnect logs as they are generated. Currently this is a manual step. |
| **Data consistency standards** | The quality of answers depends directly on how consistently source files are formatted. Standardizing shift report generation will improve system reliability over time. |
| **User trust and onboarding** | Operators need to understand what kinds of questions the system can and cannot reliably answer, to avoid over-relying on it for critical production decisions. |

---"""
]

# ── 3.3 Human Factors and Impact (cell 45) ─────────────────────────────────
nb['cells'][45]['source'] = [
"""---

### Impact on Direct Users

**Machine Operators**

Operators gain access to historical context they previously could not retrieve without supervisor involvement. Instead of waiting for a shift lead to pull up old records, an operator can ask the system directly about past events on a specific machine from their phone on the floor. The conversational interface requires no technical training — if the operator can type a question, they can use the system.

The primary behavioral change is a shift from reactive to informed: operators can check the system before starting a job on a machine with a flagged maintenance history, rather than discovering a problem mid-operation.

**Production Engineers and Supervisors**

For engineers, the largest impact is time savings on root cause analysis. Correlating a defective part with shift timing, operator, machine load history, and acoustic data currently requires manually joining multiple files. The system reduces this to a single query.

Supervisors benefit from the workforce traceability capability — cross-referencing part failure events with who was on shift — which supports targeted training decisions and quality accountability without requiring manual file auditing.

### Changes to the Nature of Work

| Role | Before the System | After the System |
|------|------------------|-----------------|
| **Operator** | Checks physical logbook or asks supervisor for machine history | Queries system directly from phone on the factory floor |
| **Engineer** | Manually opens multiple Excel files to identify failure patterns | Issues a natural language query and receives a consolidated answer |
| **Supervisor** | Manually compiles shift reports for management review | Asks the system for summaries over any date range |
| **Maintenance technician** | Schedules preventive maintenance on fixed calendar intervals | Can query acoustic fault trends to prioritize inspections by actual measured machine condition |

### Challenges and Risks

1. **Over-reliance** — There is a risk that operators treat LLM output as fully authoritative when the underlying data is incomplete or the model has retrieved an irrelevant chunk. The system must be positioned as a decision-support tool, not a decision-making one. Any fault flags or recommendations should be verified against the actual machine state before action is taken.

2. **Workplace perception of AI** — While the system does not replace any role, there is a natural concern among workers about AI tools in a factory environment. Framing the system as a tool that handles tedious data retrieval — freeing operators and engineers to focus on skilled judgment work — is important for floor-level acceptance.

3. **Acoustic data policy** — The system ingests audio recordings from the factory floor. Although these capture machine sounds rather than conversations, establishing and communicating a clear policy on what is recorded, how long it is retained, and who can access it is good practice for maintaining worker trust.

---"""
]

# ── 3.4 Personal Learnings (cell 47) ───────────────────────────────────────
nb['cells'][47]['source'] = [
"""---

**KBM (Kevin Biju Mathew):**

Coming into this project, my main expectation was that the system would give near-instant responses — I have a laptop with 32 GB of RAM and decent CPU specs, so I assumed hardware would not be a bottleneck. In reality, generating a single response still takes 15 to 45 seconds, and I came to understand that this is a fundamental property of running a multi-billion-parameter language model locally on CPU. The hardware ceiling is real, and it changed how I think about the feasibility gap between local and cloud-hosted AI systems at an industrial scale.

The bigger shift in my thinking was about data quality. I came in assuming the LLM would be intelligent enough to make sense of data regardless of how it was structured or labeled — that the model's language understanding would compensate for messy inputs. The reality was the opposite: the system is only as good as the labeling and organization of the data going into it. When data sources lack shared keys, consistent timestamps, or clear column names, the retrieval step fails before the LLM ever gets a chance to reason. I also learned that it is not enough to have data — the different data sources need to have meaningful correlations with each other, otherwise the system retrieves isolated facts that cannot be connected into a useful answer. For a RAG system specifically, data quality and inter-source correlation are the most critical engineering decisions in the entire project.

One thing I am genuinely proud of is the front-end chat interface. It was important to me that the system looked and felt professional — something operators could actually see themselves using on a phone on the factory floor rather than a rough prototype. The final UI with the animated boot loader, the HUD-style dark design, and the mobile responsiveness is something I built from scratch, and I think it reflects what interacting with an AI tool in an industrial setting should feel like.

Working with Eric, Asif, and Yuseop was one of the most valuable parts of this project. Their feedback was direct and grounded in real operational needs rather than abstract suggestions. Eric's description of the scheduling assistant he ultimately wants gave the project a concrete future direction that made the work feel meaningful beyond a course deliverable. Asif and Yuseop were especially helpful in pointing out where our approach was misaligned with how data is actually generated on the floor, which saved us from going further down the wrong path. Interacting with people from the company shifted my perspective on how academic AI projects and real industry needs differ — practical constraints like privacy, hardware cost, and user simplicity matter far more in the field than model benchmark scores.

---"""
]

# ── 3.5 Next Steps and Recommendations (cell 49) ───────────────────────────
nb['cells'][49]['source'] = [
"""---

### Immediate Steps to Make the System Fully Functional

| Step | Description |
|------|-------------|
| **Standardize data formatting** | Establish a consistent column naming convention and timestamp format across all shift report Excel files so cross-source correlation becomes reliable. This is a data governance step that does not require changes to the system code itself. |
| **Automated ingestion pipeline** | Set up a file watcher (e.g., using Python's `watchdog` library) that automatically re-runs the relevant portion of `ingest.py` whenever a new `.xlsx`, `.csv`, or `.wav` file is added to the data directory, removing the current manual step. |
| **Continuous audio feature extraction** | Instead of computing whole-file summary statistics per WAV recording, implement a sliding-window approach (e.g., 1-second windows with 50% overlap) so that fault events occurring within a longer recording are not averaged out by the surrounding normal operation. This would allow the system to pinpoint when within a shift a fault occurred, not just whether the file contained one. |
| **Dedicated server hardware** | Deploy the system on a dedicated on-site machine. Even a mid-range NVIDIA GPU with 8 GB VRAM would reduce response latency from 30 to 45 seconds down to under 5 seconds, making the system practical for active floor use. |

### Recommendations Toward the Scheduling Vision

Based on feedback from Eric during the plant visit, the highest-value extension of this system is an intelligent **production scheduling recommendation engine** — an AI that, given an urgent new part order, can recommend which machines to use and when to schedule the job based on historical production data.

The recommended development path:

1. **Use the current RAG system as the knowledge foundation** — The pipeline already contains historical job orders, part details, machining times, and machine utilization data that would serve as the retrieval base for scheduling recommendations.
2. **Build a machining time estimator** — Train a regression model on `Part_Job_Enriched` and `Job_Orders` data (part type and complexity mapped to historical average machining time per machine), and integrate its predictions as retrievable context.
3. **Add a constraint satisfaction layer** — Scheduling requires hard constraints such as shift hours, machine capacity, and tool availability that an LLM cannot enforce reliably on its own. A constraint module should validate every LLM-generated schedule before it is shown to the operator.
4. **Connect to a live MTConnect stream** — Schedule recommendations must reflect the current real-time state of the machines rather than only historical records.

### Additional Recommendations

- **Increase retrieval depth for time-range queries** — Raise `similarity_top_k` from 5 to 10 to 15 for queries that span long date ranges, or implement a query classifier that adjusts retrieval depth automatically based on the type of question detected.
- **Add a user feedback mechanism** — Allow operators to flag responses as helpful or incorrect. This data can be used to identify gaps in the ingested data and improve prompt templates over time.
- **Define and enforce a data entry standard** — A one-page internal document specifying how shift reports must be formatted and saved would ensure all future records are immediately compatible with the ingestion pipeline without preprocessing.

---"""
]

# ── Structured Technical Report (cell 52) ──────────────────────────────────
nb['cells'][52]['source'] = [
"""---

## Structured Technical Report

### 1. Components of the Project Solution

The system is composed of five software components that work together in sequence:

| Component | File | Role |
|-----------|------|------|
| **Ingestion pipeline** | `ingest.py` | Reads all data source files, converts rows and audio recordings to text documents, generates vector embeddings, and stores them in ChromaDB |
| **Audio processor** | `audio_processing.py` | Extracts 10 acoustic features from each WAV sensor recording and serializes them into a retrievable natural-language text chunk |
| **Vector database** | `chroma_db/` (directory) | Persistent storage for all embedded documents — survives server restarts and only needs to be rebuilt when new data is added |
| **API and chat server** | `server.py` | FastAPI application that loads the vector store, handles conversational queries through `llama3.2`, and serves the chat web UI |
| **Diagnostic knowledge base** | `mb4000_troubleshooting.md` | Structured fault library for the MB4000 machine — ingested alongside operational data so the LLM can cross-reference sensor readings against known fault signatures |

<!-- CONTENT TO INSERT: Architecture diagram showing the following data flow connected by labeled arrows:
  [Data files: .xlsx, .csv, .wav, .md in the Data/ folder]
    -> ingest.py (chunking and feature extraction)
    -> nomic-embed-text via Ollama (embedding generation)
    -> ChromaDB chroma_db/ (persistent vector store)
    <- LlamaIndex RAG engine retrieves top-5 chunks
    -> llama3.2 via Ollama (response generation)
    -> FastAPI server.py on port 8000
    -> Browser or phone chat UI at http://<ip>:8000
-->

**External local services required:**

| Service | Purpose |
|---------|---------|
| **Ollama** | Runs both `llama3.2` (language model) and `nomic-embed-text` (embedding model) as background local processes |

<!-- CONTENT TO INSERT: Photo of the physical machine (laptop or desktop PC) running the server, ideally shown connected to the factory network -->

---

### 2. Step-by-Step Installation and Configuration

#### Prerequisites

- Python 3.10 or higher installed on the machine
- At least 16 GB RAM (32 GB recommended)
- At least 10 GB free disk space (for downloaded models and the vector database)
- The server machine must be connected to the same local network as any phones or laptops that will access the chat UI

---

#### Step 1 — Install Ollama

**Windows:**
1. Go to `https://ollama.com/download` and download the Windows installer
2. Run the `.exe` installer and follow the on-screen prompts
3. Ollama will start automatically as a background service after installation
4. Open any terminal and confirm it works: `ollama --version`

**Ubuntu:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
# Start the Ollama service
ollama serve &
```

---

#### Step 2 — Download the required AI models

Run these commands in any terminal (same for Windows and Ubuntu). Both models must be fully downloaded before ingestion or the server will fail to start.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Total download size is approximately 4 GB. Allow 5 to 20 minutes depending on connection speed.

---

#### Step 3 — Copy the project files to the deployment machine

The project folder must contain the following structure:

```
rag-llama/
├── ingest.py
├── server.py
├── audio_processing.py
├── mb4000_troubleshooting.md
├── requirements.txt
├── .env
└── Data/
    ├── Job_enriched_ar.xlsx
    ├── file_with_utc_shift_times_Ar.xlsx
    ├── pivoted_metrics_250428-0513_filtered (1).csv
    ├── pivoted_metrics_250527-0614_filtered (1).csv
    └── Sounds/
        ├── 250428/      (WAV sensor recordings - Apr 28-29)
        └── 250527/      (WAV sensor recordings - May 27-28)
```

---

#### Step 4 — Create the Python virtual environment and install dependencies

**Windows** (run in Command Prompt or PowerShell inside the project folder):
```cmd
python -m venv iiot
iiot\Scripts\activate
pip install -r requirements.txt
```

**Ubuntu** (run in Terminal inside the project folder):
```bash
python3 -m venv iiot
source iiot/bin/activate
pip install -r requirements.txt
```

Installation of all packages takes approximately 3 to 10 minutes.

---

#### Step 5 — Configure the environment file

Open the `.env` file in any text editor and confirm or update the values:

```
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PATH=./chroma_db
DATA_PATH=./Data
KB_PATH=./
CSV_SAMPLE_EVERY=60
PORT=8000
```

`DATA_PATH` must point to the folder containing the Excel and CSV files. `CSV_SAMPLE_EVERY=60` means one row is kept per 60 rows from each CSV file to control index size. Reduce this number (e.g., to 10) for more complete coverage at the cost of a longer ingest time.

---

#### Step 6 — Run the ingestion pipeline

This step is only required once, or when new data files are added to the Data folder.

**Windows (with virtual environment active):**
```cmd
python ingest.py
```

**Ubuntu (with virtual environment active):**
```bash
python ingest.py
```

The terminal will print progress for every file processed and every embedding batch. With the full dataset (approximately 238,000 documents), ingestion takes **60 to 120 minutes** on a machine without a GPU. When complete, a `chroma_db/` folder will appear in the project directory — this is the populated vector database and must not be deleted unless you intend to re-run ingestion.

<!-- CONTENT TO INSERT: Screenshot of the terminal during ingest.py execution, showing the per-file output lines such as "[xlsx] Job_enriched_ar.xlsx / 'Employees' - 47532 rows (job_data)" and the embedding progress bars -->

---

#### Step 7 — Start the API server

**Windows (with virtual environment active):**
```cmd
python server.py
```

**Ubuntu (with virtual environment active):**
```bash
python server.py
```

When the server is ready, the terminal will print:
```
=== Server ready ===
  Local:  http://localhost:8000
  Phone:  http://192.168.x.x:8000
  Docs:   http://localhost:8000/docs
```

The `Phone:` address is what users on the same network should open in their browser. Keep this terminal window open — closing it stops the server. On Ubuntu, use `screen` or `nohup` to keep the server running after the terminal is closed.

<!-- CONTENT TO INSERT: Screenshot of the terminal showing the actual "Server ready" output with the LAN IP address visible -->

---

#### Step 8 — Access the chat interface

1. Open any browser on a device connected to the same network as the server
2. Go to `http://<server-ip>:8000` (use the IP shown in the server terminal)
3. The animated loading screen will appear for a few seconds while the system initializes
4. Once the chat window is visible, type any question in the text box at the bottom and press **Enter** or tap **Send**
5. To clear conversation history and start a new session, tap the **New** button in the top-right corner

<!-- CONTENT TO INSERT: Screen recording (15 to 30 seconds) or 3 sequential screenshots showing:
  1. The animated boot/loading screen
  2. The main chat interface with the empty message window
  3. A question being typed and the formatted answer appearing in the chat
-->

---

### 3. Functionality and Test Cases

The system handles three categories of queries. Test each category after installation to confirm the system is working correctly.

#### Category A — Machine diagnostic queries (sourced from knowledge base and MTConnect data)

**Test 1:** Type — *"What does high spindle load indicate on the MB4000?"*
- Source: `mb4000_troubleshooting.md`
- Expected: A structured answer listing causes (tool wear, aggressive parameters, bearing friction, coolant starvation) and diagnostic steps, with bold headings and a numbered or bulleted format

**Test 2:** Type — *"Were there any estop events on April 28th?"*
- Source: MTConnect CSV data
- Expected: One or more timestamps where `Mestop = TRIGGERED` was recorded, along with the spindle load and active tool at those events

#### Category B — Job and shift history queries (sourced from Excel data)

**Test 3:** Type — *"What jobs were running during the night shift on May 27th?"*
- Source: `Job_Orders` and `Employees` sheets from `Job_enriched_ar.xlsx`
- Expected: Job numbers, part types, or shift timing information for the queried window

**Test 4:** Type — *"Who were the employees working when the last overload event was flagged?"*
- Source: `Employees` sheet cross-referenced with MTConnect flags
- Expected: Employee names and shift assignment near the flagged time period

#### Category C — Acoustic fault queries (sourced from processed WAV audio data)

**Test 5:** Type — *"Were there any bearing fault indicators in the audio recordings?"*
- Source: Audio feature chunks from WAV ingestion
- Expected: References to recordings where kurtosis exceeded 4.0 and crest factor exceeded 5.0, with severity labels

<!-- CONTENT TO INSERT: Screenshots of the actual system responses to each of the 5 test cases above, taken from the live running chat UI -->

---

### 4. Sample Dataset and Calibration Queries

The system is pre-loaded with the following data after a full ingestion:

| Dataset | Source File | Volume | Time Coverage |
|---------|------------|--------|---------------|
| MTConnect machine logs | 2 CSV files | ~48,000 sampled rows | Apr 28 to May 13, 2025 and May 27 to Jun 14, 2025 |
| Job and shift reports | `Job_enriched_ar.xlsx` (10 sheets) | ~180,000 rows | Full production history |
| Shift boundary times | `file_with_utc_shift_times_Ar.xlsx` | ~2,000 rows | Shift start and end timestamps |
| Acoustic features | Processed WAV recordings | 902 + 1,261 files | Apr 28 to 29 and May 27 to 28 |
| Diagnostic knowledge base | `mb4000_troubleshooting.md` and guides | 36 sections | MB4000 fault library |

**Calibration queries to verify a correct installation:**

| Query to type | What a correct answer looks like |
|--------------|----------------------------------|
| `"What is the healthy kurtosis range for the MB4000 spindle bearing?"` | Should state 1.5 to 3.5, drawn from the knowledge base |
| `"List the MTConnect execution states and their meanings."` | Should list ACTIVE, STOPPED, INTERRUPTED, READY, UNAVAILABLE with descriptions |
| `"What happened on 2025-04-28 at 10:00?"` | Should return MTConnect data near that timestamp from the April CSV |

If the first two calibration queries return empty or irrelevant answers, the knowledge base was not ingested correctly — check that `mb4000_troubleshooting.md` is present in the project root and re-run `ingest.py`. If the third query returns nothing, confirm that `DATA_PATH` in `.env` points to the correct folder.

<!-- CONTENT TO INSERT: Screenshot of the system correctly answering one of the three calibration queries above -->

---

### 5. Interpreting the Outputs

**What the system answers reliably:**
- Factual lookups present in the ingested data: dates, part counts, employee names, shift assignments, machine states, alarm events
- Diagnostic explanations and fault signatures drawn from the MB4000 knowledge base
- Summaries over a specific short time window when the relevant data has been ingested and is well-labeled

**What the system does not answer reliably:**
- Questions requiring correlation across data sources that do not share a common timestamp or key — for example, linking an audio fault exactly to the employee on shift requires matching timestamps across two datasets, which is not always consistent in the current data
- Questions spanning a very long time window where more than 5 relevant records exist — only the top 5 most semantically similar chunks are retrieved per query, so month-wide summaries may be incomplete
- The current real-time state of any machine — the system only knows what was ingested and does not reflect live machine status

**How to read a response:**

| Signal in the response | What it means |
|----------------------|---------------|
| Cites specific values, timestamps, or names | Answer is likely grounded in retrieved data — high confidence |
| Says "I don't have specific information about..." | The relevant data was not retrieved — try rephrasing with a more specific date or field name |
| Uses vague or generic language about the topic | The query matched the knowledge base rather than actual data — useful for general questions, not for specific event lookups |
| States something that contradicts known facts | Possible hallucination — the model generated a plausible-sounding answer from an unrelated retrieved chunk. Verify against the source files before acting on the response. |

---"""
]

with open(r"k:\Kevin\rag-llama\Report\Project Template.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done - 6 cells updated successfully")
