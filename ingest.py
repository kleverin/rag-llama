"""
ingest.py — RAG Pipeline Ingestion
====================================
Reads .md knowledge base files, .xlsx MTConnect data, and .wav audio files,
embeds everything into ChromaDB via LlamaIndex + Ollama.

Run:
    source iiot/bin/activate
    python ingest.py
"""

import os
import re
import sys
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

from audio_processing import process_wav

load_dotenv()

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
CHROMA_PATH      = os.getenv("CHROMA_PATH",      "./chroma_db")
DATA_PATH        = os.getenv("DATA_PATH",         "./data")
KB_PATH          = os.getenv("KB_PATH",           "./")
CSV_SAMPLE_EVERY = int(os.getenv("CSV_SAMPLE_EVERY", "60"))  # keep 1 row per N rows

print(f"[config]  DATA_PATH        = {DATA_PATH}")
print(f"[config]  KB_PATH          = {KB_PATH}")
print(f"[config]  CHROMA_PATH      = {CHROMA_PATH}")
print(f"[config]  OLLAMA URL       = {OLLAMA_BASE_URL}")
print(f"[config]  CSV_SAMPLE_EVERY = {CSV_SAMPLE_EVERY} rows")

Settings.llm = Ollama(model="llama3.2", base_url=OLLAMA_BASE_URL, request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_BASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_knowledge_base_documents(kb_dir: str) -> list:
    """
    Load all .md and .txt files from kb_dir.
    Splits on '---' separators so each diagnostic section becomes its own chunk.
    """
    docs     = []
    kb_files = [f for f in os.listdir(kb_dir) if f.endswith(".md") or f.endswith(".txt")]

    if not kb_files:
        print(f"  [kb]  No knowledge base files found in {kb_dir}")
        return docs

    for fname in sorted(kb_files):
        fpath = os.path.join(kb_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            sections = [s.strip() for s in content.split("\n---\n") if s.strip()]
            for i, section in enumerate(sections):
                lines = section.split("\n")
                title = next(
                    (l.lstrip("#").strip() for l in lines if l.startswith("##")),
                    f"{fname} section {i+1}"
                )
                docs.append(Document(
                    text=section,
                    metadata={"source": fname, "type": "knowledge_base", "section": title}
                ))
            print(f"  [kb]  {fname} — {len(sections)} sections")

        except Exception as e:
            print(f"  [ERROR] {fpath}: {e}")

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _readable_ts(ts_str: str) -> str:
    """'2025-04-28 06:06:00' → 'April 28, 2025 at 06:06 AM'"""
    try:
        dt = datetime.fromisoformat(str(ts_str).strip())
        return dt.strftime("%B %d, %Y at %I:%M %p").replace(" 0", " ")
    except Exception:
        return str(ts_str)


def _parse_shift_filename(filename: str) -> tuple:
    """
    Extract (date_str, shift_str) from shift report filenames, e.g.:
      'Shift Report 4-28-25 Day Shift (7).xlsx'   → ('April 28, 2025', 'Day')
      'Shift Report 4-28-2025 Nights (7).xlsx'    → ('April 28, 2025', 'Night')
    """
    m = re.search(r"(\d{1,2})-(\d{1,2})-(\d{2,4})", str(filename))
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            date_str = datetime(year, month, day).strftime("%B %d, %Y")
        except Exception:
            date_str = f"{month}/{day}/{year}"
    else:
        date_str = "unknown date"

    fn_lower = str(filename).lower()
    if "night" in fn_lower:
        shift_str = "Night"
    elif "day" in fn_lower:
        shift_str = "Day"
    else:
        shift_str = "unknown"

    return date_str, shift_str


# ─────────────────────────────────────────────────────────────────────────────
# MTCONNECT CSV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _mtconnect_row_to_text(row: pd.Series) -> str:
    """Convert one MTConnect CSV row to natural-language text with readable timestamp."""
    def g(col, default="N/A"):
        return row.get(col, default)

    ts_raw  = str(g("timestamp"))
    ts_human = _readable_ts(ts_raw)

    flags = []
    try:
        load_val = float(g("MS1load", 0) or 0)
        if load_val > 80:
            flags.append("HIGH-SPINDLE-LOAD")
        if load_val > 95:
            flags.append("SPINDLE-OVERLOAD")
    except Exception:
        pass
    for ax, col in [("X", "MX1load"), ("Y", "MY1load"), ("Z", "MZ1load")]:
        try:
            if abs(float(g(col, 0) or 0)) > 70:
                flags.append(f"HIGH-{ax}-AXIS-LOAD")
        except Exception:
            pass
    if str(g("Mestop")).upper() == "TRIGGERED":
        flags.append("ESTOP-TRIGGERED")

    flag_str = " | FLAGS: " + ", ".join(flags) if flags else ""

    return (
        f"[MTCONNECT]{flag_str} On {ts_human} (timestamp {ts_raw}): "
        f"execution={g('Mpexecution')}, mode={g('MS1Mode')}, estop={g('Mestop')}, "
        f"spindle speed={g('MS1speed')} RPM (override={g('MS1ovr')}%), "
        f"spindle load={g('MS1load')}%, "
        f"active tool={g('Mp1CurrentTool')}, program={g('Mpprogram')}, "
        f"part count={g('Mppartcount')}, "
        f"position X={g('MX1actm')} Y={g('MY1actm')} Z={g('MZ1actm')}, "
        f"axis load X={g('MX1load')}% Y={g('MY1load')}% Z={g('MZ1load')}%, "
        f"B-axis position={g('B1actm')} load={g('B1load')}%, "
        f"feedrate actual={g('Mp1Fact')} commanded={g('Mp1Fcmd')} "
        f"(override={g('MpFovr')}%), "
        f"program line={g('Mp1line')} block={g('Mp1block')}."
    )


def load_csv_documents(csv_path: str, sample_every: int = 60) -> list:
    docs = []
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        flag_mask = pd.Series(False, index=df.index)
        if "MS1load" in df.columns:
            flag_mask |= pd.to_numeric(df["MS1load"], errors="coerce").fillna(0) > 80
        if "Mestop" in df.columns:
            flag_mask |= df["Mestop"].astype(str).str.upper() == "TRIGGERED"
        df["_flagged"] = flag_mask
        sampled = df.iloc[::sample_every]
        flagged = df[flag_mask]
        df = pd.concat([sampled, flagged]).drop_duplicates().sort_index().reset_index(drop=True)
        print(f"  [csv]   {os.path.basename(csv_path)} — {len(df)} rows ({len(sampled)} sampled + {int(flag_mask.sum())} flagged events)")
        for _, row in df.iterrows():
            docs.append(Document(
                text=_mtconnect_row_to_text(row),
                metadata={
                    "source":    os.path.basename(csv_path),
                    "type":      "mtconnect",
                    "timestamp": str(row.get("timestamp", "")),
                    "tool":      str(row.get("Mp1CurrentTool", "")),
                    "execution": str(row.get("Mpexecution", "")),
                    "load":      str(row.get("MS1load", "")),
                    "estop":     str(row.get("Mestop", "")),
                    "flagged":   "true" if row.get("_flagged") else "false",
                }
            ))
    except Exception as e:
        print(f"  [ERROR] {csv_path}: {e}")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# PROGRAM FREQUENCY SUMMARY — injected as searchable docs so "most common
# program" queries have a single authoritative chunk to retrieve.
# ─────────────────────────────────────────────────────────────────────────────

def _program_summary_docs(csv_files: list) -> list:
    from collections import Counter
    docs = []
    global_counter: Counter = Counter()

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if "Mpprogram" not in df.columns:
                continue
            programs = (
                df["Mpprogram"].dropna().astype(str)
                .str.strip().replace("", pd.NA).dropna()
            )
            if programs.empty:
                continue
            counter  = Counter(programs.tolist())
            global_counter.update(counter)
            top5     = counter.most_common(5)
            top_name, top_count = top5[0]
            total    = sum(counter.values())

            ts_col = pd.to_datetime(df.get("timestamp", pd.Series(dtype=str)), errors="coerce").dropna()
            date_range = ""
            if not ts_col.empty:
                s, e = ts_col.min().strftime("%B %Y"), ts_col.max().strftime("%B %Y")
                date_range = f" ({s}–{e})" if s != e else f" ({s})"

            others = ", ".join(f"{n} ({c} rows)" for n, c in top5[1:])
            text = (
                f"[PROGRAM-SUMMARY] {os.path.basename(csv_path)}{date_range}: "
                f"most common CNC program is {top_name} with {top_count} of {total} rows "
                f"({round(top_count / total * 100, 1)}%). "
                + (f"Other programs: {others}." if others else "")
            )
            docs.append(Document(
                text=text,
                metadata={"source": os.path.basename(csv_path), "type": "mtconnect"},
            ))
        except Exception as exc:
            print(f"  [WARN] program summary for {csv_path}: {exc}")

    if global_counter:
        top5g  = global_counter.most_common(5)
        lines  = ", ".join(f"{n} ({c} rows)" for n, c in top5g)
        docs.append(Document(
            text=f"[PROGRAM-SUMMARY] Across all MTConnect data, CNC program frequency: {lines}.",
            metadata={"source": "all_csvs", "type": "mtconnect"},
        ))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL SHEET HANDLERS — natural language per sheet type
# ─────────────────────────────────────────────────────────────────────────────

_MTCONNECT_COLS = {"MS1load", "Mpexecution", "MS1speed", "MS1Mode", "Mestop"}

# Sheets that are redundant or too noisy to ingest as-is
_SKIP_SHEETS = {
    "Shift_Metrics_Wide",    # duplicate of Long + RAG_Text
    "Shift_Metrics_Long",    # covered by RAG_Text
    "Shift_Long_Enriched",   # same as Long
    "Part_Job_Enriched",     # 131-col join, too noisy — Part_Details is cleaner
    "Job_Orders",            # job master without shift context
    "Failed_Files",          # empty
}


def _employees_sheet_to_docs(df: pd.DataFrame, xlsx_path: str) -> list:
    """
    Group employees by shift file → one natural-language document per shift,
    listing all employees who worked that shift.
    """
    docs  = []
    df    = df.dropna(how="all")
    cols  = {c.strip().lower(): c for c in df.columns}
    emp_c = cols.get("employee_name") or cols.get("employee name")
    src_c = cols.get("source_file")
    if not emp_c or not src_c:
        return docs

    for src_file, grp in df.groupby(src_c):
        date_str, shift_str = _parse_shift_filename(src_file)
        names = [str(n).strip() for n in grp[emp_c].dropna()
                 if str(n).strip() not in ("", "nan")]
        if not names:
            continue
        text = (
            f"Employees who worked the {shift_str} shift on {date_str}: "
            + ", ".join(names) + ". "
            f"(Source file: {src_file})"
        )
        docs.append(Document(
            text=text,
            metadata={
                "source": os.path.basename(xlsx_path),
                "sheet":  "Employees",
                "type":   "employees",
                "date":   date_str,
                "shift":  shift_str,
            }
        ))
    return docs


def _part_details_row_to_text(row: pd.Series) -> str:
    date   = str(row.get("date", ""))[:10]
    shift  = str(row.get("shift", ""))
    wc     = str(row.get("work_center", ""))
    part   = str(row.get("part_number", ""))
    job    = str(row.get("job_order", ""))
    status = str(row.get("job_status", ""))
    qty    = row.get("routing_qty", None)
    good   = row.get("good", None)

    # Skip header-repeat rows
    if part in ("Part #", "Part Number", "nan") or job in ("Job Order", "nan"):
        return ""

    qty_str  = f" Routing quantity: {int(qty)}." if pd.notna(qty) else ""
    good_str = f" Good parts produced: {int(good)}." if pd.notna(good) else ""

    try:
        date_human = _readable_ts(date + " 00:00:00").split(" at ")[0]
    except Exception:
        date_human = date

    return (
        f"On {date_human} during the {shift} shift, work center {wc} ran "
        f"part number {part} under job order {job} (status: {status}).{qty_str}{good_str}"
    )


def _machine_summary_row_to_text(row: pd.Series) -> str:
    date  = str(row.get("date", ""))[:10]
    shift = str(row.get("shift", ""))
    wc    = str(row.get("work_center", ""))
    util  = row.get("avg_utilization_percent", None)
    qty   = row.get("total_quantity", None)

    try:
        date_human = _readable_ts(date + " 00:00:00").split(" at ")[0]
    except Exception:
        date_human = date

    util_str = f"{util:.1f}%" if pd.notna(util) else "N/A"
    qty_str  = f" Total parts produced: {int(qty)}." if pd.notna(qty) else ""

    return (
        f"Work center {wc} had an average utilization of {util_str} "
        f"on the {shift} shift on {date_human}.{qty_str}"
    )


def _date_shift_summary_row_to_text(row: pd.Series) -> str:
    date  = str(row.get("date", ""))[:10]
    shift = str(row.get("shift", ""))
    util  = row.get("avg_utilization_percent", None)
    qty   = row.get("total_quantity", None)
    wcs   = row.get("unique_work_centers", None)

    try:
        date_human = _readable_ts(date + " 00:00:00").split(" at ")[0]
    except Exception:
        date_human = date

    util_str = f"{util:.1f}%" if pd.notna(util) else "N/A"
    qty_str  = f", {int(qty)} total parts" if pd.notna(qty) else ""
    wc_str   = f" across {int(wcs)} work centers" if pd.notna(wcs) else ""

    return (
        f"On {date_human}, the {shift} shift had overall utilization of "
        f"{util_str}{qty_str}{wc_str}."
    )


def _rag_text_row_to_text(row: pd.Series) -> str:
    """The RAG_Text sheet already has a pre-formatted text column — use it directly."""
    return str(row.get("text", "")).strip()


def _shift_times_row_to_text(row: pd.Series) -> str:
    date  = str(row.get("date", ""))[:10]
    shift = str(row.get("shift", ""))
    start = str(row.get("shift_start", ""))
    end   = str(row.get("shift_end", ""))
    wc    = str(row.get("work_center", ""))
    util  = row.get("summary_avg_utilization_percent", None)
    qty   = row.get("summary_total_quantity", None)

    try:
        date_human  = _readable_ts(date + " 00:00:00").split(" at ")[0]
        start_human = _readable_ts(start).split("at ")[-1] if start != "nan" else "?"
        end_human   = _readable_ts(end).split("at ")[-1]   if end   != "nan" else "?"
    except Exception:
        date_human, start_human, end_human = date, start, end

    util_str = f" Utilization: {util:.1f}%." if pd.notna(util) else ""
    qty_str  = f" Quantity produced: {int(qty)}." if pd.notna(qty) else ""

    return (
        f"On {date_human}, the {shift} shift ran from {start_human} to {end_human} "
        f"at work center {wc}.{util_str}{qty_str}"
    )


def load_xlsx_documents(xlsx_path: str) -> list:
    docs = []
    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
        for sheet_name, df in sheets.items():
            # Normalise column names for matching; keep originals in df
            norm = sheet_name.strip().replace(" ", "_")
            if norm in _SKIP_SHEETS:
                print(f"  [xlsx]  {os.path.basename(xlsx_path)} / '{sheet_name}' — skipped (redundant)")
                continue

            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(how="all")

            # ── Sheet-specific handlers ───────────────────────────────────
            if norm == "Employees":
                sheet_docs = _employees_sheet_to_docs(df, xlsx_path)
                print(f"  [xlsx]  {os.path.basename(xlsx_path)} / '{sheet_name}' — "
                      f"{len(sheet_docs)} shift groups")
                docs.extend(sheet_docs)
                continue

            # Generic row-level handlers
            row_fn   = None
            doc_type = "job_data"

            if norm == "Part_Details":
                row_fn   = _part_details_row_to_text
                doc_type = "part_details"
            elif norm == "Machine_Shift_Summary":
                row_fn   = _machine_summary_row_to_text
                doc_type = "machine_summary"
            elif norm == "Date_Shift_Summary":
                row_fn   = _date_shift_summary_row_to_text
                doc_type = "shift_summary"
            elif norm == "RAG_Text":
                row_fn   = _rag_text_row_to_text
                doc_type = "rag_text"
            elif norm == "Sheet1":   # file_with_utc_shift_times_Ar.xlsx
                row_fn   = _shift_times_row_to_text
                doc_type = "shift_times"
            elif bool(_MTCONNECT_COLS & set(df.columns)):
                row_fn   = lambda row: _mtconnect_row_to_text(row)
                doc_type = "mtconnect"

            if row_fn is None:
                # Fallback: skip sheets we have no handler for
                print(f"  [xlsx]  {os.path.basename(xlsx_path)} / '{sheet_name}' — skipped (no handler)")
                continue

            count = 0
            for _, row in df.iterrows():
                text = row_fn(row)
                if not text:
                    continue
                docs.append(Document(
                    text=text,
                    metadata={
                        "source": os.path.basename(xlsx_path),
                        "sheet":  sheet_name,
                        "type":   doc_type,
                    }
                ))
                count += 1
            print(f"  [xlsx]  {os.path.basename(xlsx_path)} / '{sheet_name}' — {count} docs")
    except Exception as e:
        print(f"  [ERROR] {xlsx_path}: {e}")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO LOADER
# ─────────────────────────────────────────────────────────────────────────────

def audio_features_to_document(features: dict) -> Document:
    return Document(
        text=features["rag_chunk"],
        metadata={
            "source":    features["file"],
            "type":      "audio",
            "timestamp": features["timestamp"],
            "severity":  features["severity"],
            "fault":     str(features["fault_flag"]),
            "rms":       str(round(features["rms"], 4)),
            "kurtosis":  str(round(features["kurtosis"], 3)),
            "dom_freq":  str(round(features["dominant_freq_hz"], 1)),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def ingest():
    documents = []

    # 1. Knowledge base
    print(f"\n── Loading Knowledge Base ─────────────────────────────────")
    kb_docs = load_knowledge_base_documents(KB_PATH)
    documents.extend(kb_docs)
    print(f"  -> {len(kb_docs)} knowledge base chunks")

    # 2. Scan data directory
    xlsx_files, csv_files, wav_files = [], [], []
    for root, dirs, files in os.walk(DATA_PATH):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.lower().endswith(".xlsx"):
                xlsx_files.append(fpath)
            elif fname.lower().endswith(".csv"):
                csv_files.append(fpath)
            elif fname.lower().endswith(".wav"):
                wav_files.append(fpath)

    print(f"\n[scan]  {len(xlsx_files)} Excel files, {len(csv_files)} CSV files, {len(wav_files)} WAV files")

    # 3. Excel / MTConnect
    print(f"\n── Ingesting Excel files ──────────────────────────────────")
    for p in sorted(xlsx_files):
        docs = load_xlsx_documents(p)
        documents.extend(docs)

    # 4. CSV / MTConnect (sampled + flagged events preserved)
    print(f"\n── Ingesting CSV files (every {CSV_SAMPLE_EVERY} rows) ────────────────────")
    for p in sorted(csv_files):
        docs = load_csv_documents(p, sample_every=CSV_SAMPLE_EVERY)
        documents.extend(docs)
    summary_docs = _program_summary_docs(sorted(csv_files))
    documents.extend(summary_docs)
    print(f"  -> {len(summary_docs)} program summary doc(s) added")

    # 5. WAV / Audio
    print(f"\n── Ingesting WAV files ────────────────────────────────────")
    faults, warnings = 0, 0
    for wav_path in sorted(wav_files):
        try:
            features = process_wav(wav_path)
            documents.append(audio_features_to_document(features))
            sev = features["severity"]
            if sev == "fault":    faults += 1
            elif sev == "warning": warnings += 1
            print(f"  [{sev.upper():7s}]  {os.path.basename(wav_path)}")
            if sev in ("warning", "fault"):
                for obs in features["observations"]:
                    print(f"             -> {obs[:120]}")
        except Exception as e:
            print(f"  [ERROR]   {os.path.basename(wav_path)} — {e}")

    print(f"\n[audio]  {faults} fault(s), {warnings} warning(s) found during ingest")

    # 6. Build ChromaDB index
    print(f"\n── Building index ({len(documents)} documents) ────────────")
    chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection("iiot_rag")
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\n[done]  {len(documents)} documents embedded into {CHROMA_PATH}")
    print(f"[done]  {len(kb_docs)} diagnostic knowledge base chunks always available")
    print("[done]  Run 'python server.py' to start the API")


if __name__ == "__main__":
    ingest()
