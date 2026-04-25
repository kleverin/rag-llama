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
import sys
import pandas as pd
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

from audio_processing import process_wav

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PATH     = os.getenv("CHROMA_PATH",     "./chroma_db")
DATA_PATH       = os.getenv("DATA_PATH",        "./data")
KB_PATH         = os.getenv("KB_PATH",          "./")

print(f"[config]  DATA_PATH   = {DATA_PATH}")
print(f"[config]  KB_PATH     = {KB_PATH}")
print(f"[config]  CHROMA_PATH = {CHROMA_PATH}")
print(f"[config]  OLLAMA URL  = {OLLAMA_BASE_URL}")

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
# MTCONNECT EXCEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def row_to_text(row: pd.Series) -> str:
    """Convert one MTConnect Excel row to natural-language text."""
    def g(col, default="N/A"):
        return row.get(col, default)

    flags = []
    try:
        if float(g("MS1load", 0)) > 80:
            flags.append("HIGH-SPINDLE-LOAD")
        if float(g("MS1load", 0)) > 95:
            flags.append("SPINDLE-OVERLOAD")
    except Exception:
        pass
    for ax, col in [("X", "MX1load"), ("Y", "MY1load"), ("Z", "MZ1load")]:
        try:
            if abs(float(g(col, 0))) > 70:
                flags.append(f"HIGH-{ax}-AXIS-LOAD")
        except Exception:
            pass
    if str(g("Mestop")).upper() == "TRIGGERED":
        flags.append("ESTOP-TRIGGERED")

    flag_str = " | FLAGS: " + ", ".join(flags) if flags else ""

    return (
        f"[MTCONNECT]{flag_str} At {g('timestamp')}: "
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


def load_xlsx_documents(xlsx_path: str) -> list:
    docs = []
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        df.columns = [c.strip() for c in df.columns]
        print(f"  [xlsx]  {os.path.basename(xlsx_path)} — {len(df)} rows")
        for _, row in df.iterrows():
            docs.append(Document(
                text=row_to_text(row),
                metadata={
                    "source":    os.path.basename(xlsx_path),
                    "type":      "mtconnect",
                    "timestamp": str(row.get("timestamp", "")),
                    "tool":      str(row.get("Mp1CurrentTool", "")),
                    "execution": str(row.get("Mpexecution", "")),
                    "load":      str(row.get("MS1load", "")),
                    "estop":     str(row.get("Mestop", "")),
                }
            ))
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
    xlsx_files, wav_files = [], []
    for root, dirs, files in os.walk(DATA_PATH):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.lower().endswith(".xlsx"):
                xlsx_files.append(fpath)
            elif fname.lower().endswith(".wav"):
                wav_files.append(fpath)

    print(f"\n[scan]  {len(xlsx_files)} Excel files, {len(wav_files)} WAV files")

    # 3. Excel / MTConnect
    print(f"\n── Ingesting Excel files ──────────────────────────────────")
    for p in sorted(xlsx_files):
        docs = load_xlsx_documents(p)
        documents.extend(docs)

    # 4. WAV / Audio
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

    # 5. Build ChromaDB index
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
