import os
import socket
import pathlib
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
PORT = int(os.getenv("PORT", "8000"))

chat_engine = None
employee_engine = None
audio_engine = None
parts_engine = None

_EMPLOYEE_KEYWORDS = {
    "employee", "employees", "worker", "workers",
    "who worked", "who was on", "who were on",
    "staff", "personnel", "worked the shift", "working the shift",
}

_QA_PROMPT_STR = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only the context information, answer the query. "
    "Always explicitly state the exact values, numbers, names, part numbers, job orders, "
    "and program names from the context — never use a pronoun like 'it' or 'this program' "
    "without also naming the specific value.\n"
    "Query: {query_str}\n"
    "Answer: "
)

_PARTS_KEYWORDS = {
    "part number", "part #", "part no", "job order", "good parts",
    "what part", "what job", "which part", "routing quantity",
    "produced on the", "running on the", "work center ran",
}

_AUDIO_KEYWORDS = {
    "audio", "sound", "wav", "recording", "sensor",
    "fault", "bearing", "vibration", "kurtosis", "crest factor",
    "rms", "spectral", "frequency", "acoustic", "noise",
    "warning", "normal operation", "pitch", "harmonic",
    "mb6000", "mb4000 audio", "zero crossing",
}


def _is_employee_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _EMPLOYEE_KEYWORDS)


def _is_parts_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _PARTS_KEYWORDS)


def _is_audio_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _AUDIO_KEYWORDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_engine, employee_engine, audio_engine, parts_engine

    if not pathlib.Path(CHROMA_PATH).exists():
        raise RuntimeError(
            f"ChromaDB not found at '{CHROMA_PATH}'. Run ingest.py first."
        )

    print("Loading ChromaDB vector store...")
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.core import VectorStoreIndex, StorageContext

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection("iiot_rag")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_BASE_URL,
    )
    llm = Ollama(model="llama3.2", base_url=OLLAMA_BASE_URL, request_timeout=120.0)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        similarity_top_k=15,
        verbose=True,
    )

    from llama_index.core import PromptTemplate
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    qa_tmpl = PromptTemplate(_QA_PROMPT_STR)
    employee_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=15,
        filters=MetadataFilters(filters=[
            ExactMatchFilter(key="type", value="employees")
        ]),
    )
    audio_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=20,
        text_qa_template=qa_tmpl,
        filters=MetadataFilters(filters=[
            ExactMatchFilter(key="type", value="audio")
        ]),
    )
    parts_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=15,
        text_qa_template=qa_tmpl,
        filters=MetadataFilters(filters=[
            ExactMatchFilter(key="type", value="part_details")
        ]),
    )

    local_ip = _get_local_ip()
    print(f"\n=== Server ready ===")
    print(f"  Local:  http://localhost:{PORT}")
    print(f"  Phone:  http://{local_ip}:{PORT}")
    print(f"  Docs:   http://localhost:{PORT}/docs\n")

    yield


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


app = FastAPI(title="RAG LlamaIndex API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <meta name="theme-color" content="#0e1013" />
  <title>Kirby Risk Assistant</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg-0: #0b0d0f;
      --bg-1: #111417;
      --bg-2: #181c20;
      --bg-3: #20252a;
      --line: #2a3037;
      --line-soft: #1c2126;
      --text: #e6e8ea;
      --text-dim: #8b939c;
      --text-mute: #5a6168;
      --accent: #ff8b1f;
      --accent-dim: #b65f12;
      --ok: #4ade80;
      --warn: #facc15;
      --err: #ef4444;
      --user-bg: #1f2630;
      --bot-bg: #14181c;
      --safe-bottom: env(safe-area-inset-bottom, 0px);
    }

    html, body {
      height: 100%;
      overflow: hidden;
      overscroll-behavior: none;
    }

    body {
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background: var(--bg-0);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100dvh;
      -webkit-font-smoothing: antialiased;
      -webkit-tap-highlight-color: transparent;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(to right, rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255,255,255,0.018) 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: radial-gradient(ellipse at center, #000 40%, transparent 90%);
      -webkit-mask-image: radial-gradient(ellipse at center, #000 40%, transparent 90%);
      z-index: 0;
    }

    .mono {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    }

    header {
      position: relative;
      z-index: 2;
      flex-shrink: 0;
      background: linear-gradient(180deg, #15191e 0%, #0f1216 100%);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 6px 18px rgba(0,0,0,0.45);
      padding: 10px 14px calc(10px + env(safe-area-inset-top, 0px));
      padding-top: calc(10px + env(safe-area-inset-top, 0px));
    }

    .head-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      flex: 1;
    }

    .brand-mark {
      width: 34px; height: 34px;
      flex-shrink: 0;
      border-radius: 6px;
      background: linear-gradient(135deg, #1c2228, #0c0f12);
      border: 1px solid var(--line);
      display: grid;
      place-items: center;
      box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 0 0 1px rgba(0,0,0,0.4);
      position: relative;
    }
    .brand-mark svg { width: 18px; height: 18px; display: block; }

    .brand-text { min-width: 0; }
    .brand-name {
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      line-height: 1.1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .brand-sub {
      font-size: 0.65rem;
      color: var(--text-mute);
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-top: 2px;
    }

    .head-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 9px 5px 8px;
      background: var(--bg-2);
      border: 1px solid var(--line);
      border-radius: 4px;
      font-size: 0.65rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--text-dim);
      transition: color 0.2s, border-color 0.2s;
    }
    .status-pill[data-state="online"]  { color: #c6e9d2; border-color: #1f3a2a; }
    .status-pill[data-state="offline"] { color: #f1c5c5; border-color: #3a1f1f; }
    .status-pill[data-state="checking"]{ color: var(--text-dim); }

    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--text-mute);
      position: relative;
    }
    .status-pill[data-state="online"] .status-dot {
      background: var(--ok);
      box-shadow: 0 0 8px rgba(74,222,128,0.6);
      animation: pulse-ok 2s ease-out infinite;
    }
    .status-pill[data-state="offline"] .status-dot {
      background: var(--err);
      box-shadow: 0 0 8px rgba(239,68,68,0.5);
    }
    .status-pill[data-state="checking"] .status-dot {
      background: var(--warn);
      animation: pulse-warn 1s ease-in-out infinite;
    }

    @keyframes pulse-ok {
      0%   { box-shadow: 0 0 0 0 rgba(74,222,128,0.5); }
      70%  { box-shadow: 0 0 0 8px rgba(74,222,128,0); }
      100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); }
    }
    @keyframes pulse-warn {
      0%, 100% { opacity: 1; }
      50%      { opacity: 0.35; }
    }

    .btn {
      background: var(--bg-2);
      border: 1px solid var(--line);
      color: var(--text);
      padding: 7px 11px;
      border-radius: 4px;
      font-size: 0.7rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s, border-color 0.15s, color 0.15s;
      font-family: inherit;
    }
    .btn:hover, .btn:focus-visible {
      background: var(--bg-3);
      border-color: #3a4149;
      outline: none;
    }
    .btn:active { transform: translateY(1px); }
    .btn svg { width: 12px; height: 12px; }

    .head-strip {
      margin-top: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.62rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--text-mute);
    }
    .head-strip .hash { color: var(--accent); }
    .head-strip .sep {
      flex: 1;
      height: 1px;
      background: repeating-linear-gradient(
        to right,
        var(--line) 0 6px,
        transparent 6px 10px
      );
    }

    #messages {
      position: relative;
      z-index: 1;
      flex: 1;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 18px 14px 24px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      scrollbar-width: thin;
      scrollbar-color: #2a3037 transparent;
    }
    #messages::-webkit-scrollbar { width: 6px; }
    #messages::-webkit-scrollbar-thumb {
      background: #2a3037;
      border-radius: 3px;
    }

    .empty-state {
      margin: auto;
      text-align: center;
      color: var(--text-mute);
      padding: 24px 16px;
      max-width: 320px;
      animation: fade-in 0.5s ease-out both;
    }
    .empty-mark {
      width: 56px; height: 56px;
      margin: 0 auto 14px;
      border-radius: 8px;
      background: linear-gradient(135deg, #1a2026, #0a0d10);
      border: 1px solid var(--line);
      display: grid;
      place-items: center;
      position: relative;
    }
    .empty-mark::before {
      content: "";
      position: absolute;
      inset: -1px;
      border-radius: 8px;
      background: linear-gradient(135deg, transparent 60%, rgba(255,139,31,0.25));
      -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
              mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
      -webkit-mask-composite: xor;
              mask-composite: exclude;
      padding: 1px;
    }
    .empty-mark svg { width: 26px; height: 26px; color: var(--accent); }
    .empty-title {
      color: var(--text);
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    .empty-sub {
      margin-top: 6px;
      font-size: 0.78rem;
      line-height: 1.55;
    }
    .empty-tag {
      display: inline-block;
      margin-top: 14px;
      padding: 5px 9px;
      background: var(--bg-2);
      border: 1px solid var(--line);
      border-radius: 4px;
      font-size: 0.6rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--text-dim);
    }
    .empty-tag .dot {
      display: inline-block;
      width: 5px; height: 5px;
      border-radius: 50%;
      background: var(--accent);
      margin-right: 6px;
      vertical-align: middle;
    }

    .bubble-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
      animation: msg-in 0.32s cubic-bezier(0.22, 1, 0.36, 1) both;
      max-width: 100%;
    }
    .bubble-row.user { align-items: flex-end; }
    .bubble-row.bot  { align-items: flex-start; }

    @keyframes msg-in {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fade-in {
      from { opacity: 0; }
      to   { opacity: 1; }
    }

    .bubble-meta {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.6rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--text-mute);
      padding: 0 4px;
    }
    .bubble-meta .who { color: var(--text-dim); }
    .bubble-row.bot .bubble-meta .who { color: var(--accent); }
    .bubble-meta .ts { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }

    .bubble {
      max-width: 86%;
      padding: 11px 14px;
      font-size: 0.92rem;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      position: relative;
    }

    .bubble-row.user .bubble {
      background: var(--user-bg);
      color: var(--text);
      border-radius: 10px 10px 2px 10px;
    }
    .bubble-row.bot .bubble {
      background: var(--bot-bg);
      color: var(--text);
      border-radius: 10px 10px 10px 2px;
      box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset;
    }
    .bubble-row.bot .bubble::before {
      content: "";
      position: absolute;
      top: -1px; left: -1px;
      width: 10px; height: 10px;
      border-top: 2px solid var(--accent);
      border-left: 2px solid var(--accent);
      border-top-left-radius: 10px;
    }

    .bubble.thinking {
      color: var(--text-dim);
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .thinking-dots {
      display: inline-flex;
      gap: 3px;
    }
    .thinking-dots span {
      width: 6px; height: 6px;
      background: var(--accent);
      border-radius: 50%;
      animation: blink 1.2s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.15s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes blink {
      0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); }
      40%           { opacity: 1;    transform: scale(1); }
    }

    .bubble.error {
      background: #1d1113;
      border-color: #3d1d22;
      color: #ffb4b4;
    }
    .bubble.error::before { display: none; }
    .bubble-row.bot .bubble.error { border-radius: 10px 10px 10px 2px; }

    .bubble p { margin: 0 0 7px; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble ol, .bubble ul { padding-left: 1.5em; margin: 4px 0 7px; }
    .bubble li { margin: 3px 0; line-height: 1.55; }
    .bubble li:last-child { margin-bottom: 0; }
    .bubble strong { font-weight: 600; color: var(--text); }
    .bubble em { font-style: italic; color: var(--text-dim); }
    .bubble .ic {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.83em;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--line);
      padding: 1px 5px;
      border-radius: 3px;
    }

    #input-bar {
      position: relative;
      z-index: 2;
      flex-shrink: 0;
      background: linear-gradient(180deg, #0f1216 0%, #0a0c0f 100%);
      border-top: 1px solid var(--line);
      padding: 10px 12px calc(10px + var(--safe-bottom));
    }

    .composer {
      background: var(--bg-1);
      border: 1px solid var(--line);
      border-radius: 10px;
      display: flex;
      align-items: flex-end;
      gap: 8px;
      padding: 6px 6px 6px 12px;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    .composer:focus-within {
      border-color: var(--accent-dim);
      box-shadow: 0 0 0 3px rgba(255,139,31,0.08);
    }

    #question {
      flex: 1;
      min-height: 28px;
      max-height: 140px;
      padding: 8px 0;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem;
      line-height: 1.45;
      resize: none;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: #2a3037 transparent;
    }
    #question::placeholder { color: var(--text-mute); }

    #send-btn {
      flex-shrink: 0;
      width: 36px; height: 36px;
      border-radius: 8px;
      border: 1px solid var(--accent-dim);
      background: linear-gradient(180deg, #ff9a3a, #e8761a);
      color: #1a0e02;
      cursor: pointer;
      display: grid;
      place-items: center;
      transition: filter 0.15s, transform 0.05s, opacity 0.2s;
      box-shadow: 0 1px 0 rgba(255,255,255,0.2) inset, 0 2px 6px rgba(255,139,31,0.15);
    }
    #send-btn:hover  { filter: brightness(1.08); }
    #send-btn:active { transform: translateY(1px); }
    #send-btn:disabled {
      background: var(--bg-3);
      border-color: var(--line);
      color: var(--text-mute);
      cursor: default;
      box-shadow: none;
    }
    #send-btn svg { width: 16px; height: 16px; }

    .footer-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 8px;
      font-size: 0.58rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--text-mute);
      padding: 0 2px;
    }
    .footer-bar .kbd {
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .kbd-key {
      display: inline-block;
      padding: 1px 5px;
      border: 1px solid var(--line);
      border-radius: 3px;
      background: var(--bg-2);
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      letter-spacing: 0;
      font-size: 0.62rem;
      color: var(--text-dim);
    }

    @media (max-width: 380px) {
      .footer-bar .kbd { display: none; }
      .brand-sub { display: none; }
    }
    @media (max-width: 360px) {
      .status-pill .label { display: none; }
      .status-pill { padding: 5px 7px; }
    }
    @media (hover: none) {
      .btn:hover { background: var(--bg-2); }
    }

    /* BOOT LOADER */
    #boot {
      position: fixed;
      inset: 0;
      z-index: 9999;
      background: #020617;
      color: #a8edff;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
      transition: opacity 0.7s ease, transform 0.7s ease, visibility 0s linear 0.7s;
      will-change: opacity, transform;
      overflow: hidden;
    }
    #boot.hide {
      opacity: 0;
      transform: scale(1.02);
      visibility: hidden;
      pointer-events: none;
    }

    #boot::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(to right, rgba(0,212,255,0.055) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(0,212,255,0.055) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: radial-gradient(ellipse at center, #000 25%, transparent 78%);
      -webkit-mask-image: radial-gradient(ellipse at center, #000 25%, transparent 78%);
      animation: grid-drift 20s linear infinite;
    }
    @keyframes grid-drift {
      from { background-position: 0 0; }
      to   { background-position: 48px 48px; }
    }

    .boot-scanlines {
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: repeating-linear-gradient(
        to bottom,
        transparent 0px,
        transparent 3px,
        rgba(0,0,0,0.2) 3px,
        rgba(0,0,0,0.2) 4px
      );
      animation: scan-shift 5s linear infinite;
      z-index: 1;
    }
    @keyframes scan-shift {
      from { background-position: 0 0; }
      to   { background-position: 0 80px; }
    }

    .boot-vignette {
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: radial-gradient(ellipse 80% 80% at center, transparent 30%, rgba(2,6,23,0.88) 100%);
      z-index: 1;
    }

    .boot-hscan {
      position: absolute;
      left: 0; right: 0; top: 0;
      height: 120px;
      background: linear-gradient(to bottom,
        transparent 0%,
        rgba(0,212,255,0.035) 45%,
        rgba(0,212,255,0.06) 50%,
        rgba(0,212,255,0.035) 55%,
        transparent 100%
      );
      animation: hscan-fall 7s ease-in-out infinite;
      pointer-events: none;
      z-index: 1;
    }
    @keyframes hscan-fall {
      0%   { transform: translateY(-120px); opacity: 1; }
      100% { transform: translateY(100vh);  opacity: 0; }
    }

    .boot-corner {
      position: absolute;
      width: 38px; height: 38px;
      border: 2px solid rgba(0,212,255,0.85);
      box-shadow: 0 0 14px rgba(0,212,255,0.45), inset 0 0 8px rgba(0,212,255,0.08);
      opacity: 0;
      animation: corner-in 0.45s cubic-bezier(0.22,1,0.36,1) forwards;
      z-index: 3;
    }
    .boot-corner.tl { top: 14px; left: 14px; border-right: none; border-bottom: none; animation-delay: 0.05s; }
    .boot-corner.tr { top: 14px; right: 14px; border-left: none; border-bottom: none; animation-delay: 0.1s; }
    .boot-corner.bl { bottom: 14px; left: 14px; border-right: none; border-top: none; animation-delay: 0.15s; }
    .boot-corner.br { bottom: 14px; right: 14px; border-left: none; border-top: none; animation-delay: 0.2s; }
    @keyframes corner-in {
      from { opacity: 0; transform: scale(0.5); }
      to   { opacity: 1; transform: scale(1); }
    }

    .boot-panel-left, .boot-panel-right {
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      width: 130px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 0.52rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      animation: boot-fade 0.5s ease-out 1.1s both;
      z-index: 3;
    }
    .boot-panel-left  { left: 22px; border-left: 1px solid rgba(0,212,255,0.18); padding-left: 10px; }
    .boot-panel-right { right: 22px; border-right: 1px solid rgba(0,212,255,0.18); padding-right: 10px; }
    @media (max-width: 720px) { .boot-panel-left, .boot-panel-right { display: none; } }

    .bpl-title {
      color: rgba(0,212,255,0.42);
      font-size: 0.46rem;
      letter-spacing: 0.26em;
      padding-bottom: 5px;
      border-bottom: 1px solid rgba(0,212,255,0.12);
      margin-bottom: 2px;
    }
    .bpl-row {
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .bpl-row-r {
      display: flex;
      align-items: center;
      gap: 5px;
      justify-content: flex-end;
    }
    .bpl-key { color: rgba(0,212,255,0.38); flex-shrink: 0; min-width: 34px; }
    .bpl-val { color: rgba(0,212,255,0.82); }
    .bpl-bar {
      flex: 1;
      height: 2px;
      background: rgba(0,212,255,0.1);
      border-radius: 1px;
      overflow: hidden;
    }
    .bpl-fill {
      height: 100%;
      width: var(--w, 50%);
      background: linear-gradient(90deg, rgba(0,212,255,0.4), #00d4ff);
      box-shadow: 0 0 4px rgba(0,212,255,0.5);
    }
    .bpl-divider { height: 1px; background: rgba(0,212,255,0.1); margin: 2px 0; }

    .boot-inner {
      position: relative;
      z-index: 4;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
      width: 100%;
      max-width: 380px;
      text-align: center;
    }

    .boot-tag {
      font-size: 0.6rem;
      letter-spacing: 0.3em;
      color: rgba(0,212,255,0.5);
      text-transform: uppercase;
      animation: boot-fade 0.5s ease-out 0.1s both;
    }
    .boot-tag .blink-dot {
      display: inline-block;
      width: 6px; height: 6px;
      background: #00d4ff;
      border-radius: 50%;
      margin-right: 8px;
      vertical-align: middle;
      box-shadow: 0 0 10px #00d4ff, 0 0 22px rgba(0,212,255,0.5);
      animation: dot-blink 1s steps(2) infinite;
    }
    @keyframes dot-blink { 50% { opacity: 0.15; } }

    .boot-emblem {
      position: relative;
      width: 130px; height: 130px;
      display: flex;
      align-items: center;
      justify-content: center;
      animation: boot-fade 0.5s ease-out 0.25s both;
    }
    .boot-hex {
      position: relative;
      z-index: 2;
      width: 70px; height: 70px;
      background: rgba(0,212,255,0.05);
      border: 1px solid rgba(0,212,255,0.45);
      clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
      display: grid;
      place-items: center;
      animation: hex-pulse 3s ease-in-out infinite;
    }
    @keyframes hex-pulse {
      0%, 100% { background: rgba(0,212,255,0.05); filter: drop-shadow(0 0 6px rgba(0,212,255,0.3)); }
      50%       { background: rgba(0,212,255,0.11); filter: drop-shadow(0 0 14px rgba(0,212,255,0.65)); }
    }
    .boot-hex svg { width: 28px; height: 28px; color: #00d4ff; filter: drop-shadow(0 0 4px #00d4ff); }

    .boot-ring-1 {
      position: absolute;
      inset: 18px;
      border: 1px solid rgba(0,212,255,0.15);
      border-top-color: #00d4ff;
      border-radius: 50%;
      animation: ring-cw 2.2s linear infinite;
    }
    .boot-ring-1::before {
      content: "";
      position: absolute;
      top: -3px; left: 50%;
      width: 5px; height: 5px;
      background: #00d4ff;
      border-radius: 50%;
      transform: translateX(-50%);
      box-shadow: 0 0 8px #00d4ff, 0 0 16px rgba(0,212,255,0.6);
    }
    .boot-ring-2 {
      position: absolute;
      inset: 0;
      border: 1px solid rgba(0,212,255,0.08);
      border-bottom-color: rgba(0,212,255,0.5);
      border-radius: 50%;
      animation: ring-ccw 3.8s linear infinite;
    }
    .boot-ring-2::before {
      content: "";
      position: absolute;
      bottom: -3px; left: 50%;
      width: 4px; height: 4px;
      background: rgba(0,212,255,0.65);
      border-radius: 50%;
      transform: translateX(-50%);
      box-shadow: 0 0 6px rgba(0,212,255,0.65);
    }
    @keyframes ring-cw  { to { transform: rotate(360deg);  } }
    @keyframes ring-ccw { to { transform: rotate(-360deg); } }

    .boot-title {
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: clamp(1.3rem, 5.5vw, 1.95rem);
      font-weight: 600;
      letter-spacing: 0.1em;
      color: #eaffff;
      position: relative;
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 0;
      line-height: 1.1;
      text-shadow:
        0 0 8px rgba(0,212,255,0.9),
        0 0 24px rgba(0,212,255,0.5),
        0 0 48px rgba(0,212,255,0.2);
    }
    .boot-title .word { display: inline-flex; margin: 0 0.18em; }
    .boot-title .ch {
      display: inline-block;
      opacity: 0;
      transform: translateY(8px);
      animation: ch-in 0.45s cubic-bezier(0.22,1,0.36,1) forwards;
    }
    @keyframes ch-in { to { opacity: 1; transform: translateY(0); } }

    .boot-title-wrap { position: relative; animation: boot-fade 0.01s ease-out 0.3s both; }
    .boot-title-wrap .glitch {
      position: absolute;
      inset: 0;
      pointer-events: none;
      color: #00d4ff;
      mix-blend-mode: screen;
      opacity: 0;
      animation: glitch-flicker 4s 1.8s infinite;
      clip-path: inset(0 0 0 0);
      text-shadow: 2px 0 #ff00e4, -2px 0 #00ffe4;
      font-family: inherit;
      font-size: inherit;
      letter-spacing: inherit;
      font-weight: inherit;
      white-space: nowrap;
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
    }
    @keyframes glitch-flicker {
      0%, 92%, 100% { opacity: 0; transform: translate(0,0); clip-path: inset(0 0 0 0); }
      93%  { opacity: 0.7; transform: translate(-3px, 0); clip-path: inset(20% 0 60% 0); }
      94%  { opacity: 0.5; transform: translate(3px, 1px); clip-path: inset(50% 0 20% 0); }
      95%  { opacity: 0.8; transform: translate(-2px, -1px); clip-path: inset(0 0 75% 0); }
      96%  { opacity: 0; }
      97%  { opacity: 0.45; transform: translate(2px, 0); clip-path: inset(65% 0 5% 0); }
      98%  { opacity: 0; }
    }

    .boot-title.flicker { animation: title-flicker 3.5s ease-in-out 0.8s infinite; }
    @keyframes title-flicker {
      0%, 100% { opacity: 1; }
      48%, 52%  { opacity: 0.88; }
      50%       { opacity: 0.3; }
    }

    .boot-progress {
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 8px;
      animation: boot-fade 0.5s ease-out 0.5s both;
    }
    .boot-bar {
      width: 100%;
      height: 6px;
      background: rgba(0,212,255,0.06);
      border: 1px solid rgba(0,212,255,0.18);
      border-radius: 1px;
      overflow: hidden;
      position: relative;
    }
    .boot-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, rgba(0,212,255,0.5) 0%, #00d4ff 60%, #b8f0ff 100%);
      box-shadow: 0 0 12px #00d4ff, 0 0 26px rgba(0,212,255,0.5);
      transition: width 0.18s linear;
      position: relative;
    }
    .boot-bar-fill::after {
      content: "";
      position: absolute;
      top: 0; right: 0; bottom: 0;
      width: 18px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.75));
      filter: blur(2px);
    }
    .boot-bar::after {
      content: "";
      position: absolute;
      inset: 0;
      background: repeating-linear-gradient(
        to right,
        transparent 0%,
        transparent calc(5% - 1px),
        rgba(2,6,23,0.55) calc(5% - 1px),
        rgba(2,6,23,0.55) 5%
      );
      pointer-events: none;
      z-index: 2;
    }

    .boot-meta {
      display: flex;
      justify-content: space-between;
      font-size: 0.6rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: rgba(0,212,255,0.5);
    }
    .boot-meta .pct { color: #d6f7ff; min-width: 4ch; text-align: right; }
    .boot-meta .status { color: rgba(0,212,255,0.75); }
    .boot-meta .status.error { color: #ff4466; animation: err-pulse 0.8s ease-in-out infinite; }
    @keyframes err-pulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.5; }
    }

    .boot-log {
      width: 100%;
      height: 64px;
      font-size: 0.58rem;
      color: rgba(0,212,255,0.45);
      letter-spacing: 0.05em;
      text-align: left;
      overflow: hidden;
      animation: boot-fade 0.5s ease-out 0.7s both;
      position: relative;
      mask-image: linear-gradient(to bottom, transparent 0%, #000 30%, #000 100%);
      -webkit-mask-image: linear-gradient(to bottom, transparent 0%, #000 30%, #000 100%);
    }
    .boot-log .line {
      opacity: 0;
      animation: boot-fade 0.3s ease-out forwards;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .boot-log .line .ok  { color: #00d4ff; }
    .boot-log .line .err { color: #ff4466; }

    @keyframes boot-fade {
      from { opacity: 0; transform: translateY(4px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    @media (prefers-reduced-motion: reduce) {
      .boot-title .ch { animation-duration: 0.01s; }
      .boot-title.flicker, .boot-title-wrap .glitch { animation: none; }
      .boot-ring-1, .boot-ring-2 { animation: none; }
      #boot::before, .boot-hscan, .boot-scanlines { animation: none; }
    }
  </style>
</head>
<body>

<!-- BOOT LOADER -->
<div id="boot" role="status" aria-live="polite">
  <div class="boot-scanlines" aria-hidden="true"></div>
  <div class="boot-vignette" aria-hidden="true"></div>
  <div class="boot-hscan" aria-hidden="true"></div>
  <div class="boot-corner tl" aria-hidden="true"></div>
  <div class="boot-corner tr" aria-hidden="true"></div>
  <div class="boot-corner bl" aria-hidden="true"></div>
  <div class="boot-corner br" aria-hidden="true"></div>
  <div class="boot-panel-left" aria-hidden="true">
    <div class="bpl-title">SYS STATUS</div>
    <div class="bpl-row"><span class="bpl-key">CPU</span><span class="bpl-bar"><span class="bpl-fill" style="--w:72%"></span></span><span class="bpl-val">72%</span></div>
    <div class="bpl-row"><span class="bpl-key">MEM</span><span class="bpl-bar"><span class="bpl-fill" style="--w:58%"></span></span><span class="bpl-val">58%</span></div>
    <div class="bpl-row"><span class="bpl-key">VEC</span><span class="bpl-bar"><span class="bpl-fill" style="--w:100%"></span></span><span class="bpl-val">OK</span></div>
    <div class="bpl-divider"></div>
    <div class="bpl-row"><span class="bpl-key">TEMP</span><span class="bpl-val">42°C</span></div>
    <div class="bpl-row"><span class="bpl-key">UP</span><span class="bpl-val" id="bpl-uptime">00:00</span></div>
  </div>
  <div class="boot-panel-right" aria-hidden="true">
    <div class="bpl-title" style="text-align:right">CONFIG</div>
    <div class="bpl-row-r"><span class="bpl-val">llama3.2</span><span class="bpl-key">LLM</span></div>
    <div class="bpl-row-r"><span class="bpl-val">nomic</span><span class="bpl-key">EMBED</span></div>
    <div class="bpl-row-r"><span class="bpl-val">8000</span><span class="bpl-key">PORT</span></div>
    <div class="bpl-divider"></div>
    <div class="bpl-row-r"><span class="bpl-val">chromadb</span><span class="bpl-key">STORE</span></div>
    <div class="bpl-row-r"><span class="bpl-val">5</span><span class="bpl-key">TOP-K</span></div>
  </div>
  <div class="boot-inner">
    <div class="boot-tag"><span class="blink-dot"></span>SYSTEM BOOT · v1.0</div>
    <div class="boot-emblem" aria-hidden="true">
      <div class="boot-ring-2"></div>
      <div class="boot-ring-1"></div>
      <div class="boot-hex">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
        </svg>
      </div>
    </div>
    <div class="boot-title-wrap">
      <div id="boot-title" class="boot-title flicker" aria-label="Kirby Risk Assistant"></div>
      <div id="boot-glitch" class="glitch" aria-hidden="true"></div>
    </div>
    <div class="boot-progress">
      <div class="boot-bar" aria-hidden="true"><div id="boot-bar-fill" class="boot-bar-fill"></div></div>
      <div class="boot-meta">
        <span id="boot-status" class="status">Initializing</span>
        <span id="boot-pct" class="pct">0%</span>
      </div>
    </div>
    <div id="boot-log" class="boot-log" aria-hidden="true"></div>
  </div>
</div>

<header>
  <div class="head-row">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="#ff8b1f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 7h16M4 12h10M4 17h16"/>
          <circle cx="18" cy="12" r="2" fill="#ff8b1f" stroke="none"/>
        </svg>
      </div>
      <div class="brand-text">
        <div class="brand-name">Kirby Risk Assistant</div>
        <div class="brand-sub mono">KIRBY RISK · PLANT KNOWLEDGE · RAG</div>
      </div>
    </div>
    <div class="head-actions">
      <div id="status-pill" class="status-pill mono" data-state="checking" role="status" aria-live="polite">
        <span class="status-dot"></span>
        <span class="label">Checking</span>
      </div>
      <button id="new-chat-btn" class="btn" title="Clear conversation">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/>
        </svg>
        <span>New</span>
      </button>
    </div>
  </div>
  <div class="head-strip mono">
    <span class="hash">//</span>
    <span>SESSION ACTIVE</span>
    <span class="sep" aria-hidden="true"></span>
    <span id="session-id">N0DE-<span id="session-suffix">····</span></span>
  </div>
</header>

<div id="messages" role="log" aria-live="polite"></div>

<div id="input-bar">
  <div class="composer">
    <textarea id="question" rows="1" placeholder="Ask about equipment, procedures, logs…" autocomplete="off" autocapitalize="sentences"></textarea>
    <button id="send-btn" title="Send" aria-label="Send message">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M5 12h14M13 6l6 6-6 6"/>
      </svg>
    </button>
  </div>
  <div class="footer-bar mono">
    <span>KIRBY RISK · v1.0</span>
    <span class="kbd">
      <span class="kbd-key">Enter</span> Send
      <span style="margin-left:6px"></span>
      <span class="kbd-key">Shift</span><span>+</span><span class="kbd-key">Enter</span> Newline
    </span>
  </div>
</div>

<script>
  /* ============== BOOT LOADER ============== */
  (function bootLoader() {
    const bootEl    = document.getElementById('boot');
    const titleEl   = document.getElementById('boot-title');
    const glitchEl  = document.getElementById('boot-glitch');
    const fillEl    = document.getElementById('boot-bar-fill');
    const pctEl     = document.getElementById('boot-pct');
    const statusEl  = document.getElementById('boot-status');
    const logEl     = document.getElementById('boot-log');

    const TITLE = 'Kirby Risk Assistant';
    glitchEl.textContent = TITLE;
    const words = TITLE.split(' ');
    words.forEach((word, wi) => {
      const w = document.createElement('span');
      w.className = 'word';
      [...word].forEach((ch, ci) => {
        const s = document.createElement('span');
        s.className = 'ch';
        s.textContent = ch;
        const idx = ci + wi * 6;
        s.style.animationDelay = (0.08 + idx * 0.045) + 's';
        w.appendChild(s);
      });
      titleEl.appendChild(w);
    });

    const LOG_LINES = [
      ['Initializing local runtime', 'ok', 80],
      ['Loading vector store', 'ok', 380],
      ['Connecting embedding model', 'ok', 760],
      ['Warming retrieval pipeline', 'ok', 1140],
      ['Pinging health endpoint', 'ok', 1500],
    ];
    LOG_LINES.forEach(([msg, cls, delay]) => {
      setTimeout(() => {
        const line = document.createElement('div');
        line.className = 'line';
        line.style.animationDelay = '0.05s';
        line.innerHTML = `<span class="${cls}">[OK]</span>  ${msg}...`;
        logEl.appendChild(line);
        while (logEl.children.length > 4) logEl.removeChild(logEl.firstChild);
      }, delay);
    });

    const START = performance.now();
    const DURATION = 2000;
    const MIN_DISPLAY = 2000;
    let healthy = null;
    let dismissed = false;

    function setPct(p) {
      const v = Math.max(0, Math.min(100, p));
      fillEl.style.width = v + '%';
      pctEl.textContent = Math.floor(v) + '%';
    }

    function tick(now) {
      if (dismissed) return;
      const t = Math.min(1, (now - START) / DURATION);
      const target = healthy === true ? 100 : Math.min(95, t * 100);
      setPct(target);
      if (healthy === true && target >= 100) {
        finish();
        return;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    const OFFLINE_AFTER = 3000;
    let offlineTimer = setTimeout(() => {
      if (healthy !== true) {
        statusEl.textContent = 'Server offline';
        statusEl.classList.add('error');
        const line = document.createElement('div');
        line.className = 'line';
        line.style.animationDelay = '0.05s';
        line.innerHTML = `<span class="err">[ERR]</span>  No response from /health... retrying`;
        logEl.appendChild(line);
        while (logEl.children.length > 4) logEl.removeChild(logEl.firstChild);
      }
    }, OFFLINE_AFTER);

    async function pingOnce() {
      try {
        const r = await fetch('/health', { cache: 'no-store' });
        if (!r.ok) throw new Error('bad');
        const j = await r.json().catch(() => ({}));
        if (j.status !== 'ok') throw new Error('not ok');
        healthy = true;
        clearTimeout(offlineTimer);
        statusEl.textContent = 'Online';
        statusEl.classList.remove('error');
      } catch (_) {
        healthy = false;
      }
    }

    async function pollHealth() {
      while (!dismissed && healthy !== true) {
        await pingOnce();
        if (healthy === true) break;
        await new Promise(res => setTimeout(res, 600));
      }
    }
    pollHealth();

    const _p2 = n => n < 10 ? '0'+n : ''+n;
    const uptimeEl = document.getElementById('bpl-uptime');
    if (uptimeEl) {
      const t0 = Date.now();
      setInterval(() => {
        const s = Math.floor((Date.now() - t0) / 1000);
        uptimeEl.textContent = _p2(Math.floor(s / 60)) + ':' + _p2(s % 60);
      }, 1000);
    }

    function finish() {
      if (dismissed) return;
      dismissed = true;
      const elapsed = performance.now() - START;
      const delay = Math.max(0, MIN_DISPLAY - elapsed) + 220;
      setTimeout(() => { bootEl.classList.add('hide'); }, delay);
    }
  })();

  /* ============== CHAT ============== */
  const messagesEl = document.getElementById('messages');
  const questionEl = document.getElementById('question');
  const sendBtn    = document.getElementById('send-btn');
  const statusPill = document.getElementById('status-pill');
  const statusLabel= statusPill.querySelector('.label');
  const sessionSuffixEl = document.getElementById('session-suffix');

  sessionSuffixEl.textContent = Math.random().toString(36).slice(2, 6).toUpperCase();

  function pad2(n) { return n < 10 ? '0' + n : '' + n; }
  function nowStamp() {
    const d = new Date();
    return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
  }

  function renderEmptyState() {
    messagesEl.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'empty-state';
    wrap.innerHTML = `
      <div class="empty-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 6h16v10H7l-3 3V6z"/>
          <path d="M8 10h8M8 13h5"/>
        </svg>
      </div>
      <div class="empty-title">Ready for queries</div>
      <div class="empty-sub">Ask about equipment manuals, maintenance procedures, sensor logs, or anything indexed in the knowledge base.</div>
      <div class="empty-tag mono"><span class="dot"></span>RAG · LOCAL · SECURE</div>
    `;
    messagesEl.appendChild(wrap);
  }

  function clearEmptyState() {
    const empty = messagesEl.querySelector('.empty-state');
    if (empty) empty.remove();
  }

  function renderMarkdown(raw) {
    const inline = s => s
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
      .replace(/\\*([^*\\n]+?)\\*/g, '<em>$1</em>')
      .replace(/`([^`\\n]+)`/g, '<code class="ic">$1</code>');
    const lines = raw.split('\\n');
    let html = '', i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\d+\.\s/.test(line)) {
        html += '<ol>';
        while (i < lines.length && /^\d+\.\s/.test(lines[i]))
          html += `<li>${inline(lines[i++].replace(/^\d+\.\s+/, ''))}</li>`;
        html += '</ol>';
      } else if (/^[-*]\s/.test(line)) {
        html += '<ul>';
        while (i < lines.length && /^[-*]\s/.test(lines[i]))
          html += `<li>${inline(lines[i++].replace(/^[-*]\s+/, ''))}</li>`;
        html += '</ul>';
      } else if (line.trim() === '') {
        i++;
      } else {
        html += `<p>${inline(line)}</p>`; i++;
      }
    }
    return html || `<p>${inline(raw)}</p>`;
  }

  function addBubble(text, role, extraClass = '') {
    clearEmptyState();
    const row = document.createElement('div');
    row.className = `bubble-row ${role}`;

    const meta = document.createElement('div');
    meta.className = 'bubble-meta mono';
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = role === 'user' ? 'YOU' : 'KIRBY';
    const ts = document.createElement('span');
    ts.className = 'ts';
    ts.textContent = nowStamp();
    if (role === 'user') {
      meta.appendChild(ts);
      meta.appendChild(who);
    } else {
      meta.appendChild(who);
      meta.appendChild(ts);
    }

    const bubble = document.createElement('div');
    bubble.className = `bubble ${extraClass}`.trim();

    if (extraClass.includes('thinking')) {
      bubble.innerHTML = `<span>Thinking</span><span class="thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span>`;
    } else if (role === 'bot') {
      bubble.innerHTML = renderMarkdown(text);
    } else {
      bubble.textContent = text;
    }

    row.appendChild(meta);
    row.appendChild(bubble);
    messagesEl.appendChild(row);

    requestAnimationFrame(() => {
      messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
    });
    return { row, bubble };
  }

  async function sendQuestion() {
    const text = questionEl.value.trim();
    if (!text) return;

    questionEl.value = '';
    questionEl.style.height = 'auto';
    sendBtn.disabled = true;

    addBubble(text, 'user');
    const { row: thinkingRow } = addBubble('', 'bot', 'thinking');

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
      });

      thinkingRow.remove();

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        addBubble('Error: ' + (err.detail || res.statusText), 'bot', 'error');
      } else {
        const data = await res.json();
        addBubble(data.answer, 'bot');
      }
    } catch (e) {
      thinkingRow.remove();
      addBubble('Could not reach the server. Check your connection.', 'bot', 'error');
      setStatus('offline');
    }

    sendBtn.disabled = false;
    questionEl.focus();
  }

  sendBtn.addEventListener('click', sendQuestion);

  questionEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  });

  document.getElementById('new-chat-btn').addEventListener('click', async () => {
    try { await fetch('/reset', { method: 'POST' }); } catch (_) {}
    messagesEl.innerHTML = '';
    renderEmptyState();
    sessionSuffixEl.textContent = Math.random().toString(36).slice(2, 6).toUpperCase();
    questionEl.focus();
  });

  questionEl.addEventListener('input', () => {
    questionEl.style.height = 'auto';
    questionEl.style.height = Math.min(questionEl.scrollHeight, 140) + 'px';
  });

  function setStatus(state) {
    statusPill.dataset.state = state;
    if (state === 'online')        statusLabel.textContent = 'Online';
    else if (state === 'offline')  statusLabel.textContent = 'Offline';
    else                           statusLabel.textContent = 'Checking';
  }

  async function pingHealth() {
    try {
      const r = await fetch('/health', { cache: 'no-store' });
      if (!r.ok) throw new Error('bad');
      const j = await r.json().catch(() => ({}));
      setStatus(j.status === 'ok' ? 'online' : 'offline');
    } catch (_) {
      setStatus('offline');
    }
  }

  renderEmptyState();
  pingHealth();
  setInterval(pingHealth, 15000);
  questionEl.focus();
</script>
</body>
</html>""")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not ready")
    if _is_parts_query(body.question):
        response = parts_engine.query(body.question)
    elif _is_employee_query(body.question):
        response = employee_engine.query(body.question)
    elif _is_audio_query(body.question):
        response = audio_engine.query(body.question)
    else:
        response = chat_engine.chat(body.question)
    return {"answer": str(response)}


@app.post("/reset")
def reset():
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not ready")
    chat_engine.reset()
    return {"status": "conversation cleared"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
