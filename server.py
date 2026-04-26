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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_engine

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
        similarity_top_k=5,
        verbose=True,
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KRPM RAG Assistant</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f0f2f5;
      display: flex;
      flex-direction: column;
      height: 100dvh;
    }

    header {
      background: #1a1a2e;
      color: #fff;
      padding: 14px 20px;
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: 0.3px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    #new-chat-btn {
      background: transparent;
      border: 1px solid rgba(255,255,255,0.4);
      color: #fff;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: background 0.15s;
    }
    #new-chat-btn:hover { background: rgba(255,255,255,0.15); }

    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .bubble-row {
      display: flex;
      align-items: flex-end;
      gap: 8px;
    }
    .bubble-row.user  { flex-direction: row-reverse; }
    .bubble-row.bot   { flex-direction: row; }

    .bubble {
      max-width: 78%;
      padding: 10px 14px;
      border-radius: 18px;
      font-size: 0.95rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .bubble-row.user .bubble {
      background: #2563eb;
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .bubble-row.bot .bubble {
      background: #fff;
      color: #1a1a1a;
      border-bottom-left-radius: 4px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .bubble.thinking {
      color: #888;
      font-style: italic;
    }
    .bubble.error {
      background: #fee2e2;
      color: #b91c1c;
    }

    #input-bar {
      display: flex;
      gap: 8px;
      padding: 12px;
      background: #fff;
      border-top: 1px solid #e5e7eb;
      flex-shrink: 0;
    }

    #question {
      flex: 1;
      padding: 10px 14px;
      border: 1px solid #d1d5db;
      border-radius: 24px;
      font-size: 1rem;
      outline: none;
      resize: none;
      max-height: 120px;
      overflow-y: auto;
    }
    #question:focus { border-color: #2563eb; }

    #send-btn {
      background: #2563eb;
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 44px;
      height: 44px;
      font-size: 1.2rem;
      cursor: pointer;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.15s;
    }
    #send-btn:hover  { background: #1d4ed8; }
    #send-btn:disabled { background: #93c5fd; cursor: default; }
  </style>
</head>
<body>

<header>
  <span>KRPM RAG Assistant</span>
  <button id="new-chat-btn" title="Clear conversation">New Chat</button>
</header>

<div id="messages"></div>

<div id="input-bar">
  <textarea id="question" rows="1" placeholder="Ask a question about your data..."></textarea>
  <button id="send-btn" title="Send">&#9658;</button>
</div>

<script>
  const messagesEl = document.getElementById('messages');
  const questionEl = document.getElementById('question');
  const sendBtn    = document.getElementById('send-btn');

  function addBubble(text, role, extraClass = '') {
    const row = document.createElement('div');
    row.className = `bubble-row ${role}`;
    const bubble = document.createElement('div');
    bubble.className = `bubble ${extraClass}`;
    bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  async function sendQuestion() {
    const text = questionEl.value.trim();
    if (!text) return;

    questionEl.value = '';
    questionEl.style.height = 'auto';
    sendBtn.disabled = true;

    addBubble(text, 'user');
    const thinkingBubble = addBubble('Thinking...', 'bot', 'thinking');

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
      });

      thinkingBubble.parentElement.remove();

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        addBubble('Error: ' + (err.detail || res.statusText), 'bot', 'error');
      } else {
        const data = await res.json();
        addBubble(data.answer, 'bot');
      }
    } catch (e) {
      thinkingBubble.parentElement.remove();
      addBubble('Could not reach the server. Is it running?', 'bot', 'error');
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
    await fetch('/reset', { method: 'POST' });
    messagesEl.innerHTML = '';
  });

  // Auto-resize textarea
  questionEl.addEventListener('input', () => {
    questionEl.style.height = 'auto';
    questionEl.style.height = questionEl.scrollHeight + 'px';
  });
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
