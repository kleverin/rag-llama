import os
import sys
import shutil
import pathlib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
DATA_PATH = os.getenv("DATA_PATH", "./data")


def row_to_text(row: pd.Series, sheet_name: str) -> str:
    parts = [f"Sheet: {sheet_name}"]
    for col, val in row.items():
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


def load_excel_documents():
    from llama_index.core import Document

    data_dir = pathlib.Path(DATA_PATH)
    xlsx_files = sorted(data_dir.rglob("*.xlsx"))

    if not xlsx_files:
        print(f"No .xlsx files found under {DATA_PATH}")
        sys.exit(1)

    print(f"Found {len(xlsx_files)} Excel file(s)")
    documents = []

    for file_path in xlsx_files:
        print(f"\nReading: {file_path.name}")
        try:
            sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)
        except Exception as e:
            print(f"  WARNING: Could not read {file_path.name}: {e}")
            continue

        for sheet_name, df in sheets.items():
            df = df.dropna(how="all")
            print(f"  Sheet '{sheet_name}': {len(df)} rows")
            for row_idx, row in df.iterrows():
                text = row_to_text(row, sheet_name)
                if not text.strip():
                    continue
                doc = Document(
                    text=text,
                    metadata={
                        "source": file_path.name,
                        "sheet": str(sheet_name),
                        "row": int(row_idx),
                    },
                )
                documents.append(doc)

    print(f"\nTotal documents (rows) prepared: {len(documents)}")
    return documents


def main():
    if pathlib.Path(CHROMA_PATH).exists():
        print(f"ChromaDB already exists at '{CHROMA_PATH}'. Skipping ingestion.")
        print("Delete the chroma_db/ folder to re-ingest.")
        return

    print("=== Starting ingestion ===")
    documents = load_excel_documents()

    print("\nInitializing embedding model (nomic-embed-text via Ollama)...")
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext
    import chromadb

    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )

    print(f"Initializing ChromaDB at '{CHROMA_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection("rag_collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Embedding and indexing documents (this may take a while)...")
    try:
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
    except Exception as e:
        print(f"\nERROR during embedding: {e}")
        print(f"Cleaning up partial chroma_db at '{CHROMA_PATH}'...")
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        print("Deleted. Fix the error above, then re-run ingest.py.")
        sys.exit(1)

    print(f"\n=== Ingestion complete! Vectors stored at '{CHROMA_PATH}' ===")


if __name__ == "__main__":
    main()
