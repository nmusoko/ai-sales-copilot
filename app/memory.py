# app/memory.py
"""
Vector Memory: chunk transcripts, embed, and persist to a local FAISS index.
- Default embeddings: HuggingFace 'all-MiniLM-L6-v2' (offline-friendly).
- Optional: set EMBEDDINGS=openai in .env to use OpenAIEmbeddings.
- Persists index to: data/memory/faiss_index/
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    # Newer LangChain (>=0.2)
    from langchain_core.documents import Document
except ImportError:
    # Older versions fallback
    from langchain.schema import Document


# Embedding backends
EMB_BACKEND = os.getenv("EMBEDDINGS", "hf").lower()

# Lazy imports (so you don’t need both backends installed to import this module)
def _make_embedder():
    if EMB_BACKEND == "openai":
        # Requires: langchain-openai and OPENAI_API_KEY
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    else:
        # Requires: langchain-huggingface + sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        # MiniLM = small & fast, good quality
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Paths
MEM_DIR = Path("data/memory")
MEM_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = MEM_DIR / "faiss_index"


# ---------------------------
# Helpers
# ---------------------------
def _split_text(text: str) -> List[str]:
    """
    Split long transcripts into overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # ~ 150–200 tokens
        chunk_overlap=120,    # keeps context continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _load_index_or_none():
    """
    Load FAISS if it exists, else return None.
    """
    if INDEX_PATH.exists():
        # allow_dangerous_deserialization is required for FAISS local load
        return FAISS.load_local(str(INDEX_PATH), _make_embedder(), allow_dangerous_deserialization=True)
    return None


def _save_index(index: FAISS):
    index.save_local(str(INDEX_PATH))


# ---------------------------
# Public API
# ---------------------------
def index_transcript(pkg, overwrite: bool = False) -> int:
    """
    Index a transcript package (from app.loader.TranscriptPackage).
    - pkg.text: the transcript text
    - pkg.meta: metadata (file_name, created_at, language, model, duration_sec)
    - overwrite: if True, attempts to remove previous chunks for this 'stem' before adding (best-effort)
    Returns: number of chunks added.
    """
    load_dotenv()  # ensure .env is read for embedding backend/API keys

    text = pkg.text or ""
    if not text.strip():
        return 0

    chunks = _split_text(text)

    # Build documents with rich metadata
    docs = []
    for i, ch in enumerate(chunks):
        docs.append(
            Document(
                page_content=ch,
                metadata={
                    "stem": pkg.stem,
                    "chunk_id": f"{pkg.stem}:{i}",
                    "file_name": pkg.meta.file_name,
                    "language": pkg.meta.language,
                    "model": pkg.meta.model,
                    "duration_sec": pkg.meta.duration_sec,
                    "created_at": pkg.meta.created_at,
                }
            )
        )

    # Load existing index if present
    embedder = _make_embedder()
    index = _load_index_or_none()

    # (Optional) Best-effort overwrite: rebuild without previous chunks of this stem
    if overwrite and index is not None:
        # FAISS doesn't support delete-by-filter; rebuild is simplest:
        # 1) pull all docs (not directly available), so we’ll do a naive approach:
        #    save current index texts to a temporary list via similarity on a wildcard set.
        #    For MVP, we’ll skip strict removal and just add fresh docs with same stem.
        #    (A later improvement: store a separate manifest of doc_ids and rebuild.)
        pass

    if index is None:
        # First-time create
        index = FAISS.from_documents(docs, embedder)
    else:
        index.add_documents(docs)

    _save_index(index)
    return len(docs)


def search_similar(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic search over the FAISS index.
    Returns a list of {text, score, metadata}.
    """
    load_dotenv()
    index = _load_index_or_none()
    if index is None:
        return []

    results = index.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in results:
        out.append({
            "text": doc.page_content,
            "score": float(score),     # lower is better in FAISS distance
            "metadata": doc.metadata,
        })
    return out
def qa_stub_answer(question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic fallback: returns a short answer-like summary using top hits.
    """
    if not hits:
        return {"answer": "I don't see that in the transcripts.", "citations": []}
    # Take 1–2 short quotes
    citations = []
    snippets = []
    for h in hits[:2]:
        meta = h["metadata"]
        quote = h["text"].strip().replace("\n", " ")
        if len(quote) > 240:
            quote = quote[:240] + " …"
        citations.append({
            "stem": meta.get("stem", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "quote": quote
        })
        snippets.append(quote)
    answer = "Here’s what I found:\n- " + "\n- ".join(snippets)
    return {"answer": answer, "citations": citations}
# --- Maintenance helpers ---

def reset_index() -> None:
    """Delete the FAISS index folder to start fresh."""
    import shutil
    if INDEX_PATH.exists():
        shutil.rmtree(INDEX_PATH)

def index_all(pkgs, overwrite: bool = False) -> int:
    """
    Bulk index a list of transcript packages.
    Returns total chunks added.
    """
    total = 0
    for pkg in pkgs:
        total += index_transcript(pkg, overwrite=overwrite)
    return total
