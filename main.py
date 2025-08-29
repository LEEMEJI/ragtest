from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, io
import fitz  # pymupdf
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

app = FastAPI(title="RAG Server")

# ===== 1) 전역 설정 =====
DATA_DIR = os.getenv("DATA_DIR", "/data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

# 멀티링궐 경량 모델(한글 OK, 속도 빠름)
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embedder = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_chunks")
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ===== 2) 유틸 =====
def pdf_to_text_chunks(pdf_bytes: bytes, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    """PDF → 텍스트 추출 → 일정 길이로 청크 쪼개기"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        t = page.get_text("text")
        if t:
            texts.append(t)
    full = "\n".join(texts)
    full = " ".join(full.split())  # 공백 정리

    chunks = []
    i = 0
    while i < len(full):
        chunk = full[i:i+max_chars]
        chunks.append(chunk)
        i += max_chars - overlap
    return [c for c in chunks if c.strip()]

def embed(texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

# ===== 3) API =====
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "collection": COLLECTION_NAME}

@app.get("/")
def root():
    return {"msg": "Hello, RAG server is running!!!"}

class IngestRequest(BaseModel):
    doc_id: str
    chunks: List[str]

@app.post("/upload")
async def upload(file: UploadFile = File(...), doc_id: Optional[str] = Form(None)):
    """PDF 업로드 → 텍스트 청크 반환(미저장). 필요하면 /ingest로 저장 요청."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF만 업로드하세요.")
    pdf_bytes = await file.read()
    chunks = pdf_to_text_chunks(pdf_bytes)
    # 너무 많을 때 클라이언트가 부담되면 일부만 미리보기 가능
    return {"doc_id": doc_id or os.path.splitext(file.filename)[0], "count": len(chunks), "chunks": chunks[:50]}

@app.post("/ingest")
def ingest(req: IngestRequest):
    """청크들을 벡터화하여 ChromaDB에 저장."""
    if not req.chunks:
        raise HTTPException(400, "chunks가 비어있습니다.")
    ids = [f"{req.doc_id}_{i}" for i in range(len(req.chunks))]
    embs = embed(req.chunks)
    metas = [{"doc_id": req.doc_id, "idx": i} for i in range(len(req.chunks))]
    collection.add(documents=req.chunks, embeddings=embs, metadatas=metas, ids=ids)
    return {"doc_id": req.doc_id, "added": len(req.chunks)}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/query")
def query(req: QueryRequest):
    q_emb = embed([req.query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=req.top_k)
    items = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    for d, m, i in zip(docs, metas, ids):
        items.append({"id": i, "doc": d, "meta": m})
    return {"query": req.query, "results": items}