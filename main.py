from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import io

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pdfminer.high_level import extract_text

app = FastAPI(title="RAG Server")
# 영속 경로(도커에서 -v /root/ragtest-data:/data 로 마운트됨)
DATA_DIR = Path("/data")
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

DATA_DIR = os.getenv("DATA_DIR", "/data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
client = PersistentClient(path=str(CHROMA_DIR))
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap if chunk_size > overlap else chunk_size
    return chunks

# ===== 3) API =====
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "collection": COLLECTION_NAME}

@app.get("/")
def root():
    return {"msg": "Hello, RAG server is running!!!"}

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form(...)
):
    # 1) 파일 저장 경로
    safe_doc_id = "".join(c for c in doc_id if c.isalnum() or c in ("-", "_"))
    save_path = UPLOAD_DIR / f"{safe_doc_id}.pdf"

    # 2) 업로드 파일 저장 (스트리밍)
    try:
        with save_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"save error: {e}")

    # 3) 텍스트 추출
    try:
        text = extract_text(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pdf parse error: {e}")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="no text extracted from pdf")

    # 4) 청크 → 임베딩 → Chroma 저장
    chunks = chunk_text(text, chunk_size=120, overlap=20)  # 필요에 맞게 조절
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunks produced")

    embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()

    col = client.get_or_create_collection(name=safe_doc_id)
    ids = [f"{safe_doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": safe_doc_id, "idx": i} for i in range(len(chunks))]
    col.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

    return JSONResponse({"ok": True, "doc_id": safe_doc_id, "chunks": len(chunks), "saved_to": str(save_path)})

@app.post("/ask")
async def ask(payload: dict):
    question = payload.get("question")
    doc_id = payload.get("doc_id")
    top_k = int(payload.get("top_k", 3))
    if not question or not doc_id:
        raise HTTPException(status_code=400, detail="question and doc_id required")

    col = client.get_collection(doc_id)
    q_emb = embedder.encode([question], convert_to_numpy=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=top_k)
    contexts = res.get("documents", [[]])[0]
    ids = res.get("ids", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    # 여기에 LLM 호출/조립 로직을 붙이면 됨(지금은 컨텍스트만 반환)
    return {"answer": None, "contexts": contexts, "ids": ids, "metadatas": metas}


