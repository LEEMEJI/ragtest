from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List
import os
import io

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pdfminer.high_level import extract_text

oai = OpenAI()
app = FastAPI(title="RAG Server")
# 영속 경로(도커에서 -v /root/ragtest-data:/data 로 마운트됨)
DATA_DIR = Path("/data")
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

DATA_DIR = os.getenv("DATA_DIR", "/data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)



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

    # 1) 벡터 검색
    col = client.get_collection(doc_id)
    q_emb = embedder.encode([question], convert_to_numpy=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=top_k)

    contexts = res.get("documents", [[]])[0]
    ids = res.get("ids", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    if not contexts:
        return {
            "answer": "관련된 문서 조각을 찾지 못했습니다.",
            "contexts": [],
            "ids": [],
            "metadatas": [],
        }

    # 2) 컨텍스트 구성 (길이 너무 길어지는 것 방지)
    #    필요하면 길이 제한/요약 로직 추가 가능
    joined_ctx = "\n\n---\n\n".join(contexts[:top_k])

    # 3) OpenAI에 질의 (gpt-4o-mini 예시)
    system_prompt = (
        "너는 주어진 문서 컨텍스트를 근거로 간결하고 정확하게 답하는 어시스턴트야. "
        "컨텍스트에 없는 정보는 추측하지 말고 모른다고 말해. "
        "가능하면 한국어로 답해줘."
    )
    user_prompt = (
        f"[질문]\n{question}\n\n"
        f"[컨텍스트]\n{joined_ctx}\n\n"
        "위 컨텍스트만 근거로 답변해줘."
    )

    chat = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = chat.choices[0].message.content

    return {
        "answer": answer,
        "contexts": contexts,
        "ids": ids,
        "metadatas": metas,
    }


