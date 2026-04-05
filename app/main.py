from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import RAGService

app = FastAPI(title="Sprint 7 RAG Bot")

rag_service = RAGService()


class AskRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return rag_service.ask(request.query)
