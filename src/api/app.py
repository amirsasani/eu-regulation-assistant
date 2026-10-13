import sys
import io
from fastapi import FastAPI
from src.services.rag_service import ask as rag_ask


app = FastAPI()


@app.get("/health")
async def health():
    ollama_connection = rag_ask(
        question="What is the AI Act?",
        top_k=1,
        candidate_k=1,
        language="en",
        llm_model="default"
    )

    if ollama_connection:
        return {"status": "ok"}
    else:
        return {"status": "error", "message": "Failed to connect to Ollama."}

@app.post("/ask")
async def ask_question(question: str, top_k: int = 5, candidate_k: int = 10, language: str = "en", llm_model: str = "default"):
    generation, sources = rag_ask(
        question=question,
        top_k=top_k,
        candidate_k=candidate_k,
        language=language,
        llm_model=llm_model
    )

    return {
        "answer": generation['answer'],
        "sources": sources
    }