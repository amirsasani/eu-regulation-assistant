import os
import requests
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status

from src.services.rag_service import RagService

from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ReadinessResponse,
    SourceResponse,
)

from src.database.connection import get_connection

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = RagService()
    yield


app = FastAPI(
    title="EU Regulation Intelligence Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        503: {
            "model": ReadinessResponse,
            "description": "One or more dependencies are unavailable",
        }
    },
)
def readiness(response: Response):
    services = {
        "postgres": {"status": "unavailable"},
        "ollama": {"status": "unavailable"},
    }

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) AS count FROM regulation_chunks"
                )
                chunk_count = cursor.fetchone()["count"]

        services["postgres"] = {
            "status": "ok",
            "chunks": chunk_count,
        }
    except Exception:
        logger.exception("PostgreSQL readiness check failed")

    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

        model = os.getenv("OLLAMA_MODEL")

        ollama_response = requests.get(f"{base_url}/api/tags", timeout=3)
        ollama_response.raise_for_status()

        installed_models = {item["name"] for item in ollama_response.json().get("models", [])}

        if model not in installed_models:
            raise RuntimeError(f"Configured Ollama model is not installed: {model}")

        services["ollama"] = {
            "status": "ok",
            "model": model,
        }
    except Exception:
        logger.exception("Ollama readiness check failed")

    ready = all(
        service["status"] == "ok"
        for service in services.values()
    )

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ok" if ready else "unavailable",
        services=services,
    )

@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request):
    service: RagService = request.app.state.rag_service

    try:
        generation, sources = service.ask(question=payload.question, top_k=payload.top_k, candidate_k=payload.candidate_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("RAG request failed")
        raise HTTPException(status_code=503, detail="The RAG service is currently unavailable.") from error

    response_sources = [
        SourceResponse(
            chunk_id=source["chunk_id"],
            article_number=str(source["article_number"]),
            article_title=source.get("article_title"),
            paragraph_number=source.get("paragraph_number"),
            text=source["text"],
            source_url=source.get("source_url"),
            reranker_score=(
                float(source["reranker_score"])
                if source.get("reranker_score") is not None
                else None
            ),
        )
        for source in sources
    ]

    return AskResponse(answer=generation["answer"], model=generation["model"], sources=response_sources)