from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.services.rag_service import RagService

from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceResponse,
)

import logging

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


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request):
    service: RagService = request.app.state.rag_service

    try:
        generation, sources = service.ask(question=payload.question, top_k=payload.top_k, candidate_k=payload.candidate_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
        # raise HTTPException(status_code=503, detail="The RAG service is currently unavailable.") from error

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