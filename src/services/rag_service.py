from pathlib import Path

from sentence_transformers import SentenceTransformer

import utils
from scripts.search_hybrid import MODEL_NAME, search_hybrid
from src.generation.client import generate_answer
from src.generation.prompt import build_messages
from src.retrieval.bm25 import create_bm25_index
from src.retrieval.reranker import create_reranker


DOCUMENTS_FILE = utils.DATA_PROCESSED_PATH / "ai-act-articles.json"


class RagService:
    def __init__(
        self,
        documents_file: Path = DOCUMENTS_FILE,
        language: str = "en",
    ):
        self.language = language
        self.documents = utils.load_json_file(documents_file)
        self.bm25_retriever = create_bm25_index(self.documents, language)
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        self.reranker_model = create_reranker()

    def ask(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 10,
        llm_model: str | None = None,
    ):
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")

        sources = search_hybrid(
            query=question,
            bm25_retriever=self.bm25_retriever,
            model=self.embedding_model,
            top_k=top_k,
            candidate_k=candidate_k,
            language=self.language,
            reranker_model=self.reranker_model,
        )

        messages = build_messages(question=question, documents=sources)

        generation = generate_answer(messages=messages, model=llm_model)

        return generation, sources