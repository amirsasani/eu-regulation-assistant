from pathlib import Path

from sentence_transformers import SentenceTransformer

import utils
from scripts.search_hybrid import MODEL_NAME, search_hybrid
from src.generation.client import generate_answer
from src.generation.prompt import build_messages
from src.retrieval.bm25 import create_bm25_index
from src.retrieval.reranker import create_reranker


DOCUMENTS_FILES = {
    "en": utils.DATA_PROCESSED_PATH / "ai-act-articles.json",
    "it": utils.DATA_PROCESSED_PATH / "ai-act-articles-it.json",
}


class RagService:
    def __init__(
        self,
        documents_file: Path | None = None,
        language: str = "en",
        documents_files: dict[str, Path] | None = None,
    ):
        if documents_file is not None and documents_files is not None:
            raise ValueError(
                "Use either documents_file or documents_files, not both."
            )

        if documents_files is not None:
            corpus_paths = documents_files
        elif documents_file is not None:
            corpus_paths = {language: documents_file}
        else:
            corpus_paths = DOCUMENTS_FILES

        self.documents_by_language = {
            corpus_language: utils.load_json_file(path)
            for corpus_language, path in corpus_paths.items()
        }
        self.bm25_retrievers = {
            corpus_language: create_bm25_index(documents, corpus_language)
            for corpus_language, documents in self.documents_by_language.items()
        }
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        self.reranker_model = create_reranker()

    def ask(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 10,
        llm_model: str | None = None,
        language: str = "en",
    ):
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if language not in utils.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        if language not in self.bm25_retrievers:
            raise ValueError(f"No indexed corpus is available for language: {language}")

        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")

        sources = search_hybrid(
            query=question,
            bm25_retriever=self.bm25_retrievers[language],
            model=self.embedding_model,
            top_k=top_k,
            candidate_k=candidate_k,
            language=language,
            reranker_model=self.reranker_model,
        )

        messages = build_messages(question=question, documents=sources, language=language)

        generation = generate_answer(messages=messages, model=llm_model)

        return generation, sources
