import numpy as np
from sentence_transformers import SentenceTransformer

import utils


DEFAULT_MODEL = utils.DEFAULT_SEMANTIC_MODEL_NAME


def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_passage_text(document: dict) -> str:
    return (
        f"passage: Article {document['article_number']}. "
        f"{document['article_title']}. "
        f"{document['text']}"
    )


def create_document_embeddings(documents: list[dict], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    passages = [
        build_passage_text(document)
        for document in documents
    ]

    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if embeddings.shape[0] != len(documents):
        raise RuntimeError(
            "Embedding count does not match document count."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embeddings contain NaN or infinite values."
        )

    return embeddings

def create_query_embedding(query: str, model: SentenceTransformer) -> np.ndarray:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    return model.encode(
        f"query: {query}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def search(
        query: str, 
        model: SentenceTransformer, 
        documents: list[dict], 
        document_embeddings: np.ndarray, 
        top_k: int = 5
    ) -> list[dict]:

    if len(documents) != len(document_embeddings):
        raise ValueError("Document and embedding counts do not match.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    top_k = min(top_k, len(documents))

    query_embedding = create_query_embedding(query, model)

    scores = document_embeddings @ query_embedding

    top_indices = np.argsort(-scores)[:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        document = documents[int(index)]

        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "chunk_id": document["chunk_id"],
            "article_number": str(document["article_number"]),
            "article_title": document["article_title"],
            "text": document["text"],
        })

    return results