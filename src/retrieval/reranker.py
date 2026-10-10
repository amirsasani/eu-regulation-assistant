import utils
from sentence_transformers import CrossEncoder

RERANKER_MODEL = utils.DEFAULT_RERANKER_MODEL

def create_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(query: str, documents: list[dict], reranker: CrossEncoder, top_k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not documents:
        return []

    pairs = [
        (
            query,
            (
                f"Article {document['article_number']}: "
                f"{document['article_title']}\n"
                f"{document['text']}"
            ),
        )
        for document in documents
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    scored_documents = []

    for document, score in zip(documents, scores):
        scored_documents.append({
            **document,
            "reranker_score": float(score),
        })

    scored_documents.sort(key=lambda document: document["reranker_score"], reverse=True)

    return scored_documents[:top_k]