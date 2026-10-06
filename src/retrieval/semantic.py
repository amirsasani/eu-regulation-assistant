import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "intfloat/multilingual-e5-base"


def load_model(
    model_name: str = DEFAULT_MODEL,
) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_passage_text(document: dict) -> str:
    return (
        f"passage: Article {document['article_number']}. "
        f"{document['article_title']}. "
        f"{document['text']}"
    )


def create_document_embeddings(
    documents: list[dict],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
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