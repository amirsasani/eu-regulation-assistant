import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = (PROJECT_ROOT / "data/processed/ai-act-articles.json")
INDEX_DIRECTORY = (PROJECT_ROOT / "data/indexes/semantic")
EMBEDDINGS_PATH = (INDEX_DIRECTORY / "embeddings.npy")
MANIFEST_PATH = (INDEX_DIRECTORY / "manifest.json")


def calculate_file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            digest.update(block)

    return digest.hexdigest()


def load_documents(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_passage_text(document: dict) -> str:
    return (
        f"passage: Article {document['article_number']}. "
        f"{document['article_title']}. "
        f"{document['text']}"
    )


def main() -> None:
    documents = load_documents(DOCUMENTS_PATH)

    passages = [
        build_passage_text(document)
        for document in documents
    ]

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        passages,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "Embedding count does not match document count."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embeddings contain NaN or infinite values."
        )

    INDEX_DIRECTORY.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_PATH, embeddings)

    manifest = {
        "model": MODEL_NAME,
        "normalized": True,
        "document_count": len(documents),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "source_file": str(DOCUMENTS_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": calculate_file_hash(DOCUMENTS_PATH),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()