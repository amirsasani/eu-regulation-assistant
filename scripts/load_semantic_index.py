import argparse
import json
from pathlib import Path

import numpy as np

import utils

from src.database.schema import EMBEDDING_DIMENSION
from src.database.vector_repository import upsert_documents


DEFAULT_DOCUMENTS = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")

DEFAULT_INDEX_DIRECTORY = (utils.DATA_INDEXES_PATH / "semantic")


def validate_documents(documents: list[dict]) -> None:
    if not isinstance(documents, list):
        raise ValueError("Documents file must contain a JSON array.")

    if not documents:
        raise ValueError("Documents file is empty.")

    required_fields = {
        "chunk_id",
        "article_number",
        "text",
    }

    chunk_ids = set()

    for position, document in enumerate(documents):
        missing = required_fields - document.keys()

        if missing:
            raise ValueError(f"Document {position} is missing: {sorted(missing)}")

        chunk_id = document["chunk_id"]

        if chunk_id in chunk_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")

        chunk_ids.add(chunk_id)


def validate_index(documents: list[dict], embeddings: np.ndarray,manifest: dict) -> None:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if len(documents) != len(embeddings):
        raise ValueError("Document and embedding counts do not match.")

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected {EMBEDDING_DIMENSION} dimensions, received {embeddings.shape[1]}.")

    if manifest["document_count"] != len(documents):
        raise ValueError("Manifest document count does not match.")

    if (manifest["embedding_dimension"] != EMBEDDING_DIMENSION):
        raise ValueError("Manifest embedding dimension does not match.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")


def main() -> None:
    parser = argparse.ArgumentParser(description=("Load regulation documents and embeddings into PostgreSQL."))

    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--index-directory", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--document-id", default="32024R1689")

    args = parser.parse_args()

    embeddings_path = (args.index_directory / "embeddings.npy")
    manifest_path = (args.index_directory / "manifest.json")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    documents = utils.load_json_file(args.documents)
    manifest = utils.load_json_file(manifest_path)

    embeddings = np.load(embeddings_path, allow_pickle=False)

    validate_documents(documents)

    validate_index(documents, embeddings, manifest)

    loaded_count = upsert_documents(
        documents=documents,
        embeddings=embeddings,
        default_document_id=args.document_id,
    )

    print(f"Loaded {loaded_count} regulation chunks into PostgreSQL.")


if __name__ == "__main__":
    main()