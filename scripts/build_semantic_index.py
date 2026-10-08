import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import utils

import numpy as np

from src.retrieval.semantic import (
    DEFAULT_MODEL,
    create_document_embeddings,
    load_model,
)

DEFAULT_DOCUMENTS_PATH = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")

DEFAULT_OUTPUT_DIRECTORY = (utils.DATA_INDEXES_PATH / "semantic")


def load_documents(path: Path):
    documents = utils.load_json_file(path)

    if not isinstance(documents, list):
        raise ValueError(
            "Documents file must contain a JSON array."
        )

    if not documents:
        raise ValueError("Documents file is empty.")

    return documents


def calculate_file_hash(path: Path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            digest.update(block)

    return digest.hexdigest()


def build_index(documents_path: Path, output_directory: Path, model_name: str, batch_size: int):
    documents = load_documents(documents_path)

    print(f"Loading model: {model_name}")
    model = load_model(model_name)

    print(
        f"Generating embeddings for "
        f"{len(documents)} documents..."
    )

    embeddings = create_document_embeddings(documents=documents, model=model, batch_size=batch_size)

    output_directory.mkdir(parents=True, exist_ok=True)

    embeddings_path = (output_directory / "embeddings.npy")
    manifest_path = (output_directory / "manifest.json")

    np.save(embeddings_path, embeddings)

    manifest = {
        "model": model_name,
        "normalized": True,
        "document_count": len(documents),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "source_file": str(documents_path.relative_to(utils.PROJECT_ROOT)),
        "source_sha256": calculate_file_hash(documents_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Saved embeddings: {embeddings_path}")
    print(f"Saved manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the semantic vector index.")

    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    build_index(
        documents_path=args.documents,
        output_directory=args.output_directory,
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()