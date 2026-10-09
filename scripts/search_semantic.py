import argparse
from pathlib import Path

import utils

import numpy as np

from src.retrieval.semantic import load_model, search as semantic_search


DEFAULT_DOCUMENTS = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")

DEFAULT_INDEX_DIRECTORY = (utils.DATA_INDEXES_PATH / "semantic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search AI Act chunks using NumPy.")

    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--index-directory", type=Path, default=DEFAULT_INDEX_DIRECTORY)

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    embeddings_path = (args.index_directory / "embeddings.npy")
    manifest_path = (args.index_directory / "manifest.json")

    documents = utils.load_json_file(args.documents)
    manifest = utils.load_json_file(manifest_path)

    embeddings = np.load(embeddings_path, allow_pickle=False)

    if len(documents) != len(embeddings):
        raise ValueError("Document and embedding counts do not match. Rebuild the semantic index.")

    if (embeddings.shape[1] != manifest["embedding_dimension"]):
        raise ValueError("Embedding dimension does not match the manifest.")

    print(f"Loading model: {manifest['model']}")

    model = load_model(manifest["model"])

    results = semantic_search(
        query=args.query,
        model=model,
        documents=documents,
        document_embeddings=embeddings,
        top_k=args.top_k,
    )

    for result in results:
        print(f"\nArticle {result['article_number']} — {result['article_title']}")
        print(f"Similarity: {result['score']:.6f}")
        print(f"Chunk: {result['chunk_id']}")
        print(result["text"][:300])


if __name__ == "__main__":
    main()