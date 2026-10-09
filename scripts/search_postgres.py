import argparse
import json
from pathlib import Path

import utils

from src.database.vector_repository import search_similar
from src.retrieval.semantic import create_query_embedding, load_model

DEFAULT_MANIFEST = (utils.DATA_INDEXES_PATH / "semantic/manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search AI Act chunks in PostgreSQL.")

    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    manifest = utils.load_json_file(args.manifest)

    print(f"Loading model: {manifest['model']}")

    model = load_model(manifest["model"])

    query_embedding = create_query_embedding(query=args.query, model=model)

    results = search_similar(query_embedding=query_embedding, top_k=args.top_k)

    for result in results:
        print(f"\nArticle {result['article_number']} — {result['article_title']}")
        print(f"Similarity: {result['similarity']:.6f}")
        print(f"Chunk: {result['chunk_id']}")
        print(result["text"][:300])


if __name__ == "__main__":
    main()