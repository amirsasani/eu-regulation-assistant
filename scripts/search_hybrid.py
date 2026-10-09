import argparse

from sentence_transformers import SentenceTransformer

import utils

from src.database.vector_repository import search_similar
from src.retrieval.bm25 import create_bm25_index, search as bm25_search
from src.retrieval.hybrid import reciprocal_rank_fusion


DOCUMENTS_FILE = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")
MODEL_NAME = utils.DEFAULT_SEMANTIC_MODEL_NAME


def create_query_embedding(query: str, model: SentenceTransformer):
    return model.encode(f"query: {query}", normalize_embeddings=True)


def search_hybrid(
    query: str,
    bm25_retriever,
    model: SentenceTransformer,
    top_k: int,
    candidate_k: int,
    language: str,
):
    bm25_results, _ = bm25_search(
        query=query,
        retriever=bm25_retriever,
        top_k=candidate_k,
        language=language,
    )
    bm25_documents = list(bm25_results[0])

    query_embedding = create_query_embedding(query, model)
    semantic_documents = search_similar(query_embedding=query_embedding, top_k=candidate_k)

    return reciprocal_rank_fusion(
        bm25_documents=bm25_documents,
        semantic_documents=semantic_documents,
        top_k=top_k,
    )


def show_results(results: list[dict]):
    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. Article {result['article_number']} — {result['article_title']}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Retrieved by: {', '.join(result['retrieved_by'])}")
        print(f"Chunk: {result['chunk_id']}")
        print(result["text"][:400])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=20)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    documents = utils.load_json_file(DOCUMENTS_FILE)
    bm25_retriever = create_bm25_index(documents, args.language)
    model = SentenceTransformer(MODEL_NAME)

    results = search_hybrid(
        query=args.query,
        bm25_retriever=bm25_retriever,
        model=model,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        language=args.language,
    )
    show_results(results)


if __name__ == "__main__":
    main()