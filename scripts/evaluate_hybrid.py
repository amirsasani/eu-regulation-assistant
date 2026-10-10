import argparse
import json
import time
from pathlib import Path
from pprint import pprint

import utils

from sentence_transformers import SentenceTransformer, CrossEncoder

from src.retrieval.reranker import create_reranker

from scripts.search_hybrid import search_hybrid, MODEL_NAME
from src.retrieval.bm25 import create_bm25_index


DOCUMENTS_FILE = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")
QUESTIONS_FILE = (utils.DATA_EVAL_PATH / "questions.json")
LANGUAGE = "en"


def find_first_relevant_rank(retrieved_documents: list[dict], relevant_articles: set[str]):
    for rank, document in enumerate(retrieved_documents, start=1):
        article = str(document["article_number"])

        if article in relevant_articles:
            return rank

    return None


def evaluate(
    questions: list[dict],
    bm25_retriever,
    model: SentenceTransformer,
    top_k: int,
    candidate_k: int,
    language: str,
    reranker_model: CrossEncoder | None,
) -> dict:
    answerable_questions = [question for question in questions if question["relevant_articles"]]

    if not answerable_questions:
        raise ValueError("The evaluation set has no answerable questions.")

    recall_at_1 = []
    recall_at_k = []
    reciprocal_ranks = []
    latencies_ms = []
    failures = []
    query_results = []

    for question in answerable_questions:
        started_at = time.perf_counter()

        results = search_hybrid(
            query=question["question"],
            bm25_retriever=bm25_retriever,
            model=model,
            top_k=top_k,
            candidate_k=candidate_k,
            language=language,
            reranker_model=reranker_model,
        )

        latency_ms = (time.perf_counter() - started_at) * 1000
        latencies_ms.append(latency_ms)

        relevant_articles = {
            str(article)
            for article in question["relevant_articles"]
        }

        first_rank = find_first_relevant_rank(results, relevant_articles)

        recall_at_1.append(first_rank == 1)
        recall_at_k.append(first_rank is not None)
        reciprocal_ranks.append((1 / first_rank) if first_rank else 0)

        result = {
            "id": question["id"],
            "question": question["question"],
            "expected_articles": sorted(relevant_articles),
            "retrieved_articles": [str(document["article_number"]) for document in results],
            "retrieved_chunks": [document["chunk_id"] for document in results],
            "first_relevant_rank": first_rank,
            "latency_ms": round(latency_ms, 3),
        }

        query_results.append(result)

        if first_rank is None:
            failures.append(result)

    question_count = len(answerable_questions)

    return {
        "method": ("Hybrid RRF + reranker" if reranker_model else "Hybrid RRF"),
        "questions_evaluated": question_count,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "recall_at_1": round(sum(recall_at_1) / question_count, 3),
        f"recall_at_{top_k}": round(sum(recall_at_k) / question_count, 3),
        "mrr": round(sum(reciprocal_ranks) / question_count, 3),
        "average_latency_ms": round(sum(latencies_ms) / question_count, 3),
        "failures": failures,
        "query_results": query_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_file", type=Path, default=DOCUMENTS_FILE)
    parser.add_argument("--questions_file", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    documents = utils.load_json_file(args.documents_file)
    questions = utils.load_json_file(args.questions_file)

    bm25_retriever = create_bm25_index(documents, args.language)
    model = SentenceTransformer(MODEL_NAME)

    reranker_model = (create_reranker() if args.rerank else None)

    results = evaluate(
        questions=questions,
        bm25_retriever=bm25_retriever,
        model=model,
        top_k=args.top_k,
        candidate_k=min(args.candidate_k, len(documents)),
        language=args.language,
        reranker_model=reranker_model,
    )

    pprint(results)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(
            json.dumps(results, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()