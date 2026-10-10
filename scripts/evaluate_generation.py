import argparse
import json
import time
from pathlib import Path
from pprint import pprint

from sentence_transformers import SentenceTransformer

import utils
from scripts.search_hybrid import MODEL_NAME, search_hybrid
from src.evaluation.generation import evaluate_answer
from src.generation.client import generate_answer
from src.generation.prompt import build_messages
from src.retrieval.bm25 import create_bm25_index
from src.retrieval.reranker import create_reranker


DOCUMENTS_FILE = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")
QUESTIONS_FILE = (utils.DATA_EVAL_PATH / "generation_questions.json")
OUTPUT_FILE = (utils.DATA_EVAL_PATH / "generation-results.json")


def average(results: list[dict], key: str) -> float:
    if not results:
        return 0.0

    return round(sum(result[key] for result in results)/ len(results), 3)


def evaluate(
    questions: list[dict],
    bm25_retriever,
    embedding_model,
    reranker_model,
    language: str,
    top_k: int,
    candidate_k: int,
    llm_model: str | None,
) -> dict:
    query_results = []

    for item in questions:
        retrieval_started = time.perf_counter()

        sources = search_hybrid(
            query=item["question"],
            bm25_retriever=bm25_retriever,
            model=embedding_model,
            top_k=top_k,
            candidate_k=candidate_k,
            language=language,
            reranker_model=reranker_model,
        )

        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        messages = build_messages(question=item["question"], documents=sources)

        generation_started = time.perf_counter()

        generation = generate_answer(messages=messages, model=llm_model)

        generation_ms = (time.perf_counter() - generation_started) * 1000

        metrics = evaluate_answer(
            answer=generation["answer"],
            sources=sources,
            evaluation_item=item,
        )

        query_results.append({
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "answer": generation["answer"],
            "model": generation["model"],
            "sources": [
                {
                    "chunk_id": source["chunk_id"],
                    "article_number": (source["article_number"]),
                    "paragraph_number": (source.get("paragraph_number")),
                    "reranker_score": (source.get("reranker_score")),
                }
                for source in sources
            ],
            "retrieval_ms": round(retrieval_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "total_latency_ms": round(retrieval_ms + generation_ms, 3),
            **metrics,
        })

    answerable_results = [
        result
        for result in query_results
        if result["answerable"]
    ]

    return {
        "method": "Hybrid RRF + reranker + LLM",
        "questions_evaluated": len(query_results),
        "citation_precision": average(answerable_results, "citation_precision"),
        "valid_citation_rate": round(
            sum(
                result["has_valid_citation"]
                for result in answerable_results
            ) / len(answerable_results),
            3,
        ) if answerable_results else 0.0,
        "expected_source_recall": average(answerable_results, "expected_source_recall"),
        "refusal_accuracy": round(
            sum(
                result["refusal_correct"]
                for result in query_results
            ) / len(query_results),
            3,
        ),
        "average_retrieval_ms": average(query_results, "retrieval_ms"),
        "average_generation_ms": average(query_results, "generation_ms"),
        "average_total_latency_ms": average(query_results, "total_latency_ms"),
        "query_results": query_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_file", type=Path, default=DOCUMENTS_FILE)
    parser.add_argument("--questions_file", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--output_file", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--language", default="en")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=10)
    parser.add_argument("--llm_model", default=None)
    args = parser.parse_args()

    documents = utils.load_json_file(args.documents_file)
    questions = utils.load_json_file(args.questions_file)

    bm25_retriever = create_bm25_index(documents, args.language)
    embedding_model = SentenceTransformer(MODEL_NAME)
    reranker_model = create_reranker()

    results = evaluate(
        questions=questions,
        bm25_retriever=bm25_retriever,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        language=args.language,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        llm_model=args.llm_model,
    )

    pprint({key: value for key, value in results.items() if key != "query_results"})

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()