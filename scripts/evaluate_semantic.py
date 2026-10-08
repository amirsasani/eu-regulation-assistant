import argparse
import time
import json
import numpy as np
import utils
from pathlib import Path

from src.retrieval.semantic import load_model
from src.retrieval.semantic import search as semantic_search

DEFAULT_DOCUMENTS = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")

DEFAULT_QUESTIONS = (utils.DATA_EVAL_PATH / "questions.json")

DEFAULT_INDEX_DIRECTORY = (utils.DATA_INDEXES_PATH / "semantic")

DEFAULT_OUTPUT = (utils.DATA_EVAL_PATH / "semantic-results.json")

def load_semantic_index(index_directory: Path) -> tuple[np.ndarray, dict]:
    embeddings_path = (index_directory / "embeddings.npy")
    manifest_path = (index_directory / "manifest.json")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}. Run build_semantic_index first.")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}.")

    embeddings = np.load(embeddings_path, allow_pickle=False)
    manifest = utils.load_json_file(manifest_path)

    return embeddings, manifest

def validate_index(documents: list[dict], embeddings: np.ndarray, manifest: dict):
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if len(documents) != len(embeddings):
        raise ValueError("Document and embedding counts do not match. Rebuild the semantic index.")

    if manifest["document_count"] != len(documents):
        raise ValueError("Manifest document count does not match.")

    if (manifest["embedding_dimension"] != embeddings.shape[1]):
        raise ValueError("Manifest embedding dimension does not match.")

    if not manifest.get("normalized"):
        raise ValueError("Evaluation expects normalized embeddings.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")


def find_first_relevant_rank(results: list[dict], relevant_articles: set[str]):
    for result in results:
        if result["article_number"] in relevant_articles:
            return result["rank"]

    return None

def evaluate(documents: list[dict], questions: list[dict], embeddings: np.ndarray, model, top_k: int):
    answerable_questions = [question for question in questions if question["relevant_articles"]]

    if not answerable_questions:
        raise ValueError("No answerable evaluation questions found.")

    semantic_search(
        query=answerable_questions[0]["question"],
        model=model,
        documents=documents,
        document_embeddings=embeddings,
        top_k=1,
    )

    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_k = []
    latencies_ms = []
    failures = []
    query_results = []

    effective_top_k = min(top_k, len(documents))

    for question in answerable_questions:
        started_at = time.perf_counter()

        results = semantic_search(
            query=question["question"],
            model=model,
            documents=documents,
            document_embeddings=embeddings,
            top_k=effective_top_k,
        )

        latency_ms = (time.perf_counter() - started_at) * 1000

        relevant_articles = {str(article) for article in question["relevant_articles"]}

        first_rank = find_first_relevant_rank(results, relevant_articles)

        recall_at_1.append(first_rank == 1)
        recall_at_k.append(first_rank is not None)

        reciprocal_ranks.append(1 / first_rank if first_rank else 0)

        latencies_ms.append(latency_ms)

        result = {
            "id": question["id"],
            "question": question["question"],
            "expected_articles": sorted(relevant_articles),
            "retrieved_articles": [item["article_number"] for item in results],
            "retrieved_chunks": [item["chunk_id"] for item in results],
            "scores": [round(item["score"], 6) for item in results],
            "first_relevant_rank": first_rank,
            "latency_ms": round(latency_ms, 3),
        }

        query_results.append(result)

        if first_rank is None:
            failures.append(result)

    question_count = len(answerable_questions)

    return {
        "method": "semantic",
        "questions_evaluated": question_count,
        "top_k": effective_top_k,
        "recall_at_1": (sum(recall_at_1) / question_count),
        f"recall_at_{effective_top_k}": (sum(recall_at_k) / question_count),
        "mrr": (sum(reciprocal_ranks) / question_count),
        "average_latency_ms": (sum(latencies_ms) / question_count),
        "failures": failures,
        "query_results": query_results,
    }

def print_report(report: dict) -> None:
    top_k = report["top_k"]

    print("\nSemantic retrieval evaluation")
    print("-" * 35)
    print(f"Questions: {report['questions_evaluated']}")
    print(f"Recall@1: {report['recall_at_1']:.3f}")
    print(f"Recall@{top_k}: {report[f'recall_at_{top_k}']:.3f}")
    print(f"MRR: {report['mrr']:.3f}")
    print(f"Average latency: {report['average_latency_ms']:.3f} ms")

    print("\nPer-question results")

    for result in report["query_results"]:
        print(
            f"{result['id']}: "
            f"expected={result['expected_articles']} "
            f"retrieved={result['retrieved_articles']} "
            f"rank={result['first_relevant_rank']}"
        )

def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic retrieval.")

    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index-directory", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    documents = utils.load_json_file(args.documents)
    questions = utils.load_json_file(args.questions)

    embeddings, manifest = load_semantic_index(args.index_directory)

    validate_index(documents, embeddings, manifest)

    print(f"Loading model: {manifest['model']}")
    model = load_model(manifest["model"])

    report = evaluate(
        documents=documents,
        questions=questions,
        embeddings=embeddings,
        model=model,
        top_k=args.top_k,
    )

    report["model"] = manifest["model"]
    report["embedding_dimension"] = (manifest["embedding_dimension"])

    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()