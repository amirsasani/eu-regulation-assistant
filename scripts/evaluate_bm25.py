import json
import argparse
from pprint import pprint
from pathlib import Path
from src.retrieval.bm25 import create_bm25_index, search
import utils

JSON_FILE = utils.DATA_PROCESSED_PATH / "ai-act-articles.json"
QUESTIONS_FILE = utils.DATA_EVAL_PATH / "questions.json"
LANGUAGE = "en"

def find_first_relevant_rank(retrieved_documents, relevant_articles: set[str]):
    for rank, document in enumerate(retrieved_documents, start=1):
        article = str(document["article_number"])

        if article in relevant_articles:
            return rank

    return None

def evaluate(questions, documents, top_k: int, language: str):
    retriever = create_bm25_index(documents, language)

    answerable_questions = [
        question
        for question in questions
        if question["relevant_articles"]
    ]

    if not answerable_questions:
        raise ValueError(
            "The evaluation set has no answerable questions."
        )

    reciprocal_ranks = []
    recall_at_1 = []
    recall_at_k = []
    failures = []
    query_results = []

    for question in answerable_questions:
        results, _ = search(
            query=question["question"],
            retriever=retriever,
            top_k=min(top_k, len(documents)),
            language=language,
        )

        retrieved_documents = list(results[0])
        relevant_articles = {
            str(article)
            for article in question["relevant_articles"]
        }

        first_rank = find_first_relevant_rank(
            retrieved_documents,
            relevant_articles,
        )

        recall_at_1.append(first_rank == 1)
        recall_at_k.append(first_rank is not None)
        reciprocal_ranks.append(
            1 / first_rank if first_rank else 0
        )

        retrieved_articles = [
            str(document["article_number"])
            for document in retrieved_documents
        ]

        result = {
            "id": question["id"],
            "question": question["question"],
            "expected_articles": sorted(relevant_articles),
            "retrieved_articles": retrieved_articles,
            "first_relevant_rank": first_rank,
        }

        query_results.append(result)

        if first_rank is None:
            failures.append(result)

    question_count = len(answerable_questions)

    return {
        "method": "BM25",
        "questions_evaluated": question_count,
        "top_k": top_k,
        "recall_at_1": sum(recall_at_1) / question_count,
        f"recall_at_{top_k}": (
            sum(recall_at_k) / question_count
        ),
        "mrr": sum(reciprocal_ranks) / question_count,
        "failures": failures,
        "query_results": query_results,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=Path, default=JSON_FILE)
    parser.add_argument("--questions_file", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--language", type=str, default=LANGUAGE)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    documents = utils.load_json_file(args.json_file)

    top_k = min(args.top_k, len(documents))

    questions = utils.load_json_file(args.questions_file)
    results = evaluate(questions, documents, top_k=top_k, language=args.language)

    pprint(results)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(
            json.dumps(results, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )
