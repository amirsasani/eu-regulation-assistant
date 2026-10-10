import argparse

from sentence_transformers import SentenceTransformer

import utils

from scripts.search_hybrid import search_hybrid, MODEL_NAME
from src.generation.client import generate_answer
from src.generation.prompt import build_messages
from src.retrieval.bm25 import create_bm25_index


DOCUMENTS_FILE = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=20)
    parser.add_argument("--language", default="en")
    parser.add_argument("--llm_model", default=utils.DEFAULT_OPENROUTER_MODEL)
    args = parser.parse_args()

    documents = utils.load_json_file(DOCUMENTS_FILE)

    bm25_retriever = create_bm25_index(documents, args.language)
    embedding_model = SentenceTransformer(MODEL_NAME)

    sources = search_hybrid(
        query=args.question,
        bm25_retriever=bm25_retriever,
        model=embedding_model,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        language=args.language,
    )

    messages = build_messages(question=args.question, documents=sources)

    generation = generate_answer(messages=messages, model=args.llm_model)

    print("\nANSWER\n")
    print(generation["answer"])

    print("\nSOURCES")
    for source in sources:
        print(
            f"- Article {source['article_number']}, "
            f"paragraph {source.get('paragraph_number')} "
            f"({source['chunk_id']})"
        )


if __name__ == "__main__":
    main()