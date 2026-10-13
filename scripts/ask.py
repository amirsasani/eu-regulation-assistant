import argparse

from src.services.rag_service import RagService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=10)
    parser.add_argument("--language", default="en")
    parser.add_argument("--llm_model", default=None)
    args = parser.parse_args()

    service = RagService(language=args.language)

    generation, sources = service.ask(
        question=args.question,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        llm_model=args.llm_model,
    )

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