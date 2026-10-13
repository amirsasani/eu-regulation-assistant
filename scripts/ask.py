import argparse
import utils
from src.services.rag_service import ask as rag_ask


DOCUMENTS_FILE = (utils.DATA_PROCESSED_PATH / "ai-act-articles.json")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=10)
    parser.add_argument("--language", default="en")
    parser.add_argument("--llm_model", default=utils.DEFAULT_OLLAMA_MODEL)
    args = parser.parse_args()

    generation, sources = rag_ask(
        question=args.question,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        language=args.language,
        llm_model=args.llm_model
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