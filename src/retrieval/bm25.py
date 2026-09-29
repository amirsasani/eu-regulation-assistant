import argparse
import json
from pathlib import Path
from bm25s import BM25, tokenize

INPUT_JSON = Path("data/processed/ai-act-articles.json")
LANGUAGE = "en"

def create_corpus(json_file: Path):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)


    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of documents.")

    if not data:
        raise ValueError("Input JSON contains no documents.")

    return data

def create_bm25_index(documents, language: str):
    texts = [document["text"] for document in documents]
    
    bm25 = BM25(corpus=documents)

    tokens = tokenize(texts, stopwords=language)
    bm25.index(tokens)
    
    return bm25

def search(query: str, retriever: BM25, top_k: int, language: str):
    tokens = tokenize([query], stopwords=language)
    results, scores = retriever.retrieve(tokens, k=top_k)

    return results, scores

def show_results(results, scores):
    for rank, (document, score) in enumerate(zip(results[0], scores[0]), start=1):
        article = document.get("article_number", "Unknown")
        title = document.get("article_title", "Untitled")
        text = document["text"]

        print(f"\n{rank}. Article {article} — {title}")
        print(f"   Score: {float(score):.4f}")
        print(f"   {text[:400]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=Path, default=INPUT_JSON)
    parser.add_argument("--language", type=str, default=LANGUAGE)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    corpus = create_corpus(args.input_json)
    retriever = create_bm25_index(corpus, args.language)

    if args.query:
        results, scores = search(args.query, retriever, args.top_k, args.language)
        show_results(results, scores)