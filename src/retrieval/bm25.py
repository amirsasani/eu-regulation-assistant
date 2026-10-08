import argparse
import json
from pathlib import Path
from bm25s import BM25, tokenize
from pprint import pprint
import utils

INPUT_JSON = utils.DATA_PROCESSED_PATH / "ai-act-articles.json"
LANGUAGE = "en"

def create_corpus(json_file: Path):
    required_fields = {
        "chunk_id",
        "article_number",
        "article_title",
        "text",
    }
    
    data = utils.load_json_file(json_file)


    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of documents.")

    if not data:
        raise ValueError("Input JSON contains no documents.")

    for position, document in enumerate(data):
        missing_fields = required_fields - document.keys()

        if missing_fields:
            raise ValueError(
                f"Document {position} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if not document["text"].strip():
            raise ValueError(f"Document {position} contains empty text.")

    return data

def create_bm25_index(documents, language: str):
    texts = [
        (
            f"Article {document['article_number']}. "
            f"{document['article_title']}. "
            f"{document['text']}"
        )
        for document in documents
    ]
    
    bm25 = BM25(corpus=documents)

    tokens = tokenize(texts, stopwords=language)
    bm25.index(tokens)
    
    return bm25

def search(query: str, retriever: BM25, top_k: int, language: str):
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")


    tokens = tokenize([query], stopwords=language)
    results, scores = retriever.retrieve(tokens, k=top_k)

    return results, scores

def show_results(results, scores, text_length=250):
    output = []
    for rank, (document, score) in enumerate(zip(results[0], scores[0]), start=1):
        chunk_id = document.get("chunk_id", "Unknown")
        article_number = document.get("article_number", "Unknown")
        article_title = document.get("article_title", "Untitled")
        text = document["text"][:text_length] + "..." if len(document["text"]) > text_length else document["text"]

        output.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": chunk_id,
            "article": article_number,
            "title": article_title,
            "text": text
        })

    pprint(output)

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=Path, default=INPUT_JSON)
    parser.add_argument("--language", type=str, default=LANGUAGE)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--query", type=str, default="", required=True)
    args = parser.parse_args()

    corpus = create_corpus(args.input_json)
    retriever = create_bm25_index(corpus, args.language)

    results, scores = search(args.query, retriever, args.top_k, args.language)
    show_results(results, scores)