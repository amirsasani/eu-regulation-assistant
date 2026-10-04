# EU Regulation Intelligence Assistant

A retrieval system for searching European Union regulations and locating the exact articles that support an answer.

The project currently focuses on the **EU Artificial Intelligence Act** and uses official documents published by [EUR-Lex](https://eur-lex.europa.eu/).

> This project is an information-retrieval demonstration and does not provide legal advice.

## Current status

- [x] Download and parse the EU AI Act
- [x] Preserve article and paragraph metadata
- [x] Implement BM25 keyword retrieval
- [x] Create a labelled retrieval evaluation set
- [ ] Add semantic vector search
- [ ] Implement hybrid retrieval
- [ ] Add reranking
- [ ] Generate answers with source citations
- [ ] Expose the system through an API
- [ ] Add Italian-language support

Only completed features should be checked.

## Why this project?

Legal documents are long, structured and difficult to search using ordinary keyword matching. This project explores how retrieval and language models can help users locate relevant regulatory passages while keeping every answer traceable to an official source.

The project is designed around three requirements:

1. Retrieve the correct regulatory passages.
2. Cite the exact supporting articles.
3. Refuse to answer when the available documents do not contain enough evidence.

## Data source

The initial dataset is the official English text of:

- Regulation (EU) 2024/1689 — Artificial Intelligence Act
- CELEX identifier: `32024R1689`
- Source: [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689)

Each extracted record contains:

```json
{
  "document_id": "32024R1689",
  "language": "en",
  "article_number": "9",
  "article_title": "Risk management system",
  "paragraph_number": "1",
  "text": "...",
  "source_url": "...",
  "chunk_id": "ai-act-en-article-9-paragraph-1"
}
```

## Project structure

```text
eu-regulation-assistant/
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── scripts/
│   └── evaluate_retrieval.py
├── src/
│   ├── ingestion/
│   └── retrieval/
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone <repository-url>
cd eu-regulation-assistant

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Document ingestion

Download and process the AI Act:

```bash
python -m src.ingestion.fetch_document
python -m src.ingestion.parse_document
```

The structured output is saved to:

```text
data/processed/ai_act_articles.jsonl
```

## Search

Run a search from the command line:

```bash
python -m src.retrieval.search \
  "What risk management obligations apply to high-risk AI systems?"
```

Example output:

```text
1. Article 9 — Risk management system
2. Article 17 — Quality management system
3. Article 15 — Accuracy, robustness and cybersecurity
```

## Evaluation

Retrieval is evaluated using manually labelled questions mapped to relevant AI Act articles.

Run the evaluation:

```bash
python scripts/evaluate_retrieval.py
```

## Limitations

- Only the English EU AI Act is currently indexed.
- Keyword retrieval may fail when the question and regulation use different terminology.
- Retrieval results are not legal interpretations.
- The system does not yet generate answers.
- Document amendments and consolidated versions are not yet synchronized automatically.

## Roadmap

1. Establish a BM25 retrieval baseline.
2. Add semantic retrieval using sentence-transformer embeddings.
3. Combine keyword and semantic results.
4. Add a cross-encoder reranker.
5. Generate grounded answers with article-level citations.
6. Evaluate citation correctness and unsupported-question handling.
7. Add a FastAPI service and simple user interface.
8. Add Italian-language queries, GDPR and NIS2.

## License

The source code in this repository is licensed under the Apache License 2.0.

EU regulatory texts and metadata are not covered by the software license.
They remain subject to the applicable EU and EUR-Lex reuse conditions.
Sources are attributed in the corresponding data records.

Third-party libraries and models remain subject to their respective licenses.