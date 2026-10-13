# EU Regulation Intelligence Assistant

A retrieval-augmented generation (RAG) system for answering questions about European Union regulations with article and paragraph citations.

The project currently focuses on the **EU Artificial Intelligence Act** and uses the official text published by [EUR-Lex](https://eur-lex.europa.eu/).

> This project is an information-retrieval demonstration and does not provide legal advice.

## Features

- Downloads and parses the official EU AI Act
- Preserves article and paragraph metadata
- Supports BM25 keyword retrieval
- Stores semantic embeddings in PostgreSQL with pgvector
- Combines keyword and semantic search using Reciprocal Rank Fusion
- Reranks candidates with a cross-encoder
- Generates grounded answers using local Ollama models
- Adds article and paragraph citations
- Refuses questions that cannot be answered from the retrieved sources
- Evaluates retrieval, grounding and citation correctness

## Architecture

```mermaid
flowchart LR
    A[EUR-Lex] --> B[Document parser]
    B --> C[Article and paragraph chunks]

    C --> D[BM25 index]
    C --> E[Multilingual E5 embeddings]
    E --> F[(PostgreSQL and pgvector)]

    Q[User question] --> D
    Q --> G[Query embedding]
    G --> F

    D --> H[Reciprocal Rank Fusion]
    F --> H
    H --> I[BGE reranker]
    I --> J[Top source chunks]
    J --> K[Ollama generator]
    K --> L[Answer with citations]
```

## Current status

### Data ingestion

- [x] Download and parse the EU AI Act
- [x] Preserve article and paragraph metadata

### Retrieval

- [x] Implement BM25 keyword retrieval
- [x] Create a labelled retrieval evaluation set
- [x] Add semantic vector search
- [x] Store embeddings in PostgreSQL with pgvector
- [x] Implement hybrid retrieval with Reciprocal Rank Fusion
- [x] Evaluate BM25, semantic and hybrid retrieval
- [x] Add reranking and evaluate its impact

### RAG application

- [x] Generate grounded answers with article and paragraph citations
- [x] Evaluate answer grounding and citation correctness
- [x] Expose the system through an API

### Expansion

- [ ] Add Italian-language support
- [ ] Add GDPR and NIS2 documents

## Why this project?

Legal documents are long, structured and difficult to search using ordinary keyword matching. This project explores how retrieval and language models can locate relevant regulatory passages while keeping every answer traceable to an official source.

The system is designed around three requirements:

1. Retrieve the correct regulatory passages.
2. Cite the exact supporting articles and paragraphs.
3. Refuse to answer when the supplied sources do not contain enough evidence.

## Retrieval results

The retrieval evaluation contains nine answerable questions labelled with their relevant EU AI Act articles.

All retrieval configurations used the same document chunks and returned five results per question. The reranker processed the top 10 hybrid candidates.

| Method | Recall@1 | Recall@5 | MRR | Average latency |
|---|---:|---:|---:|---:|
| BM25 | 0.556 | 0.778 | 0.667 | 0.313 ms |
| Multilingual E5 | 0.889 | 1.000 | 0.944 | 15.114 ms |
| Hybrid RRF | 0.778 | 1.000 | 0.889 | 88.215 ms |
| Hybrid RRF + BGE reranker | 1.000 | 1.000 | 1.000 | 1,488.462 ms |

Semantic search uses `intfloat/multilingual-e5-base`. Its latency includes query embedding and vector comparison but excludes model loading.

Reranking uses `BAAI/bge-reranker-v2-m3` over the top 10 hybrid candidates. Model-loading time is excluded, while the first query includes inference warm-up.

These results are preliminary because the evaluation set is small.

## Generation evaluation

Answer generation was evaluated on four questions using local Ollama models:

- Generator: `qwen3.5:4b`
- Judge: `gemma3:4b`
- Retrieval candidates: 10
- Context chunks: 5

| Metric | Result |
|---|---:|
| Groundedness | 2.000 / 2 |
| Citation support | 2.000 / 2 |
| Point coverage | 2.000 / 2 |
| Citation precision | 1.000 |
| Valid citation rate | 1.000 |
| Refusal accuracy | 1.000 |
| Expected-source recall | 0.889 |
| Average retrieval latency | 2,843.769 ms |
| Average generation latency | 11,255.577 ms |
| Average total latency | 14,099.346 ms |

Groundedness, citation support and point coverage are scored from 0 to 2 by the local `gemma3:4b` judge. Total latency includes retrieval and answer generation but excludes judge evaluation.

These results are preliminary because the generation evaluation contains only four questions.

## Data source

The initial dataset is the official English text of:

- Regulation (EU) 2024/1689 — Artificial Intelligence Act
- CELEX identifier: `32024R1689`
- Source: [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689)

The parser currently produces 519 article and paragraph chunks.

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
│   ├── evaluation/
│   ├── indexes/
│   │   └── semantic/
│   ├── processed/
│   └── raw/
├── scripts/
│   ├── ask.py
│   ├── build_semantic_index.py
│   ├── check_database.py
│   ├── evaluate_bm25.py
│   ├── evaluate_generation.py
│   ├── evaluate_hybrid.py
│   ├── evaluate_semantic.py
│   ├── initialize_database.py
│   ├── load_semantic_index.py
│   ├── search_hybrid.py
│   ├── search_postgres.py
│   └── search_semantic.py
├── src/
│   ├── database/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   └── retrieval/
├── .env.example
├── docker-compose.yml
├── LICENSE
├── README.md
├── requirements.txt
└── utils.py
```

## Requirements

- Python 3.12 or later
- Docker with Docker Compose
- [Ollama](https://ollama.com/)

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Create the environment configuration:

```bash
cp .env.example .env
```

Configure `.env`:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/eu_regulations
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_JUDGE_MODEL=gemma3:4b
```

Do not commit the `.env` file.

## PostgreSQL setup

Start PostgreSQL with pgvector:

```bash
docker compose up -d
```

Initialize the database extension and table:

```bash
python -m scripts.initialize_database
```

Verify the connection:

```bash
python -m scripts.check_database
```

## Ollama setup

Download the generation and evaluation models:

```bash
ollama pull qwen3.5:4b
ollama pull gemma3:4b
```

Ensure Ollama is running:

```bash
ollama serve
```

## Document ingestion

Download and parse the EU AI Act:

```bash
python -m src.ingestion.fetch_document
python -m src.ingestion.parse_document
```

The structured output is saved to:

```text
data/processed/ai-act-articles.json
```

## Semantic index

Generate the document embeddings:

```bash
python -m scripts.build_semantic_index
```

Load the document chunks and embeddings into PostgreSQL:

```bash
python -m scripts.load_semantic_index
```

## Search

### BM25 search

```bash
python -m src.retrieval.bm25 \
  --query "What risk management obligations apply to high-risk AI systems?"
```

### Semantic search

```bash
python -m scripts.search_postgres \
  --query "What risk management obligations apply to high-risk AI systems?" \
  --top-k 5
```

### Hybrid search with reranking

```bash
python -m scripts.search_hybrid \
  --query "What risk management obligations apply to high-risk AI systems?" \
   --rerank \
  --top_k 5 \
  --candidate_k 10
```

## Generate an answer

Run the complete RAG pipeline:

```bash
python -m scripts.ask \
  --question "What risk management obligations apply to high-risk AI systems?" \
  --top_k 5 \
  --candidate_k 10
```

The command retrieves relevant passages, reranks them and generates an answer containing article and paragraph citations.

## Evaluation

The retrieval evaluation uses manually labelled questions mapped to their relevant AI Act articles.

Run all retrieval evaluations:

```bash
python -m scripts.evaluate_bm25
python -m scripts.evaluate_semantic
python -m scripts.evaluate_hybrid --rerank --candidate_k 10
```

Run the generation evaluation:

```bash
python -m scripts.evaluate_generation \
  --top_k 5 \
  --candidate_k 10
```

The detailed generation results are saved to:

```text
data/evaluation/generation-results.json
```

## Limitations

- Only the English EU AI Act is currently indexed.
- The generation evaluation contains only four questions.
- The answer-quality judge is a local language model rather than a legal expert.
- Local-model latency depends on the available hardware.
- Document amendments and consolidated versions are not synchronized automatically.
- The project does not yet expose an HTTP API or user interface.

## Roadmap

1. Expose retrieval and generation through FastAPI.
2. Add API tests and containerized deployment.
3. Expand the retrieval and generation evaluation sets.
4. Add Italian-language queries and answers.
5. Add GDPR and NIS2 as additional data sources.
6. Detect and process updated consolidated regulations automatically.

## License

The source code in this repository is licensed under the Apache License 2.0.

EU regulatory texts and metadata are not covered by the software license. They remain subject to the applicable EU and EUR-Lex reuse conditions. Sources are attributed in the corresponding data records.

Third-party libraries and models remain subject to their respective licenses.