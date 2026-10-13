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
- Exposes the complete RAG pipeline through FastAPI
- Reports application and dependency readiness
- Supports containerized API deployment

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
    L --> M[FastAPI response]
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
- [x] Add API endpoint tests
- [x] Containerize the FastAPI application

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
│   ├── api/
│   ├── database/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── retrieval/
│   └── services/
├── tests/
│   └── test_api.py
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── README.md
├── requirements-dev.txt
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
python -m pip install -r requirements.txt
```

Install the development dependencies when running tests:

```bash
python -m pip install -r requirements-dev.txt
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
OLLAMA_DOCKER_BASE_URL=http://host.docker.internal:11434
```

Do not commit the `.env` file.

## PostgreSQL setup

Start PostgreSQL with pgvector:

```bash
docker compose up -d postgres
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

## API

The FastAPI application loads the BM25 index, embedding model and reranker once during startup so they can be reused across requests.

Start the API from the activated virtual environment:

```bash
python -m uvicorn src.api.app:app --reload
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Confirms that the FastAPI process is running |
| `GET` | `/ready` | Checks PostgreSQL, indexed chunk count and the configured Ollama model |
| `POST` | `/ask` | Runs retrieval, reranking and grounded answer generation |

### Health check

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok"
}
```

### Readiness check

```bash
curl http://127.0.0.1:8000/ready
```

```json
{
  "status": "ok",
  "services": {
    "postgres": {
      "status": "ok",
      "chunks": 519
    },
    "ollama": {
      "status": "ok",
      "model": "qwen3.5:4b"
    }
  }
}
```

The readiness endpoint returns HTTP `503` when PostgreSQL, the indexed chunks or the configured Ollama model are unavailable.

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What risk management obligations apply to high-risk AI systems?",
    "top_k": 5,
    "candidate_k": 10
  }'
```

Example response:

```json
{
  "answer": "A risk management system must be established, implemented, documented and maintained [Article 9, paragraph 1].",
  "model": "qwen3.5:4b",
  "sources": [
    {
      "chunk_id": "ai-act-en-article-9-paragraph-1",
      "article_number": "9",
      "article_title": "Risk management system",
      "paragraph_number": "1",
      "text": "A risk management system shall be established...",
      "source_url": "https://eur-lex.europa.eu/",
      "reranker_score": 0.99
    }
  ]
}
```

## API tests

The API tests use mocked RAG and dependency services, so they do not load local ML models or require PostgreSQL and Ollama to be running.

Run the tests with:

```bash
python -m pytest
```

The tests cover:

- Application health
- PostgreSQL and Ollama readiness
- Dependency failure responses
- Successful question answering
- Request validation

## Containerized API

The API and PostgreSQL can run through Docker Compose. Ollama remains on the host so it can use the locally installed model without duplicating it inside a container.

Make Ollama reachable from Docker:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Build the application image:

```bash
docker compose build api
```

Initialize PostgreSQL, generate the semantic index and load the chunks:

```bash
docker compose run --rm indexer
```

Start the API and PostgreSQL:

```bash
docker compose up -d api
```

Check container status and application readiness:

```bash
docker compose ps
curl http://127.0.0.1:8000/ready
```

Follow the API logs:

```bash
docker compose logs -f api
```

Stop the containers:

```bash
docker compose down
```

The image installs the CPU-only PyTorch build. The first build and startup can take several minutes because PyTorch and the sentence-transformer models must be downloaded. Hugging Face model files are retained in the `model_cache` Docker volume.

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
- The project exposes an HTTP API but does not yet include a graphical user interface.
- Containerized API mode still requires Ollama to run on the host.

## Roadmap

1. Expand the retrieval and generation evaluation sets.
2. Add Italian-language queries and answers.
3. Add GDPR and NIS2 as additional data sources.
4. Detect and process updated consolidated regulations automatically.
5. Add a simple web interface.

## License

The source code in this repository is licensed under the Apache License 2.0.

EU regulatory texts and metadata are not covered by the software license. They remain subject to the applicable EU and EUR-Lex reuse conditions. Sources are attributed in the corresponding data records.

Third-party libraries and models remain subject to their respective licenses.