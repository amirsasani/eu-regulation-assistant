from collections import defaultdict


def reciprocal_rank_fusion(
    bm25_documents: list[dict],
    semantic_documents: list[dict],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    scores = defaultdict(float)
    documents_by_id = {}
    sources = defaultdict(list)

    rankings = {"bm25": bm25_documents, "semantic": semantic_documents}

    for source, documents in rankings.items():
        for rank, document in enumerate(documents, start=1):
            chunk_id = document["chunk_id"]

            scores[chunk_id] += (1 / (rrf_k + rank))
            documents_by_id[chunk_id] = document
            sources[chunk_id].append(source)

    ranked_chunk_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

    return [
        {
            **documents_by_id[chunk_id],
            "rrf_score": scores[chunk_id],
            "retrieved_by": sources[chunk_id],
        }
        for chunk_id in ranked_chunk_ids
    ]