from app.rag.indexer import get_client, embed_text, COLLECTION_NAME, ensure_collection

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    ensure_collection()
    c = get_client()
    vector = embed_text(query)

    results = c.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "score": round(hit.score, 3),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
        }
        for hit in results
    ]