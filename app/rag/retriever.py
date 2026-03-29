"""
VisionAgent — RAG Retriever Modülü.
Kullanıcının sorusuna/metnine benzer geçmiş kayıtları Qdrant'tan getirir.
"""

from app.rag.indexer import get_client, embed_text, ensure_collection, COLLECTION_NAME


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Verilen sorguya semantik olarak en yakın kayıtları döndürür.
    
    Args:
        query: Aranacak metin (VLM çıktısı veya kullanıcı sorusu)
        top_k: Döndürülecek maksimum sonuç sayısı
    
    Returns:
        Eşleşen dokümanların listesi (text, score, metadata)
    """
    try:
        ensure_collection()
        client = get_client()
    except ConnectionError:
        # Qdrant kapalıysa boş döndür, uygulamayı çökertme
        print("[RAG] Qdrant bağlantısı yok, boş sonuç dönüyor.")
        return []

    vector = embed_text(query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "score": round(hit.score, 3),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
        }
        for hit in results
    ]