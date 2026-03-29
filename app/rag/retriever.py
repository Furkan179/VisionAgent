"""
VisionAgent — RAG Retriever Modülü.
Kullanıcının sorusuna/metnine benzer geçmiş kayıtları Qdrant'tan getirir.
"""

from typing import List, Dict
from app.rag.indexer import get_client, embed_text, ensure_collection, COLLECTION_NAME


def retrieve(query: str, top_k: int = 3) -> List[Dict]:
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

    # qdrant-client v1.12+ → query_points kullanır (search deprecated)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "score": round(hit.score, 3),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
        }
        for hit in results.points
    ]