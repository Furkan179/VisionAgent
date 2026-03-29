"""
VisionAgent — RAG Indexer Modülü.
VLM çıktılarını vektöre dönüştürüp Qdrant'a kaydeder.
Böylece geçmiş analizler "uzun süreli hafıza" olarak kullanılabilir.
"""

from typing import Optional, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import os
import uuid

# Qdrant bağlantı ayarları (.env'den okunur)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "vision_agent_memory"

# Embedding modeli — hafif ve hızlı (384 boyutlu vektör üretir)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Singleton nesneler
_client: Optional[QdrantClient] = None
_embed_model: Optional[SentenceTransformer] = None


def get_client() -> QdrantClient:
    """Qdrant client'ını singleton olarak döndürür."""
    global _client
    if _client is None:
        try:
            _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
            # Bağlantıyı doğrula
            _client.get_collections()
            print(f"[RAG] Qdrant bağlantısı başarılı: {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception as e:
            print(f"[RAG] Qdrant bağlantı hatası: {e}")
            _client = None
            raise ConnectionError(
                f"Qdrant sunucusuna bağlanılamadı ({QDRANT_HOST}:{QDRANT_PORT}). "
                "Docker compose çalışıyor mu? → docker compose up -d"
            ) from e
    return _client


def get_embedding_model() -> SentenceTransformer:
    """Embedding modelini singleton olarak yükler."""
    global _embed_model
    if _embed_model is None:
        print(f"[RAG] Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[RAG] Embedding modeli hazır.")
    return _embed_model


def embed_text(text: str) -> List[float]:
    """Metni vektöre dönüştürür (384 boyutlu)."""
    model = get_embedding_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()


def ensure_collection():
    """Qdrant'ta koleksiyon yoksa oluşturur."""
    client = get_client()
    existing = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"[RAG] Koleksiyon oluşturuldu: {COLLECTION_NAME}")


def index_document(text: str, metadata: Optional[Dict] = None) -> Dict:
    """
    Metni vektöre çevirip Qdrant'a kaydeder.
    
    Args:
        text: Kaydedilecek metin (VLM analiz çıktısı vb.)
        metadata: Ek bilgi (soru, zaman damgası vb.)
    
    Returns:
        İndeksleme sonucu bilgisi
    """
    if metadata is None:
        metadata = {}

    ensure_collection()
    client = get_client()
    vector = embed_text(text)
    point_id = str(uuid.uuid4())

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"text": text, **metadata},
            )
        ],
    )

    print(f"[RAG] Doküman indekslendi: {text[:80]}...")
    return {
        "status": "indexed",
        "id": point_id,
        "text_preview": text[:100],
    }