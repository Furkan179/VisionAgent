from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModel
import torch
import os
import uuid

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "vision_agent_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = None
embed_model = None
embed_tokenizer = None

def get_client():
    global client
    if client is None:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return client

def get_embedding_model():
    global embed_model, embed_tokenizer
    if embed_model is None:
        embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    return embed_model, embed_tokenizer

def embed_text(text: str) -> list:
    model, tokenizer = get_embedding_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding

def ensure_collection():
    c = get_client()
    existing = [col.name for col in c.get_collections().collections]
    if COLLECTION_NAME not in existing:
        c.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def index_document(text: str, metadata: dict = {}):
    ensure_collection()
    c = get_client()
    vector = embed_text(text)
    c.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, **metadata}
            )
        ]
    )
    return {"status": "indexed", "text_preview": text[:100]}