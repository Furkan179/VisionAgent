from app.vision import analyze_image
from app.rag.retriever import retrieve
from app.rag.indexer import index_document
from mlflow.tracking import log_run
import time

async def run_agent(image_bytes: bytes, question: str) -> dict:
    start = time.time()

    # 1. VLM ile görseli analiz et
    vision_result = analyze_image(image_bytes, question)

    # 2. RAG — geçmişten ilgili kayıtları getir
    context_docs = retrieve(vision_result, top_k=3)
    context_text = "\n".join([d["text"] for d in context_docs])

    # 3. Bağlamla zenginleştirilmiş final yanıt
    if context_text:
        final_answer = f"{vision_result}\n\n[Context from memory]:\n{context_text}"
    else:
        final_answer = vision_result

    # 4. Bu analizi Qdrant'a kaydet (sonraki sorgular için)
    index_document(
        text=vision_result,
        metadata={"question": question, "timestamp": time.time()}
    )

    duration = round(time.time() - start, 2)

    # 5. MLflow'a logla
    log_run(question=question, answer=final_answer, duration=duration)

    return {
        "question": question,
        "vision_analysis": vision_result,
        "context_used": context_docs,
        "final_answer": final_answer,
        "duration_seconds": duration
    }