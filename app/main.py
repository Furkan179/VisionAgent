"""
VisionAgent API — Ana FastAPI uygulaması.
Görsel analiz + RAG ile zenginleştirilmiş yanıt dönen AI servisi.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="VisionAgent API",
    description="AI agent that analyzes images and answers questions using RAG",
    version="1.0.0",
)

# CORS — tüm originlere izin ver (geliştirme ortamı)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Servisin çalışıp çalışmadığını kontrol eder."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = "What do you see in this image?",
):
    """
    Görsel + soru alır, VLM + RAG ile analiz eder.
    Henüz agent modülü tamamlanmadığı için şimdilik placeholder.
    """
    # Dosya tipini kontrol et
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Sadece görsel dosyaları kabul edilir (image/png, image/jpeg vb.)",
        )

    image_bytes = await file.read()

    # Agent modülü hazır olunca bu satır aktif edilecek:
    # from app.agent import run_agent
    # result = await run_agent(image_bytes, question)

    return {
        "status": "placeholder",
        "message": "Agent modülü henüz aktif değil. Görsel alındı.",
        "question": question,
        "image_size_bytes": len(image_bytes),
        "content_type": file.content_type,
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)