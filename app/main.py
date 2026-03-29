"""
VisionAgent API — Ana FastAPI uygulaması.
Görsel analiz + RAG ile zenginleştirilmiş yanıt dönen AI servisi.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken/kapanırken çalışan lifecycle hook."""
    print("[API] VisionAgent başlatılıyor...")
    # Model yüklemesini ilk istek yerine burada da tetikleyebiliriz
    # Ancak başlangıç süresini uzatır, bu yüzden lazy loading tercih ediyoruz
    yield
    print("[API] VisionAgent kapatılıyor...")


app = FastAPI(
    title="VisionAgent API",
    description=(
        "Qwen2-VL görsel dil modeli ve Qdrant RAG ile güçlendirilmiş "
        "yapay zeka ajanı. Görselleri analiz eder, geçmiş analizlerden "
        "bağlam çeker ve zenginleştirilmiş yanıtlar üretir."
    ),
    version="1.0.0",
    lifespan=lifespan,
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
    return {"status": "ok", "service": "VisionAgent"}


@app.post("/analyze")
async def analyze_image_endpoint(
    file: UploadFile = File(..., description="Analiz edilecek görsel dosyası"),
    question: str = Query(
        default="What do you see in this image?",
        description="Görsel hakkında sorulacak soru",
    ),
):
    """
    Görsel + soru alır → Qwen2-VL ile analiz eder → Qdrant RAG ile
    geçmiş bağlamı çeker → Zenginleştirilmiş yanıt döner.
    Her çalışma MLflow'a loglanır.
    """
    # Dosya tipini kontrol et
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Sadece görsel dosyaları kabul edilir (image/png, image/jpeg vb.)",
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Boş dosya yüklendi.")

    # LangGraph agent pipeline'ını çalıştır
    from app.agent import run_agent

    result = await run_agent(image_bytes, question)

    # Eğer agent hata döndüyse 500 dön
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return result


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)