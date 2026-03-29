---
description: VisionAgent Proje Bağlamı ve Devam Rehberi
---

# VisionAgent — Proje Rehberi (Project Brain)

Bu dosya sistem ajanlarının projeyi bir defada "tek nefeste" anlayabilmesi, teknoloji yığınını ve mevcut durumu unutmaması için oluşturulmuştur. Ajanlar `view_file` ile bu dosyaya bakarak projenin hangi noktasında olduğumuzu hatırlayabilir.

## 🎯 Hedef
İleri MLOps ve AI mühendisliği iş başvurularında kullanılmak üzere hazırlanmış (Örn: Turkish Technology, MELIC); RAG, LangChain, Qdrant, MLflow ve VLM ile kurgulanmış uca-uca bir yapay zeka ajanı geliştirmek.

## 🛠 Tech Stack (Katı Sınırlar)
- **FastAPI** — İstekleri karşılayan REST API
- **LangChain + LangGraph** — RAG ve zincirleme cevap üretimi (Agentic Flow)
- **Qwen2-VL-2B-Instruct** — Görsel dil modeli (VLM). *ÖNEMLİ: Apple Silicon (M3) GPU hızlandırıcılarına uygun `mps` cihaz optimizasyonu ve float16 veri tipi kullanılır. Model singleton objede bellekte tutulmalıdır.*
- **Qdrant** — Vektör Veritabanı (RAG belleği için, Docker üzerinden)
- **MLflow** — Deney takibi ve run metrikleri (Docker üzerinden)
- **Python:** 3.9 (Yerel venv ortamı, M3 uyumu için)

## ✅ Başarılanlar (Geçmiş Adımlar)
1. **Ortam ve Altyapı:** `requirements.txt` güncellendi, versiyon çakışmaları (`pydantic, torchvision, accelerate`) giderildi. FastAPI `/health` endpoint'i başarılı.
2. **VLM Modülü (`app/vision.py`):** Qwen2-VL entegrasyonu tamamlandı. Bellek optimizasyonları ve `qwen_vl_utils` işleme metotları ile test resmi üzerinde başarılı mps inference'ı yapıldı (`test_vision.py`).
3. **RAG Sistemi (`app/rag/`):** `indexer.py` (sentence-transformers ile embedding + Qdrant upsert) ve `retriever.py` (semantik arama) tamamlandı. Savunmacı hata yakalama ile Qdrant down olsa bile uygulama çökmez.
4. **LangGraph Agent (`app/agent.py`):** Gerçek StateGraph ile 3 düğümlü pipeline (analyze → retrieve → synthesize). Singleton compiled graph.
5. **MLflow Tracking (`app/tracking.py`):** Lazy init, graceful degradation. Her analiz run'ını loglar.
6. **main.py Entegrasyonu:** Tüm modüller birleştirildi, `/analyze` endpoint'i gerçek LangGraph pipeline'ına bağlı.
7. **GitHub Push:** Proje `furkan179/VisionAgent` reposuna pushlandı.

## 🚀 Sonraki Görevler (Sıradaki Bekleyenler)
1. **Docker Kurulumu:** Mac'e Docker Desktop kurulacak, `docker compose up -d` ile Qdrant + MLflow servisleri ayaklandırılacak.
2. **Uçtan Uca Test:** Docker servisleri çalışırken `/analyze` endpoint'ine görsel gönderip tüm pipeline'ı (VLM → RAG → MLflow) test etmek.
3. **README.md:** Proje dokümantasyonu ve kurulum rehberi yazmak.

## 💡 Kurallar
- (Kullanıcının Global Kuralları): Türkçe iletişim, DRY, Defensive Coding (hata yakalama `try/except`), güncel ve asenkron modern syntax.
- Bütün değişiklikler adım adım ve tek dosya bazında tamamlanır. Adım sonu çalışabilir test scripti/komutu üretilir.
