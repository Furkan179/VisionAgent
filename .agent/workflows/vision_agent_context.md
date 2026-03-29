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

## 🚀 Sonraki Görevler (Sıradaki Bekleyenler)
1. **RAG Entegrasyonu (`app/rag/indexer.py` & `retriever.py`):** Docker ile qdrant sunucusunun ayağa kalktığından emin olup, VLM çıktılarını + orijinal soruyu kaydedip getirebilecek basit bir RAG sistemi kurgulanacak.
2. **LangGraph Agent (`app/agent.py`):** VLM sonucunu alıp Qdrant ile karşılaştıran ve nihai akıllı yanıtı üreten agent akışı yazılacak.
3. **MLFlow (`mlflow/tracking.py`):** İsteklerin Qwen + RAG sonucundaki response time ve veri logları MLFlow serverı üzerinden (Docker port 5000) track edilecek.

## 💡 Kurallar
- (Kullanıcının Global Kuralları): Türkçe iletişim, DRY, Defensive Coding (hata yakalama `try/except`), güncel ve asenkron modern syntax.
- Bütün değişiklikler adım adım ve tek dosya bazında tamamlanır. Adım sonu çalışabilir test scripti/komutu üretilir.
