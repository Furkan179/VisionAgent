# 🤖 VisionAgent

**AI-powered visual analysis agent** — Görselleri analiz eden, geçmiş analizlerden öğrenen ve zenginleştirilmiş yanıtlar üreten uçtan uca yapay zeka pipeline'ı.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Qwen2-VL](https://img.shields.io/badge/Qwen2--VL--2B-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Mimari](#-mimari)
- [Tech Stack](#-tech-stack)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [API Referansı](#-api-referansı)
- [Proje Yapısı](#-proje-yapısı)
- [Pipeline Detayları](#-pipeline-detayları)

---

## 🎯 Proje Hakkında

VisionAgent, kullanıcının yüklediği bir görseli ve soruyu alarak **Qwen2-VL** görsel dil modeli ile analiz eden, **Qdrant** vektör veritabanı üzerinden geçmiş analizlerden bağlam çeken (RAG) ve **LangGraph** ile orkestre edilen bir AI ajanıdır.

### Ne Yapar?

1. 📸 Kullanıcı bir **görsel + soru** yükler (`/analyze` endpoint)
2. 🧠 **Qwen2-VL-2B** modeli görseli analiz eder (Apple Silicon MPS GPU desteği)
3. 🔍 **Qdrant** vektör veritabanından ilgili geçmiş analizler **RAG** ile çekilir
4. 🔗 **LangGraph StateGraph** pipeline'ı tüm adımları orkestre eder
5. 📊 Her çalışma **MLflow**'a loglanır (süre, metrikler, parametreler)
6. ✅ Zenginleştirilmiş yanıt kullanıcıya döner

---

## 🏗 Mimari

```
┌──────────────┐     ┌────────────────────────────────────────┐
│   Client     │     │          VisionAgent API               │
│  (curl/web)  │────▶│           FastAPI :8000                │
└──────────────┘     │                                        │
                     │  ┌──────────────────────────────────┐  │
                     │  │     LangGraph StateGraph          │  │
                     │  │                                    │  │
                     │  │  ┌──────────┐   ┌─────────────┐  │  │
                     │  │  │ analyze  │──▶│  retrieve    │  │  │
                     │  │  │ (VLM)    │   │  (RAG)      │  │  │
                     │  │  └──────────┘   └──────┬──────┘  │  │
                     │  │                         │         │  │
                     │  │                 ┌───────▼──────┐  │  │
                     │  │                 │ synthesize   │  │  │
                     │  │                 │ (Birleştir)  │  │  │
                     │  │                 └──────────────┘  │  │
                     │  └──────────────────────────────────┘  │
                     └──────────┬──────────────┬──────────────┘
                                │              │
                     ┌──────────▼──┐   ┌───────▼───────┐
                     │   Qdrant    │   │    MLflow     │
                     │  :6333      │   │   :5001       │
                     │ Vector DB   │   │  Experiment   │
                     │ (Docker)    │   │  Tracking     │
                     └─────────────┘   └───────────────┘
```

---

## 🛠 Tech Stack

| Kategori | Teknoloji | Açıklama |
|----------|-----------|----------|
| **API Framework** | FastAPI | Async REST API, otomatik Swagger dokümantasyonu |
| **Agent Orchestration** | LangGraph (LangChain) | StateGraph ile 3 düğümlü agentic pipeline |
| **Vision Language Model** | Qwen2-VL-2B-Instruct | Multimodal görsel analiz (HuggingFace) |
| **Vector Database** | Qdrant | Semantik arama ve RAG hafızası |
| **Embedding** | sentence-transformers/all-MiniLM-L6-v2 | 384 boyutlu metin vektörleri |
| **Experiment Tracking** | MLflow | Run metrikleri, parametreler, deney karşılaştırma |
| **Containerization** | Docker Compose | Qdrant + MLflow servisleri |
| **Hardware Optimization** | Apple Silicon MPS | M1/M2/M3 GPU hızlandırma, float16 inference |

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.9+
- Docker Desktop ([indir](https://docs.docker.com/desktop/setup/install/mac-install/))
- ~5 GB disk alanı (model ağırlıkları için)

### 1. Repoyu Klonla

```bash
git clone https://github.com/furkan179/VisionAgent.git
cd VisionAgent
```

### 2. Sanal Ortam Oluştur

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ortam Değişkenlerini Ayarla

```bash
cp .env.example .env
```

### 4. Docker Servislerini Başlat

```bash
docker compose up -d qdrant mlflow
```

Bu komut arka planda şunları çalıştırır:
- **Qdrant** → `http://localhost:6333` (Vektör veritabanı)
- **MLflow** → `http://localhost:5001` (Deney takip paneli)

### 5. API'yi Başlat

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **Not:** İlk istekte Qwen2-VL model ağırlıkları HuggingFace'den indirilir (~4.5 GB). Bu işlem internet hızınıza göre 5-15 dakika sürebilir. Sonraki istekler çok daha hızlıdır.

---

## 💡 Kullanım

### Görsel Analiz İsteği

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@foto.jpg" \
  -F "question=Bu resimde ne görüyorsun?"
```

### Yanıt Örneği

```json
{
  "question": "Bu resimde ne görüyorsun?",
  "vision_analysis": "Resimde bir kafe ortamında oturan insanlar görünüyor...",
  "context_used": [
    {
      "text": "Önceki bir analizden benzer sahne...",
      "score": 0.87,
      "metadata": {"question": "...", "timestamp": 1774820349.68}
    }
  ],
  "final_answer": "Resimde bir kafe ortamında oturan insanlar görünüyor...\n\n📚 Geçmiş Analizlerden Bağlam:\n...",
  "duration_seconds": 24.26,
  "error": null
}
```

### Sağlık Kontrolü

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "VisionAgent"}
```

### Swagger UI

Tarayıcıda `http://localhost:8000/docs` adresini açarak interaktif API dokümantasyonuna erişebilirsiniz.

### MLflow Dashboard

Tarayıcıda `http://localhost:5001` adresini açarak deney metriklerini ve run geçmişini görüntüleyebilirsiniz.

---

## 📚 API Referansı

### `GET /health`

Servis sağlık kontrolü.

| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| — | — | Parametre almaz |

**Yanıt:** `{"status": "ok", "service": "VisionAgent"}`

---

### `POST /analyze`

Görsel analiz endpoint'i.

| Parametre | Tip | Zorunlu | Açıklama |
|-----------|-----|---------|----------|
| `file` | `UploadFile` | ✅ | Analiz edilecek görsel (PNG, JPEG vb.) |
| `question` | `string` | ❌ | Görsel hakkındaki soru (varsayılan: "What do you see in this image?") |

**Yanıt Alanları:**

| Alan | Tip | Açıklama |
|------|-----|----------|
| `question` | string | Gönderilen soru |
| `vision_analysis` | string | VLM'in ham analizi |
| `context_used` | array | RAG ile getirilen geçmiş kayıtlar |
| `final_answer` | string | Bağlamla zenginleştirilmiş son yanıt |
| `duration_seconds` | float | Toplam işlem süresi |
| `error` | string\|null | Hata mesajı (varsa) |

---

## 📁 Proje Yapısı

```
VisionAgent/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI endpoints (/health, /analyze)
│   ├── agent.py            # LangGraph StateGraph pipeline
│   ├── vision.py           # Qwen2-VL inference (MPS optimized)
│   ├── tracking.py         # MLflow experiment logging
│   └── rag/
│       ├── __init__.py
│       ├── indexer.py       # Qdrant'a doküman kaydet (embedding)
│       └── retriever.py     # Qdrant'tan semantik arama
├── docker-compose.yml       # Qdrant + MLflow servisleri
├── Dockerfile               # API container tanımı
├── requirements.txt         # Python bağımlılıkları
├── .env.example             # Ortam değişkenleri şablonu
└── test_vision.py           # VLM modül testi
```

---

## ⚙️ Pipeline Detayları

### LangGraph Agent Akışı

VisionAgent, **LangGraph StateGraph** ile orkestre edilen 3 aşamalı bir pipeline kullanır:

```python
# Akış: START → analyze → retrieve → synthesize → END

class AgentState(TypedDict):
    image_bytes: bytes        # Kullanıcının görseli
    question: str             # Kullanıcının sorusu
    vision_result: str        # VLM çıktısı
    context_docs: List[Dict]  # RAG sonuçları
    final_answer: str         # Zenginleştirilmiş yanıt
    duration: float           # İşlem süresi
    error: Optional[str]      # Hata (varsa)
```

| Düğüm | Görev | Detay |
|-------|-------|-------|
| `analyze` | Görsel analiz | Qwen2-VL ile görseli ve soruyu işler |
| `retrieve` | RAG sorgusu | VLM çıktısını embedding'e çevirip Qdrant'ta arar |
| `synthesize` | Birleştirme | VLM çıktısı + RAG bağlamını birleştirir, Qdrant'a kaydeder |

### RAG (Retrieval-Augmented Generation)

- **Embedding Model:** `all-MiniLM-L6-v2` (384 boyutlu vektör)
- **Vector DB:** Qdrant (cosine similarity)
- Her analiz sonucu otomatik olarak Qdrant'a kaydedilir
- Sonraki sorgularda semantik benzerliğe göre geçmiş analizler getirilir

### MLflow Tracking

Her `/analyze` isteği şu metrikleri loglar:

| Metrik | Açıklama |
|--------|----------|
| `duration_seconds` | Toplam pipeline süresi |
| `answer_length` | Üretilen yanıtın karakter uzunluğu |
| `question_length` | Sorunun karakter uzunluğu |
| `context_count` | RAG'dan getirilen doküman sayısı |

---

## 🔧 Geliştirme Notları

### Apple Silicon (M1/M2/M3) Optimizasyonu

- PyTorch **MPS** (Metal Performance Shaders) backend'i otomatik algılanır
- Model ağırlıkları `float16` ile yüklenir (bellek tasarrufu)
- MPS uyumsuzluğu durumunda otomatik CPU fallback

### Graceful Degradation

- Qdrant kapalıysa → RAG atlanır, salt VLM yanıtı döner
- MLflow kapalıysa → Loglama atlanır, analiz devam eder
- Hiçbir harici servis arızası API'yi çökertmez

---

## 📄 Lisans

MIT License — Detaylar için [LICENSE](LICENSE) dosyasına bakınız.