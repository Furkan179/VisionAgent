"""
VisionAgent — MLflow Deney Takip Modülü.
Her VLM + RAG analizini MLflow'a loglar.
"""

from typing import Optional, Dict
from dotenv import load_dotenv
import mlflow
import os

# .env dosyasını yükle
load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# MLflow bağlantısı — sunucu yoksa sessizce devam et
_initialized = False


def _init_mlflow():
    """MLflow bağlantısını bir kez kur."""
    global _initialized
    if not _initialized:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            # Experiment yoksa oluştur, varsa ID'sini al
            experiment = mlflow.get_experiment_by_name("VisionAgent")
            if experiment is None:
                mlflow.create_experiment("VisionAgent")
            mlflow.set_experiment("VisionAgent")
            _initialized = True
            print(f"[MLflow] Bağlantı kuruldu: {MLFLOW_URI}")
        except Exception as e:
            print(f"[MLflow] Bağlantı hatası (sunucu kapalı olabilir): {e}")
            _initialized = False


def log_run(question: str, answer: str, duration: float, extra_metrics: Optional[Dict] = None):
    """
    Bir analiz çalışmasını MLflow'a loglar.
    
    Args:
        question: Kullanıcının sorduğu soru
        answer: Modelin ürettiği yanıt
        duration: İşlem süresi (saniye)
        extra_metrics: Ek metrikler (opsiyonel)
    """
    _init_mlflow()

    if not _initialized:
        print("[MLflow] Sunucu erişilemez, log atlanıyor.")
        return

    try:
        with mlflow.start_run():
            # Parametreleri logla
            mlflow.log_param("question", question[:250])
            mlflow.log_param("answer_preview", answer[:250])

            # Metrikleri logla
            mlflow.log_metric("duration_seconds", duration)
            mlflow.log_metric("answer_length", len(answer))
            mlflow.log_metric("question_length", len(question))

            # Ek metrikler varsa ekle
            if extra_metrics:
                for key, value in extra_metrics.items():
                    mlflow.log_metric(key, value)

            # Tam yanıtı parametre olarak kaydet (artifact yerine)
            mlflow.log_param("full_answer", answer[:500])

        print(f"[MLflow] Run loglandı — süre: {duration}s, cevap uzunluğu: {len(answer)}")
    except Exception as e:
        # MLflow hatası uygulamayı çökertmemeli
        print(f"[MLflow] Loglama hatası: {e}")
