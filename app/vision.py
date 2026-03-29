"""
VisionAgent — Qwen2-VL Görsel Analiz Modülü.
Kullanıcının yüklediği görseli ve metni işleyerek VLM modeli ile analiz eder.
Apple Silicon (M3) için MPS ve Float optimizasyonları yapılmıştır.
"""

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import io
import os

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")

# Modeli bellekte tutmak için global değişkenler
model = None
processor = None

def get_device() -> str:
    """M3 için MPS desteğini kontrol et, yoksa CPU dön."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model():
    """Qwen2-VL modelini ve processor'ı yükleyip bellekte tutar (Singleton)."""
    global model, processor
    if model is None:
        print(f"[Hizmet] Modeli yüklemeye başlıyorum: {MODEL_NAME}")
        device = get_device()
        
        # M3 üzerinde MPS için float16 daha stabil çalışıyor olabilir
        # CPU için bfloat16 kullanılabilir. 
        dtype = torch.float16 if device == "mps" else torch.bfloat16
        
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                device_map=device
            )
        except Exception as e:
            print(f"[Uyarı] MPS yükleme hatası: {e}. Sadece CPU moduna düşülüyor...")
            device = "cpu"
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch.float32, 
                device_map="cpu"
            )
            
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        print(f"[Hizmet] Model başarıyla yüklendi! Cihaz: {device}, Veri Tipi: {model.dtype}")
    
    return model, processor

def analyze_image(image_bytes: bytes, question: str) -> str:
    """
    Görseli ve soruyu alarak, modele analiz yaptırır ve yanıt döner.
    """
    model, processor = load_model()

    # Byte dizisini PIL objesine dönüştür
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Qwen2-VL'nin beklediği yapı
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    # Text template'i oluştur
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Görüntü padding/işleme ayarları
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Tensorları GPU / MPS / CPU ortamına taşı
    inputs = inputs.to(model.device)

    # İnference (Analiz) aşaması
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
        )
    
    # Modele verilen context inputlarını çıktıdan kes
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]