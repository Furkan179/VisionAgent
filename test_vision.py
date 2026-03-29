import sys
import os

# Sisteme app dizinini tanıt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision import analyze_image

def main():
    print("VLM testi başlatılıyor...")
    image_path = "/tmp/visionagent_test/sample.jpg"
    
    if not os.path.exists(image_path):
        print("Test resmi bulunamadı. Lütfen önce resmi indirin.")
        sys.exit(1)
        
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print("Resim yüklendi. Soru: 'Bu resimde neler görüyorsun? Kısa ve öz tarif et.'")
    question = "Bu resimde neler görüyorsun? Kısa ve öz tarif et."
    
    # Not: İlk analiz modeli HF'den indireceği için internet hızınıza göre 5-10 dk sürebilir
    print("Model hazırlanıyor ve analiz ediliyor (Modelin inmesi uzun sürebilir)...")
    
    try:
        response = analyze_image(image_bytes, question)
        print("\n--- ANALİZ SONUCU ---")
        print(response)
        print("---------------------")
    except Exception as e:
        print(f"\n[Hata oluştu]: {e}")

if __name__ == "__main__":
    main()
