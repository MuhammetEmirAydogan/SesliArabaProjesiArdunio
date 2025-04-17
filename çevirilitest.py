import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from scipy.io.wavfile import write
import joblib
import whisper
from deep_translator import GoogleTranslator
from difflib import get_close_matches
import serial 
import time

fs = 16000
seconds = 2
n_mfcc = 20
model_confidence_threshold = 0.7 

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_path = os.path.join(desktop_path, "voice_command_model.h5")
scaler_path = os.path.join(desktop_path, "scaler.save")
temp_audio_path = os.path.join(desktop_path, "temp_command_audio.wav") 


commands = ["ileri git", "geri gel", "saga dön", "sola dön", "dur"]
command_map = {
    "ileri git": "0",
    "geri gel": "1",
    "saga dön": "2",
    "sola dön": "3",
    "dur": "4"
}



arduino_port = "COM7" 
baud_rate = 9600
arduino = None 


try:
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    print("Keras modeli ve scaler yüklendi.")
    whisper_model = whisper.load_model("small") 
    print(" Whisper modeli yüklendi.")
except Exception as e:
    print(f"Modeller yüklenirken hata: {e}")
    exit()


def send_command_usb(command_text):
    global arduino 
    if arduino and arduino.is_open:
        arduino_char = command_map.get(command_text)
        if arduino_char:
            try:
                arduino.write(arduino_char.encode()) 
                print(f" USB ile gönderildi: {command_text} -> '{arduino_char}'")
            except Exception as send_error:
                print(f" Komut gönderirken hata: {send_error}")
        else:
            print(f" Bilinmeyen komut: '{command_text}'. Arduino'ya gönderilmedi.")
    else:
        print(" Arduino bağlantısı kapalı veya yok. Komut gönderilemedi.")


def record_audio():
    print(f"\n Komutu söyleyin... ({seconds} saniye kayıt alınacak)")
    try:
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        return recording
    except Exception as audio_error:
        print(f" Ses kaydı sırasında hata: {audio_error}")
        return None


def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=fs) 
        if len(y) == 0:
            print(" Ses dosyası boş veya yüklenemedi.")
            return None

        
        y_trimmed, index = librosa.effects.trim(y, top_db=25) 
        if len(y_trimmed) == 0:
            print(" Kırpma sonrası ses boş veya çok kısa.")
            
            y_trimmed = y
            if len(y_trimmed) == 0: return None

        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])

        if combined.size == 0:
            print(" Özellik çıkarılamadı (muhtemelen çok kısa ses).")
            return None
        mfcc_mean = np.mean(combined, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f" MFCC çıkarılırken hata: {e}")
        return None


def predict_command(features):
    if features is None:
        return None, 0.0
    try:
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if 0 <= predicted_index < len(commands):
             return commands[predicted_index], confidence
        else:
            print(f" Geçersiz model tahmin indeksi: {predicted_index}")
            return None, 0.0
    except Exception as e:
        print(f" Model tahmini sırasında hata: {e}")
        return None, 0.0


def translate_and_match(whisper_text):
    if not whisper_text or whisper_text.strip() == "":
         print("Whisper çıktısı boş.")
         return None
    try:
        
        if any(c in whisper_text for c in "ğüşöçİĞÜŞÖÇ"): 
            translated = whisper_text
            print(f"Whisper çıktısı zaten Türkçe görünüyor: '{translated}'")
        else:
            
            translated = GoogleTranslator(source='auto', target='tr').translate(whisper_text.lower())
            print(f"Whisper çıktısı: '{whisper_text}' -> Türkçe çeviri: '{translated}'")

       
        match = get_close_matches(translated.lower(), commands, n=1, cutoff=0.6) 
        if match:
            print(f" Whisper + Translate eşleşen komut: {match[0]}")
            return match[0]
        else:
            print(" Whisper çevirisinden yakın eşleşme bulunamadı.")
            return None
    except Exception as e:
        print(f" Çeviri veya eşleştirme hatası: {e}")
        return None


def main():
    global arduino 

    
    try:
        arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
        print(f" Arduino'ya bağlanılıyor ({arduino_port})...")
        time.sleep(2) 
        print(f" Arduino bağlantısı başarılı!")
    except serial.SerialException as se:
        print(f" Seri porta bağlanılamadı ({arduino_port}): {se}")
        print(" Lütfen Arduino'nun bağlı ve doğru portun seçili olduğundan emin olun.")
        return 
    except Exception as e:
        print(f" Beklenmedik bağlantı hatası: {e}")
        return 

    try:
        while True:
            print("\n" + "="*30)
            final_command_to_send = None 

           
            audio_recording = record_audio()
            if audio_recording is None:
                continue 

           
            try:
                write(temp_audio_path, fs, audio_recording)
                print(f" Geçici ses dosyası kaydedildi: {temp_audio_path}")
            except Exception as write_error:
                print(f" Ses dosyası yazılırken hata: {write_error}")
                continue 

            
            print(" Whisper konuşmayı metne çeviriyor...")
            try:
                result = whisper_model.transcribe(temp_audio_path, fp16=False) 
                spoken_text = result['text'].strip()
                print(f" Whisper Algılanan Metin: '{spoken_text}'")
            except Exception as whisper_error:
                print(f" Whisper çevrim hatası: {whisper_error}")
                spoken_text = "" 
            
            whisper_matched_command = translate_and_match(spoken_text)

           
            print(" MFCC modeli analiz ediyor...")
            mfcc_features = extract_features(temp_audio_path)
            model_command, confidence = predict_command(mfcc_features)
            if model_command:
                print(f" Model Tahmini: {model_command} (Güven: {confidence:.2f})")
            else:
                 print(" Model bir komut tahmin edemedi.")

            if whisper_matched_command:
                final_command_to_send = whisper_matched_command
                print(f" Karar: Whisper+Translate sonucu kullanılacak -> {final_command_to_send}")
           
            elif model_command and confidence >= model_confidence_threshold:
                final_command_to_send = model_command
                print(f" Karar: Model tahmini kullanılacak (Güven > {model_confidence_threshold}) -> {final_command_to_send}")
            
            else:
                print(" Güvenilir bir komut belirlenemedi. Arduino'ya komut gönderilmeyecek.")

            
            if final_command_to_send:
                send_command_usb(final_command_to_send)
            else:
                 
                 pass 
            
            try:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
            except Exception as delete_error:
                print(f" Geçici ses dosyası silinirken hata: {delete_error}")
            
            again = input("\nTekrar denemek için ENTER'a basın (çıkmak için 'q' yazıp Enter): ")
            if again.lower() == 'q':
                print(" Çıkılıyor...")
                break

    finally:
        
        if arduino and arduino.is_open:
            print("\n Çıkış öncesi Arduino'ya son bir 'dur' komutu gönderiliyor...")
            send_command_usb("dur")
            time.sleep(0.5) 
            arduino.close()
            print(" Arduino bağlantısı kapatıldı.")
        
        try:
             if os.path.exists(temp_audio_path):
                 os.remove(temp_audio_path)
        except Exception:
            pass 

if __name__ == "__main__":
    main()