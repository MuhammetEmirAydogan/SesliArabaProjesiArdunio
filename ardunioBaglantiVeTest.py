import serial 
import time
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import tensorflow as tf
import joblib
import datetime
import os


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_path = os.path.join(desktop_path, "voice_command_model.h5")
scaler_path = os.path.join(desktop_path, "scaler.save")



commands = ["ileri git", "geri gel", "saga dön", "sola dön", "dur"]
command_map = {
    "ileri git": "0",
    "geri gel": "1",
    "saga dön": "2",
    "sola dön": "3",
    "dur": "4"
}


try:
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    print(" Model ve scaler yüklendi!")
except Exception as e:
    print(f" Model veya scaler yüklenirken hata: {e}")
    exit() 


test_path = os.path.join(desktop_path, "testler")
os.makedirs(test_path, exist_ok=True)


arduino_port = "COM7" 
baud_rate = 9600

arduino = None 
try:
   
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2) 
    print(f" Arduino'ya {arduino_port} portundan bağlanıldı!")
except serial.SerialException as se:
    print(f" Seri porta bağlanılamadı ({arduino_port}): {se}")
    print("İpuçları:")
    print("- Arduino'nun bağlı olduğundan emin olun.")
    print("- Doğru COM portunu seçtiğinizden emin olun (Arduino IDE'den kontrol edin).")
    print("- Başka bir programın (örneğin Arduino IDE Seri Monitör) portu kullanmadığından emin olun.")
    exit() 
except Exception as e:
    print(f" Beklenmedik bir bağlantı hatası: {e}")
    exit() 


def send_command_usb(command_text):
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


def recognize_command():
    fs = 16000
    seconds = 2

    print("\n Komutu söyle (2 saniye):")
    try:
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
    except Exception as audio_error:
        print(f" Ses kaydı sırasında hata: {audio_error}")
        return 

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_audio_path = os.path.join(test_path, f"test_{timestamp}.wav")
    try:
        write(test_audio_path, fs, recording)
        print(f" Ses kaydedildi: {test_audio_path}")
    except Exception as write_error:
        print(f" Ses dosyası yazılırken hata: {write_error}")
        return 

    try:
        y, sr = librosa.load(test_audio_path, sr=16000)
        if len(y) == 0:
            print(" Kaydedilen ses boş veya çok kısa. Komut algılanamadı.")
            return
        
        y_trimmed, _ = librosa.effects.trim(y, top_db=20) 
        if len(y_trimmed) == 0:
            print(" Kırpma sonrası ses boş kaldı. Komut algılanamadı.")
            return
        y = y_trimmed

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        
        if combined.size == 0:
             print(" Özellik çıkarılamadı (muhtemelen çok kısa ses).")
             return
        mfcc_mean = np.mean(combined, axis=1)

       
        X_input = scaler.transform([mfcc_mean])
        
        prediction = model.predict(X_input, verbose=0)
        predicted_label_index = np.argmax(prediction)
        confidence = np.max(prediction) 

       
        if 0 <= predicted_label_index < len(commands):
            command_output = commands[predicted_label_index]
            print(f" Tahmin edilen komut: {command_output} (Güven: {confidence:.2f})")
             
            send_command_usb(command_output)
        else:
            print(f"Geçersiz tahmin indeksi: {predicted_label_index}")

    except Exception as process_error:
        print(f" Ses işleme veya tahmin sırasında hata: {process_error}")



try:
    recognize_command()

    while True:
        user_input = input("\nYeni komut için ENTER'a bas, çıkmak için q yaz: ")
        if user_input.lower() == 'q':
            print("Çıkılıyor...")
            break
        recognize_command()

finally:
    
    if arduino and arduino.is_open:
        print("Çıkış öncesi Arduino'ya dur komutu gönderiliyor...")
        send_command_usb("dur")
        time.sleep(0.5) 
        arduino.close()
        print("Arduino bağlantısı kapatıldı.")