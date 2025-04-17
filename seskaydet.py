import os
import sounddevice as sd
from scipy.io.wavfile import write


fs = 16000  
seconds = 2  

commands = ["ileri git", "geri gel", "saga dön", "sola dön", "dur"]

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
dataset_path = os.path.join(desktop_path, "dataset") 

os.makedirs(dataset_path, exist_ok=True)

num_samples = 40  
for command in commands:
    command_path = os.path.join(dataset_path, command)
    os.makedirs(command_path, exist_ok=True)
    for i in range(num_samples):
        input(f"{command.upper()} komutunu söylemeye hazır mısın? Enter'a bas...")
        print(f"{command} komutunu söylüyorsunuz...")
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  
        file_path = os.path.join(command_path, f"{command}_{i}.wav")
        write(file_path, fs, recording)
        print(f"Kayıt {file_path} olarak kaydedildi!\n")
