import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
dataset_path = os.path.join(desktop_path, "dataset")


commands = ["ileri git", "geri gel", "saga dön", "sola dön", "dur"]


X = []
Y = []


for label, command in enumerate(commands):
    command_path = os.path.join(dataset_path, command)
    if not os.path.exists(command_path):
        print(f"Hata: {command_path} klasörü bulunamadı, atlanıyor.")
        continue
    for file_name in os.listdir(command_path):
        if not file_name.endswith(".wav"):
            continue
        file_path = os.path.join(command_path, file_name)
        y, sr = librosa.load(file_path, sr=16000)
        y = librosa.effects.trim(y)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        mfcc_mean = np.mean(combined, axis=1)
        X.append(mfcc_mean)
        Y.append(label)


X = np.array(X)
Y = np.array(Y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


scaler_path = os.path.join(desktop_path, "scaler.save")
joblib.dump(scaler, scaler_path)
print(f"Ölçekleyici {scaler_path} olarak kaydedildi!")


df = pd.DataFrame(X_scaled)
df["label"] = Y
csv_path = os.path.join(desktop_path, "mfcc_features_scaled.csv")
df.to_csv(csv_path, index=False)
print(f"Ön işlenmiş ve ölçeklenmiş veri {csv_path} olarak kaydedildi!")
