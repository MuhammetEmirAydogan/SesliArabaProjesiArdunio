import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
csv_path = os.path.join(desktop_path, "mfcc_features_scaled.csv")

data = pd.read_csv(csv_path)


X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, batch_size=32, 
          validation_data=(X_test, y_test),
          callbacks=[early_stopping])


model_save_path = os.path.join(desktop_path, "voice_command_model.h5")
model.save(model_save_path)
print(f"Model {model_save_path} olarak kaydedildi!")
