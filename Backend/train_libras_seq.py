"""
train_libras_seq.py
Treina LSTM a partir do dataset gerado.
Saída: libras_seq_model.h5 e classes.npy
"""
import os
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_DIR = "synthetic_dataset"
MODEL_OUT = "libras_seq_model.h5"
SEQ_LEN = 32  # deve bater com o gerador
# -------- load data ----------
gestures = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d)) and d!="renders"])
X = []
y = []
for g in gestures:
    files = sorted(glob.glob(os.path.join(DATA_DIR,g,"*.npy")))
    for f in files:
        seq = np.load(f)  # shape (T, K, 2)
        if seq.shape[0] != SEQ_LEN:
            # pad/truncate
            if seq.shape[0] < SEQ_LEN:
                pad = np.tile(seq[-1:], (SEQ_LEN - seq.shape[0], 1, 1))
                seq = np.concatenate([seq, pad], axis=0)
            else:
                seq = seq[:SEQ_LEN]
        T,K,_ = seq.shape
        X.append(seq.reshape(T, K*2))
        y.append(g)
X = np.array(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)
np.save("classes.npy", le.classes_)
print("Loaded data:", X.shape, "labels", len(le.classes_))

# split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, stratify=y_enc, random_state=42)

# model
model = Sequential([
    LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUT, monitor='val_loss', save_best_only=True)
]

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=32, callbacks=callbacks)
model.save(MODEL_OUT)
print("Saved model to", MODEL_OUT)
