import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# === CAMINHO DO DATASET ===
DATA_PATH = "landmarks_augmented_fixed"  # Pasta com os arquivos .npy

# === LISTAS PARA ARMAZENAR OS DADOS ===
X, y = [], []

print("📂 Carregando dataset...")

# === LEITURA DOS ARQUIVOS .NPY ===
for label in os.listdir(DATA_PATH):
    class_path = os.path.join(DATA_PATH, label)
    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        try:
            keypoints = np.load(file_path, allow_pickle=True)

            # Verifica se tem o formato correto (21 landmarks × 3 coords = 63)
            if keypoints.shape[0] == 74:
                X.append(keypoints)
                y.append(label)
            else:
                print(f"⚠️ Pulando arquivo inválido: {file_path} (shape={keypoints.shape})")

        except Exception as e:
            print(f"❌ Erro ao abrir {file_path}: {e}")

print(f"\n✅ Total de amostras válidas: {len(X)}")

# === CONVERSÃO PARA ARRAYS ===
X = np.array(X)
y = np.array(y)

# === ENCODER DAS CLASSES ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# === SALVA O ENCODER PARA O PREDICT ===
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# === DIVISÃO TREINO / TESTE ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === DEFINIÇÃO DO MODELO ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(74,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# === COMPILAÇÃO ===
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === CALLBACK PARA EVITAR OVERFITTING ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === TREINAMENTO ===
print("\n🚀 Iniciando o treinamento...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# === AVALIAÇÃO ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Acurácia final no conjunto de teste: {acc*100:.2f}%")

# === SALVA O MODELO ===
model.save("sign_model.h5")
print("\n💾 Modelo salvo como 'sign_model.h5'")



