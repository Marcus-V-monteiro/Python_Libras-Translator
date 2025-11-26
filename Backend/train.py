import numpy as np
import cv2
import mediapipe as mp
import os
import albumentations as A

# === CONFIGURAÇÕES DO MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === CAMINHOS ===
DATASET_PATH = r'C:\Users\marcu\Desktop\College\Projects\PEX -ECP6NA\Datasets\aplhabet'
OUTPUT_PATH = 'landmarks_augmented_fixed'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# === AUGMENTATIONS (Albumentations) ===
augment = A.Compose([
    A.RandomBrightnessContrast(p=0.4),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.4),
    A.Blur(blur_limit=3, p=0.2),
    A.RandomGamma(p=0.3),
    A.RandomScale(scale_limit=0.2, p=0.3)
])

# === FUNÇÃO PARA CALCULAR FEATURES DERIVADAS ===
def extract_features(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 3)
    wrist = keypoints[0]
    keypoints -= wrist  # Normaliza com base no pulso

    features = keypoints.flatten().tolist()

    # Distâncias entre pontas dos dedos (10 valores fixos)
    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            dist = float(np.linalg.norm(keypoints[fingertips[i]] - keypoints[fingertips[j]]))
            features.append(dist)

    # Abertura da mão
    abertura = float(np.linalg.norm(keypoints[4] - keypoints[20]))
    features.append(abertura)

    # Garante shape fixo (74)
    features = (features + [0.0] * 74)[:74]
    return np.array(features, dtype=np.float32)

# === FUNÇÃO SEGURA PARA PROCESSAR ===
def process_image_safe(image):
    """Tenta extrair landmarks e sempre retorna vetor válido (74,)"""
    try:
        if image is None or image.size == 0:
            return np.zeros(74, dtype=np.float32)

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            return extract_features(keypoints)
        else:
            # Nenhuma mão detectada
            return np.zeros(74, dtype=np.float32)

    except Exception as e:
        # Qualquer erro inesperado → vetor padrão
        print(f"⚠️ Erro ao processar imagem ({type(e).__name__}): {e}")
        return np.zeros(74, dtype=np.float32)

# === LOOP PRINCIPAL ===
for label in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_class_path, exist_ok=True)
    print(f"\n📁 Processando letra: {label}")

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)

        # Sempre gera arquivo — mesmo se imagem for inválida
        keypoints = process_image_safe(image)

        # Salva original
        base_name = os.path.splitext(image_name)[0]
        np.save(os.path.join(output_class_path, base_name + ".npy"), keypoints, allow_pickle=False)

        # Gera augmentations
        num_aug = 6 if label in ['F', 'T', 'U'] else 3
        for i in range(num_aug):
            try:
                augmented = augment(image=image)["image"]
            except Exception:
                augmented = image  # fallback se falhar
            aug_keypoints = process_image_safe(augmented)
            np.save(os.path.join(output_class_path, f"{base_name}_aug{i}.npy"), aug_keypoints, allow_pickle=False)

print("\n✅ Dataset completo gerado sem pular nenhum arquivo! Salvo em:", OUTPUT_PATH)







