import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # força CPU

import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
from collections import deque
from datetime import datetime
import csv

# ========== CONFIG ==========
MODEL_PATH = "sign_model.h5"          # seu modelo original (por-frame ou imagem)
SEQ_MODEL_PATH = "seq_model.h5"       # opcional: modelo sequencial (LSTM) para movimentos
LABEL_ENCODER_PATH = "label_encoder.pkl"
SEQ_WINDOW_LEN = 32
PRED_EVERY_N_FRAMES = 3
HOLD_FRAMES = 10
MIN_CONFIDENCE = 0.6
WEIGHT_FRAME = 1.0
WEIGHT_SEQ = 1.6
LOG_CSV_PATH = os.path.join("logs", "predictions.csv")
SNAPSHOT_DIR = "snapshots"
SNAPSHOT_THRESH = 0.90
DRAW_LANDMARKS = False
TOP3_X = 10
TOP3_Y = 400
LINE_SPACING = 30
# ============================

# guarda do último top-3 válido (evita flicker)
last_top_labels = []
last_top_probs = []

# cria pastas
os.makedirs(os.path.dirname(LOG_CSV_PATH) or ".", exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ---------- carregar encoder ----------
encoder = None
if os.path.exists(LABEL_ENCODER_PATH):
    with open(LABEL_ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    print("Label encoder carregado com classes:", list(encoder.classes_))
else:
    print("Nenhum label_encoder.pkl encontrado; fallback para indices será usado.")

# ---------- carregar modelos ----------
models_info = []

def mode_from_shape(shape):
    if len(shape) == 2:
        return "frame_vector"
    elif len(shape) == 3:
        return "sequence"
    elif len(shape) == 4:
        return "image"
    else:
        return None

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo principal não encontrado em: {MODEL_PATH}")
print("Carregando modelo principal:", MODEL_PATH)
model_main = tf.keras.models.load_model(MODEL_PATH)
shape_main = model_main.input_shape
mode_main = mode_from_shape(shape_main)
if mode_main is None:
    raise ValueError(f"Formato de input do modelo principal não suportado: {shape_main}")
models_info.append({
    "name": "main",
    "path": MODEL_PATH,
    "model": model_main,
    "mode": mode_main,
    "input_shape": shape_main
})
print("Modelo principal mode:", mode_main, "shape:", shape_main)

if os.path.exists(SEQ_MODEL_PATH):
    try:
        print("Carregando modelo sequencial (opcional):", SEQ_MODEL_PATH)
        model_seq = tf.keras.models.load_model(SEQ_MODEL_PATH)
        shape_seq = model_seq.input_shape
        mode_seq = mode_from_shape(shape_seq)
        if mode_seq != "sequence":
            print(f"[WARN] Modelo {SEQ_MODEL_PATH} não parece sequencial (input_shape={shape_seq}). Será usado se compatível.")
        info = {
            "name": "seq",
            "path": SEQ_MODEL_PATH,
            "model": model_seq,
            "mode": mode_seq,
            "input_shape": shape_seq
        }
        if mode_seq == "sequence":
            seq_len = shape_seq[1] if shape_seq[1] is not None else SEQ_WINDOW_LEN
            info["seq_len"] = seq_len
            info["timestep_features"] = shape_seq[2]
            info["window"] = deque(maxlen=seq_len)
            print("Seq model seq_len:", seq_len, "timestep_features:", info["timestep_features"])
        models_info.append(info)
    except Exception as e:
        print("Erro ao carregar seq model:", e)
else:
    print("Nenhum modelo sequencial encontrado em", SEQ_MODEL_PATH)

# ---------- mediapipe ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ---------- utilitários ----------
def extract_landmarks_xyz(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        kps = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        return kps
    else:
        return np.zeros((21,3), dtype=np.float32)

def extract_features_from_flat63(flat63):
    k = np.array(flat63).reshape(-1, 3)
    wrist = k[0].copy()
    k = k - wrist
    features = k.flatten().tolist()
    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            dist = np.linalg.norm(k[fingertips[i]] - k[fingertips[j]])
            features.append(dist)
    abertura = np.linalg.norm(k[4] - k[20])
    features.append(abertura)
    return np.array(features, dtype=np.float32)

def prepare_image_input(frame, target_shape):
    H = target_shape[1]
    W = target_shape[2]
    C = target_shape[3]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    resized = cv2.resize(gray, (W, H))
    normalized = resized.astype(np.float32) / 255.0
    if C == 1:
        return np.expand_dims(normalized, axis=(0, -1))
    elif C == 3:
        stacked = np.stack([normalized, normalized, normalized], axis=-1)
        return np.expand_dims(stacked, axis=0)
    else:
        return np.expand_dims(normalized, axis=(0, -1))

def probs_to_labels(probs):
    N = len(probs)
    if encoder is not None and len(encoder.classes_) == N:
        return list(encoder.inverse_transform(np.arange(N)))
    else:
        return [str(i) for i in range(N)]

def append_log(timestamp, top_labels, top_probs):
    exists = os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "top1", "p1", "top2", "p2", "top3", "p3"])
        row = [timestamp]
        for i in range(3):
            if i < len(top_labels):
                row.append(top_labels[i])
                row.append(f"{top_probs[i]:.4f}")
            else:
                row += ["", ""]
        writer.writerow(row)

# ---------- MAIN LOOP ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam.")

frame_count = 0
prev_time = time.time()
display_text = "Reconhecendo..."
display_color = (0,0,255)
hold_counter = 0

# top-3 pos iniciais (calculados a partir da resolução se necessários)
h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

print("Iniciando webcam. Pressione ESC para sair.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # extrai keypoints
        k_xyz = extract_landmarks_xyz(frame)   # (21,3)
        k_xy = k_xyz[:, :2]
        flat_xyz = k_xyz.flatten()

        votes = {}
        any_prediction = False

        for info in models_info:
            m = info["model"]
            mode = info["mode"]
            try:
                if mode == "frame_vector":
                    feats = extract_features_from_flat63(flat_xyz)
                    expected = info["input_shape"][1]
                    if feats.shape[0] < expected:
                        feats = np.pad(feats, (0, expected - feats.shape[0]), mode='constant')
                    elif feats.shape[0] > expected:
                        feats = feats[:expected]
                    X = feats.reshape(1, -1).astype(np.float32)
                    if frame_count % PRED_EVERY_N_FRAMES != 0:
                        continue
                    probs = m.predict(X, verbose=0)[0]
                    weight = WEIGHT_FRAME

                elif mode == "image":
                    Ximg = prepare_image_input(frame, info["input_shape"])
                    if frame_count % PRED_EVERY_N_FRAMES != 0:
                        continue
                    probs = m.predict(Ximg, verbose=0)[0]
                    weight = WEIGHT_FRAME

                elif mode == "sequence":
                    tsz = info["timestep_features"]

                    # --- garantido: usar XY quando o model espera 42 ---
                    if tsz == 42:
                        vec = k_xy.flatten()
                    elif tsz == 63:
                        vec = flat_xyz
                    else:
                        fallback = k_xy.flatten()
                        if fallback.shape[0] < tsz:
                            vec = np.pad(fallback, (0, tsz - fallback.shape[0]), mode='constant')
                        else:
                            vec = fallback[:tsz]

                    # evita empurrar zeros: usa último vetor válido se existir
                    if np.allclose(vec, 0.0):
                        if len(info["window"]) > 0:
                            vec_to_append = info["window"][-1]
                        else:
                            vec_to_append = vec
                    else:
                        vec_to_append = vec
                    info["window"].append(vec_to_append.astype(np.float32))

                    if len(info["window"]) < info["seq_len"]:
                        continue
                    if frame_count % PRED_EVERY_N_FRAMES != 0:
                        continue

                    Xseq = np.stack(list(info["window"]), axis=0).reshape(1, info["seq_len"], -1).astype(np.float32)
                    probs = m.predict(Xseq, verbose=0)[0]
                    weight = WEIGHT_SEQ

                    # DEBUG print for seq model top3
                    try:
                        labels_dbg = probs_to_labels(probs)
                        top3_idx = probs.argsort()[-3:][::-1]
                        top3 = [(labels_dbg[i], float(probs[i])) for i in top3_idx]
                        print(f"[DEBUG] Modelo {os.path.basename(info.get('path','<model>'))} mode={info.get('mode')} top3={top3}")
                    except Exception:
                        pass

                else:
                    continue

                labels = probs_to_labels(probs)
                if encoder is not None and len(encoder.classes_) != len(probs):
                    print(f"[WARN] Modelo {info.get('path')} output len {len(probs)} != encoder len {len(encoder.classes_)} -> pulando")
                    continue

                for i, p in enumerate(probs):
                    lab = labels[i]
                    votes[lab] = votes.get(lab, 0.0) + float(p) * float(weight)

                any_prediction = True

            except Exception as e:
                print("Erro ao predizer com", info.get("path"), e)
                continue

        # combina votos e normaliza -> atualiza last_top_* somente quando houver predição
        if any_prediction and votes:
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            combined_top = [lab for lab, s in sorted_votes[:3]]
            combined_scores = np.array([s for lab, s in sorted_votes[:3]], dtype=float)
            ssum = combined_scores.sum()
            if ssum > 0:
                combined_probs = combined_scores / ssum
            else:
                combined_probs = combined_scores

            # atualiza último top estável
            last_top_labels = combined_top
            last_top_probs = combined_probs.tolist()

            # exibe top1 se acima do limiar
            if combined_probs[0] >= MIN_CONFIDENCE:
                display_text = f"{combined_top[0]} ({combined_probs[0]:.2f})"
                display_color = (0,255,0)
            else:
                display_text = "Reconhecendo..."
                display_color = (0,0,255)
            hold_counter = HOLD_FRAMES

            # log
            append_log(datetime.utcnow().isoformat(), last_top_labels, last_top_probs)

            # snapshot opcional
            if combined_probs[0] >= SNAPSHOT_THRESH:
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                fname = os.path.join(SNAPSHOT_DIR, f"{ts}_{combined_top[0]}_{combined_probs[0]:.2f}.jpg")
                cv2.imwrite(fname, frame)
                print("Snapshot salvo:", fname)

        # desenha landmarks reduzidos (se habilitado)
        if DRAW_LANDMARKS and np.any(k_xyz):
            h,w = frame.shape[:2]
            pts = (k_xy * np.array([w,h])).astype(int)
            try:
                cv2.circle(frame, tuple(pts[0]), 5, (0,255,0), -1)   # pulso
                cv2.circle(frame, tuple(pts[8]), 5, (0,0,255), -1)   # ponta indicador
                cv2.line(frame, tuple(pts[0]), tuple(pts[8]), (120,120,120), 1)
            except Exception:
                pass

        # hold counter handling
        if hold_counter > 0:
            hold_counter -= 1
        else:
            display_text = "Reconhecendo..."
            display_color = (0,0,255)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        # desenha info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, display_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 2)

        # desenha top-3 (usa last_top_* para evitar flicker)
        if hold_counter > 0 and last_top_labels:
            try:
                for i in range(min(3, len(last_top_labels))):
                    txt = f"{i+1}. {last_top_labels[i]} {last_top_probs[i]:.2f}"
                    cv2.putText(frame, txt, (TOP3_X, TOP3_Y + i * LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            except Exception:
                pass

        cv2.imshow("Predict (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Finalizado. Logs salvos em", LOG_CSV_PATH)




