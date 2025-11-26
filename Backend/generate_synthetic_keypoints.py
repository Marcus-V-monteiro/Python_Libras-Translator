# gerar poses sinteticas para letras com movimento (H J K X Z)

# importação de bibliotecas

#Autor: Marcus Monteiro

import os
import numpy as np
import cv2
from scipy.interpolate import interp1d
from typing import List

# ---------- Configuração ----------
OUT_DIR = "synthetic_dataset"
FPS = 25 #frames por segundo para os videos
SEQ_LEN = 32 # numero de frames por sequência
N_SAMPLES_PER_GESTURE = 240 #numero de amostras
N_KEYPOINTS = 21 #número de keypoints da mão
RENDER_SAMPLE_PER_GESTURE = 2
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Funções utilitárias
def lerp(a, b, t):
    return a + (b - a) * t
# Augmentations


def add_noise(seq, sigma=0.012):
    return seq + np.random.normal(scale=sigma, size=seq.shape)

#rotação
def rotate_points_frame(frame, angle_deg):
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    s = frame.reshape(-1, 2)
    s = (s - 0.5).dot(R.T) + 0.5
    return s.reshape(frame.shape)


# escala
def scale_points_frame(frame, scale):
    s = frame.reshape(-1,2)
    s = (s - 0.5) * scale + 0.5
    return s.reshape(frame.shape)


def temporal_warp(seq, factor):
    t_old = np.linspace(0,1,seq.shape[0])
    t_new = np.clip(np.linspace(0,1, max(2, int(round(seq.shape[0] / factor)) )) , 0,1)
    f = interp1d(t_old, seq, axis=0, kind='linear', fill_value="extrapolate")
    out = f(t_new)
    f2 = interp1d(np.linspace(0,1,out.shape[0]), out, axis=0)
    out2 = f2(np.linspace(0,1,seq.shape[0]))
    return out2

def ensure_bounds(seq):
    return np.clip(seq, 0.0, 1.0)

# ---------- BASE POSES (mais detalhadas) ----------
# Nota: Mediapipe indexação real é diferente; aqui uso arranjos aproximados
def palm_open():
    base = np.zeros((N_KEYPOINTS,2))
    xs = np.linspace(0.34, 0.66, N_KEYPOINTS)
    ys = 0.45 + 0.28 * np.sin(np.linspace(-0.6,1.4,N_KEYPOINTS))
    base[:,0] = xs
    base[:,1] = ys
    return base

def fist():
    base = np.zeros((N_KEYPOINTS,2))
    base[:,0] = 0.5 + 0.015*np.sin(np.linspace(0,6,N_KEYPOINTS))
    base[:,1] = 0.57 + 0.015*np.cos(np.linspace(0,6,N_KEYPOINTS))
    return base

def index_extended():
    b = fist().copy()
    # estica índice (simula pontos finais mais altos)
    b[-4:,1] -= 0.20
    b[-4:,0] += 0.02
    return b

def index_middle_extended():
    b = fist().copy()
    # estica índice e médio (do -6 ao -2)
    b[-6:-2,1] -= 0.20
    b[-6:-2,0] += np.linspace(-0.01,0.02,4)
    return b

def thumb_index_opposed():
    b = fist().copy()
    # polegar e índice opostos (útil para K ou X variações)
    b[-1,0] += 0.10
    b[-1,1] -= 0.03
    b[-4,1] -= 0.12
    return b

def hook_x_shape():
    b = fist().copy()
    # dedo dobrado tipo X (forma de gancho)
    b[-4:,1] -= 0.08
    b[-4:,0] += np.array([0.01,0.02,0.02,0.01])
    return b

# Ajustador automático: tweak fino para cada letra usando heurísticas
def auto_tweak_for_letter(base_pose, letter):
    p = base_pose.copy()
    if letter == "H":
        # H: dois dedos estendidos paralelos (índice e médio)
        p = index_middle_extended()
        # variar separação e inclinação: ajustar a coordenada x dos keypoints
        p[..., 0] += np.linspace(-0.01, 0.01, p.shape[0])
    elif letter == "J":
        # J: movimento curvo com dedo indicador (no tempo) -> representado como curve
        p = index_extended()
        # adiciona alongamento lateral progressivo ao longo dos keypoints
        p[..., 0] += np.linspace(0, 0.08, p.shape[0])
    elif letter == "K":
        # K: indicador e médio estendidos, polegar entre eles — simular com thumb oposto
        p = index_middle_extended()
        # combina com polegar ligeiramente deslocado
        thumb = thumb_index_opposed()
        # mistura leve entre as duas poses (50/50)
        p = 0.5 * p + 0.5 * thumb
    elif letter == "X":
        # X: dedo indicador levemente curvado (hook)
        p = hook_x_shape()
    elif letter == "Z":
        # Z: traçado (curva no tempo) - simulamos com deslocamento horizontal ao longo frames
        p = index_extended()
        # adiciona pequeno deslocamento horizontal progressivo
        p[..., 0] += np.linspace(-0.04, 0.04, p.shape[0])
    return ensure_bounds(p)


# Map gestures to keyframes (usamos poses e aplicamos auto_tweak para letras)
GESTURE_KEYFRAMES = {
    "tchau": [palm_open(), palm_open()],
    "ola": [palm_open(), palm_open()],
    "bom_dia": [fist(), palm_open()],
    "boa_tarde": [fist(), palm_open(), index_extended()],
    "tudo_bem": [index_extended(), palm_open()],
    "H": [auto_tweak_for_letter(index_middle_extended(), "H")],
    "J": [auto_tweak_for_letter(index_extended(), "J")],
    "K": [auto_tweak_for_letter(index_middle_extended(), "K")],
    "X": [auto_tweak_for_letter(hook_x_shape(), "X")],
    "Z": [auto_tweak_for_letter(index_extended(), "Z")],
}

# interpolation + generation same as before
def interpolate_keyframes(kfs: List[np.ndarray], seq_len: int):
    if len(kfs) == 0:
        return np.repeat(fist()[np.newaxis,...], seq_len, axis=0)
    parts = len(kfs)-1
    if parts <= 0:
        return np.repeat(kfs[0][np.newaxis,...], seq_len, axis=0)
    frames_per_part = seq_len // parts
    seq = []
    for i in range(parts):
        a = kfs[i]
        b = kfs[i+1]
        for t in range(frames_per_part):
            tt = t / max(1, frames_per_part-1)
            frame = lerp(a, b, tt)
            seq.append(frame)
    seq = seq[:seq_len]
    while len(seq) < seq_len:
        seq.append(kfs[-1])
    return np.stack(seq, axis=0)

def generate_samples_for_gesture(gesture_name, n_samples):
    kfs = GESTURE_KEYFRAMES[gesture_name]
    out_g = os.path.join(OUT_DIR, gesture_name)
    os.makedirs(out_g, exist_ok=True)
    for i in range(n_samples):
        base_seq = interpolate_keyframes(kfs, SEQ_LEN)
        # small random temporal warp for gestures like J and Z (curvo)
        if gesture_name in ["J","Z"] and np.random.rand() < 0.66:
            base_seq = temporal_warp(base_seq, np.random.uniform(0.8,1.2))
        # common augmentations
        if np.random.rand() < 0.7:
            ang = np.random.uniform(-9, 9)
            base_seq = np.stack([rotate_points_frame(f, ang) for f in base_seq], axis=0)
        if np.random.rand() < 0.6:
            s = np.random.uniform(0.93, 1.1)
            base_seq = np.stack([scale_points_frame(f, s) for f in base_seq], axis=0)
        base_seq = add_noise(base_seq, sigma=0.01*np.random.uniform(0.6,1.4))
        # occasional occlusion simulation
        if np.random.rand() < 0.18:
            t0 = np.random.randint(0, SEQ_LEN-3)
            p0 = np.random.randint(0, N_KEYPOINTS-3)
            base_seq[t0:t0+3, p0:p0+2, :] *= np.random.uniform(0.0, 0.45)
        # optional mirror rarely
        if np.random.rand() < 0.06:
            base_seq[...,0] = 1.0 - base_seq[...,0]
        base_seq = ensure_bounds(base_seq)
        fname = os.path.join(out_g, f"{gesture_name}_{i:04d}.npy")
        np.save(fname, base_seq)
    print(f"Saved {n_samples} samples for {gesture_name} in {out_g}")

def render_sequence_to_video(seq, out_path, fps=FPS, size=(512,512)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, size)
    h,w = size
    for f in seq:
        frame = np.ones((h,w,3), dtype=np.uint8)*255
        pts = (f * np.array([w,h])).astype(int)
        for (x,y) in pts:
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)
        for i in range(len(pts)-1):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0,0,0), 1)
        out.write(frame)
    out.release()

if __name__ == "__main__":
    for g in list(GESTURE_KEYFRAMES.keys()):
        generate_samples_for_gesture(g, N_SAMPLES_PER_GESTURE)

    renders_dir = os.path.join(OUT_DIR, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    for g in list(GESTURE_KEYFRAMES.keys()):
        files = sorted([f for f in os.listdir(os.path.join(OUT_DIR,g)) if f.endswith(".npy")])
        for j in range(min(RENDER_SAMPLE_PER_GESTURE, len(files))):
            seq = np.load(os.path.join(OUT_DIR,g, files[j]))
            outp = os.path.join(renders_dir, f"{g}_{j:02d}.mp4")
            render_sequence_to_video(seq, outp)
    print("Generation done. Check", OUT_DIR)
