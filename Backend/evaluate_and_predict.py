"""
evaluate_and_predict.py
Avalia o modelo salvo em algumas amostras e mostra matriz de confusão.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import cv2

DATA_DIR = "synthetic_dataset"
MODEL_PATH = "libras_seq_model.h5"
SEQ_LEN = 32

# load model + classes
model = load_model(MODEL_PATH)
classes = np.load("classes.npy", allow_pickle=True)
classes = list(classes)

# load test samples (uma amostra por classe para quick-check)
X = []
y = []
labels = []
for g in classes:
    files = sorted(glob.glob(os.path.join(DATA_DIR,g,"*.npy")))
    if len(files) == 0:
        continue
    # pegar alguns arquivos (até 20)
    sample_files = files[:20]
    for f in sample_files:
        seq = np.load(f)
        if seq.shape[0] < SEQ_LEN:
            pad = np.tile(seq[-1:], (SEQ_LEN - seq.shape[0], 1, 1))
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:SEQ_LEN]
        X.append(seq.reshape(SEQ_LEN, seq.shape[1]*2))
        y.append(g)
        labels.append(os.path.basename(f))
X = np.array(X)
y_true = np.array([classes.index(v) for v in y])

# predict
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_true, y_pred, target_names=classes))

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
plt.yticks(range(len(classes)), classes)
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Confusion Matrix (amostra sintética)")
plt.tight_layout()
plt.show()

# mostra alguns vídeos stick-figure com label + predição
render_dir = os.path.join(DATA_DIR, "renders")
os.makedirs(render_dir, exist_ok=True)
def render_seq_pngs(seq, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    T = seq.shape[0]
    h,w = 400,400
    for t in range(T):
        frame = np.ones((h,w,3), dtype=np.uint8)*255
        pts = (seq[t] * np.array([w,h])).astype(int)
        for (x,y) in pts:
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)
        for i in range(len(pts)-1):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0,0,0), 1)
        cv2.putText(frame, f"{prefix}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{t:03d}.png"), frame)

# salva alguns MP4s
for i in range(min(12, len(X))):
    seq = X[i].reshape(SEQ_LEN, -1)
    K2 = seq.shape[1]
    K = K2 // 2
    seq_pts = seq.reshape(SEQ_LEN, K, 2)
    outp = os.path.join(render_dir, f"check_{i}_{y[i]}_pred_{classes[y_pred[i]]}.mp4")
    # render to video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outp, fourcc, 12, (400,400))
    for t in range(SEQ_LEN):
        frame = np.ones((400,400,3), dtype=np.uint8)*255
        pts = (seq_pts[t] * np.array([400,400])).astype(int)
        for (x,y) in pts:
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)
        for j in range(len(pts)-1):
            cv2.line(frame, tuple(pts[j]), tuple(pts[j+1]), (0,0,0), 1)
        cv2.putText(frame, f"GT:{y[i]} PRED:{classes[y_pred[i]]}", (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        out.write(frame)
    out.release()
    print("Saved check video:", outp)
print("Done. Confira os MP4s em", render_dir)
