import cv2, numpy as np, os

def sample_frames(video_path, out_dir, n=8):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total-1), n).astype(int)
    frames = []
    for k, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok: continue
        fp = os.path.join(out_dir, f"frame_{k:02d}.jpg")
        cv2.imwrite(fp, frame)
        frames.append(fp)
    cap.release()
    return frames