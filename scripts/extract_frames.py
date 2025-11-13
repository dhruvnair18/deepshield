# scripts/extract_frames.py
import os, cv2, numpy as np, argparse, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)   # folder with real/ fake subfolders
parser.add_argument("--dst", required=True)   # output folder with real/ fake subfolders
parser.add_argument("--per_video", type=int, default=12)
args = parser.parse_args()

os.makedirs(args.dst, exist_ok=True)
for cls in ("real","fake"):
    src_cls = os.path.join(args.src, cls); dst_cls = os.path.join(args.dst, cls)
    os.makedirs(dst_cls, exist_ok=True)
    for v in os.listdir(src_cls):
        if not v.lower().endswith((".mp4",".mov",".mkv",".avi")): continue
        path = os.path.join(src_cls, v)
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0: cap.release(); continue
        idxs = np.linspace(0, max(0,total-1), num=args.per_video, dtype=int)
        for i,idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok: continue
            out = os.path.join(dst_cls, f"{pathlib.Path(v).stem}_{i:02d}.jpg")
            cv2.imwrite(out, bgr)
        cap.release()
