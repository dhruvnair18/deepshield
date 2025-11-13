# api/model_runner.py
import os, json, subprocess
from PIL import Image
import torch
import numpy as np
from torchvision import transforms


import cv2

# Haar cascade for faces (bundled with OpenCV)
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Optional env toggles
FACE_DETECT      = os.getenv("FACE_DETECT", "true").lower() in ("1","true","yes")
FACE_MARGIN      = float(os.getenv("FACE_MARGIN", "0.25"))   # 25% padding around face
FACE_MIN_SIZE    = int(os.getenv("FACE_MIN_SIZE", "64"))     # min face box size (px)

def _crop_main_face_bgr(bgr, margin=FACE_MARGIN):
    """Return a face-centered crop from a BGR image, or original frame if none found."""
    if not FACE_DETECT or _CASCADE.empty():
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE)
    )
    if len(faces) == 0:
        return bgr
    # largest face
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    H, W = bgr.shape[:2]
    dx, dy = int(w * margin), int(h * margin)
    x0 = max(0, x - dx); y0 = max(0, y - dy)
    x1 = min(W, x + w + dx); y1 = min(H, y + h + dy)
    return bgr[y0:y1, x0:x1]

def _maybe_face_crop_pil(img_pil):
    """Crop face from a PIL RGB image (returns PIL RGB)."""
    if not FACE_DETECT:
        return img_pil
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    crop = _crop_main_face_bgr(bgr)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)




# ImageNet normalization for MobileNetV2
_IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
_TFORM = transforms.Compose([
    transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

def _preprocess_pil(pil):
    return _TFORM(pil).unsqueeze(0)  # [1,3,H,W]

@torch.inference_mode()
def _infer_logits_or_sigmoid(model, batch_tensor, device="cpu"):
    """
    Returns prob_fake in [0,1] for each sample in the batch,
    automatically handling head type: 2-logit softmax OR 1-logit sigmoid.
    Also respects FAKE_CLASS_INDEX meta if provided by model_loader.
    """
    model.eval().to(device)
    out = model(batch_tensor.to(device))

    # handle various head shapes:
    # case A: [N,2] logits -> softmax then take fake_class_index
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.ndim == 2 and out.shape[-1] == 2:
        logits = out
        probs = torch.softmax(logits, dim=-1)  # [N,2]
        fake_idx = int(os.getenv("FAKE_CLASS_INDEX", "1"))
        prob_fake = probs[:, fake_idx]
        return prob_fake.detach().float().cpu().numpy()

    # case B: [N,1] or [N] -> sigmoid gives P(fake)
    if out.ndim == 2 and out.shape[-1] == 1:
        prob_fake = torch.sigmoid(out[:, 0])
        return prob_fake.detach().float().cpu().numpy()
    if out.ndim == 1:
        prob_fake = torch.sigmoid(out)
        return prob_fake.detach().float().cpu().numpy()

    # fallback: try sigmoid over last dim
    prob_fake = torch.sigmoid(out.squeeze())
    if prob_fake.ndim == 0:
        prob_fake = prob_fake[None]
    return prob_fake.detach().float().cpu().numpy()

def _prob_fake_from_pil_with_meta(pil, model, device="cpu", fake_class_index=None, invert=False):
    x = _preprocess_pil(pil)
    if fake_class_index is not None:
        os.environ["FAKE_CLASS_INDEX"] = str(fake_class_index)
    p = _infer_logits_or_sigmoid(model, x, device=device)[0]
    if invert:
        p = 1.0 - float(p)
    return float(p)


FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.5"))


DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1","true","yes")
SAMPLE_FRAMES = int(os.getenv("SAMPLE_FRAMES", "24"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

def _sigmoid(x): 
    return float(1/(1+np.exp(-x)))

# ---------- ffprobe audio check (safe) ----------
def _has_audio(path: str) -> bool:
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error","-show_entries","stream=codec_type",
            "-of","json", path
        ])
        info = json.loads(out)
        return any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    except Exception:
        return False

# ---------- Frame sampler using OpenCV (lazy import) ----------
# def _sample_frames_cv2(path: str, n: int = 24):
#     try:
#         import cv2
#         try:
#             cv2.setNumThreads(0)  # avoid thread issues
#         except Exception:
#             pass
#     except Exception:
#         return []  # cv2 not installed/available

#     cap = cv2.VideoCapture(path)
#     frames = []
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     if length <= 0:
#         # fallback: try to read first n frames sequentially
#         for _ in range(n):
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             frames.append(frame[:, :, ::-1].copy())  # BGR->RGB
#         cap.release()
#         return frames

#     idxs = np.linspace(0, max(length-1, 0), num=min(n, max(1, length)), dtype=int)
#     for i in idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
#         ok, frame = cap.read()
#         if ok:
#             frames.append(frame[:, :, ::-1].copy())  # BGR->RGB
#     cap.release()
#     return frames

def _sample_frames_cv2(path, n=32):
    """Sample n frames uniformly from a video, face-crop each if possible; return list of RGB np.ndarrays."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            # fallback: read sequentially
            frames, step = [], max(1, n)
            i = 0
            while i < n:
                ok, bgr = cap.read()
                if not ok: break
                bgr = _crop_main_face_bgr(bgr)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                # skip ahead roughly
                for _ in range(step-1):
                    cap.grab()
                i += 1
            cap.release()
            return frames

        # uniform indices
        idxs = np.linspace(0, max(0, total - 1), num=n, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok: continue
            bgr = _crop_main_face_bgr(bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        cap.release()
        return frames
    except Exception:
        return []

# ---------- Torch preprocessing (lazy) ----------
def _preprocess_pil_for_torch(pil_img):
    import torch
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    return t(pil_img)

def _batch_infer_pytorch(model, pil_list, device="cpu", batch_size=8, fake_class_index: int = 1):
    import torch
    import torch.nn.functional as F
    xs = [_preprocess_pil_for_torch(p) for p in pil_list]
    xs = torch.stack(xs, dim=0).to(device)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, xs.size(0), batch_size):
            chunk = xs[i:i+batch_size]
            logits = model(chunk)  # [B,1] or [B,2] or [B]
            if logits.dim() == 2 and logits.size(1) == 1:
                p = torch.sigmoid(logits[:,0]).cpu().numpy().tolist()
            elif logits.dim() == 2 and logits.size(1) >= 2:
                p = F.softmax(logits, dim=1)[:, fake_class_index].cpu().numpy().tolist()
            elif logits.dim() == 1:
                p = torch.sigmoid(logits).cpu().numpy().tolist()
            else:
                p = torch.sigmoid(logits).cpu().numpy().tolist()
            probs.extend(p)
    return probs

# ---------- IMAGE ----------
def analyze_image(job_id: str, path: str):
    # Demo stub unchanged
    if DEMO_MODE:
        img = Image.open(path).convert("RGB")
        img = _maybe_face_crop_pil(img)
        w, h = img.size
        s = _sigmoid(((w * h) % 997) / 200.0 - 2.0)
        verdict = "fake" if s >= FAKE_THRESHOLD else "real"
        return {
            "prob_fake": round(s, 4),
            "verdict": verdict,
            "message": "demo: Pillow stub (no weights)"
        }

    from .model_loader import get_model
    kind, model, meta = get_model()

    # Load once
    img = Image.open(path).convert("RGB")
    img = _maybe_face_crop_pil(img)  
    if kind == "pytorch":
        fake_idx = int(meta.get("fake_class_index", 1))
        invert = bool(meta.get("invert", False))

        # Use your batch infer (handles 2-class/1-logit depending on implementation)
        prob = float(
            _batch_infer_pytorch(
                model, [img],
                device=os.getenv("DEVICE", "cpu"),
                batch_size=BATCH_SIZE,
                fake_class_index=fake_idx
            )[0]
        )

        # Allow flipping if the checkpoint’s class mapping is reversed
        if invert:
            prob = 1.0 - prob

        verdict = "fake" if prob >= FAKE_THRESHOLD else "real"
        return {
            "prob_fake": round(prob, 4),
            "verdict": verdict,
            "message": f"pytorch-image (2-class aware, invert={invert}, thr={FAKE_THRESHOLD})"
        }

    elif kind == "keras":
        import tensorflow as tf
        fake_idx = int(meta.get("fake_class_index", 1))
        invert = bool(meta.get("invert", False))

        x = np.asarray(img.resize((224, 224))) / 255.0
        x = np.expand_dims(x, 0)
        raw = model.predict(x)

        if raw.shape[-1] == 1:
            prob = float(raw[0][0])
        else:
            prob = float(raw[0][fake_idx])

        if invert:
            prob = 1.0 - prob

        verdict = "fake" if prob >= FAKE_THRESHOLD else "real"
        return {
            "prob_fake": round(prob, 4),
            "verdict": verdict,
            "message": f"keras-image (invert={invert}, thr={FAKE_THRESHOLD})"
        }


# ---------- VIDEO ----------
def analyze_video(job_id: str, path: str):
    frames = _sample_frames_cv2(path, n=SAMPLE_FRAMES)
    if not frames:
        return {
            "prob_fake": None,
            "verdict": None,
            "message": "Could not read frames (cv2 missing or corrupt video)"
        }

    pil_frames = [Image.fromarray(f) for f in frames]

    # Demo stub unchanged
    if DEMO_MODE:
        probs = [_sigmoid(((p.size[0] * p.size[1]) % 997) / 200.0 - 2.0) for p in pil_frames]
        median_prob = float(np.median(probs)) if probs else None
        verdict = "fake" if (median_prob is not None and median_prob >= FAKE_THRESHOLD) else (
                  "real" if median_prob is not None else None)
        return {
            "prob_fake": (round(median_prob, 4) if median_prob is not None else None),
            "verdict": verdict,
            "message": f"demo: frame-sampled median (samples={len(probs)})"
        }

    from .model_loader import get_model
    kind, model, meta = get_model()

    if kind == "pytorch":
        fake_idx = int(meta.get("fake_class_index", 1))
        invert = bool(meta.get("invert", False))

        probs = _batch_infer_pytorch(
            model, pil_frames,
            device=os.getenv("DEVICE", "cpu"),
            batch_size=BATCH_SIZE,
            fake_class_index=fake_idx
        )

        # Optional inversion to fix class mapping if needed
        if invert:
            probs = [1.0 - float(p) for p in probs]
        else:
            probs = [float(p) for p in probs]

    elif kind == "keras":
        import tensorflow as tf
        fake_idx = int(meta.get("fake_class_index", 1))
        invert = bool(meta.get("invert", False))

        xs = np.stack([np.asarray(p.resize((224, 224))) / 255.0 for p in pil_frames], axis=0)
        raw = model.predict(xs, batch_size=BATCH_SIZE)
        if raw.shape[-1] == 1:
            probs = raw[:, 0].astype(float).tolist()
        else:
            probs = raw[:, fake_idx].astype(float).tolist()

        if invert:
            probs = [1.0 - float(p) for p in probs]

    else:
        return {"prob_fake": None, "verdict": None, "message": "unknown model type"}

    # ----- Robust aggregation: trimmed mean (10%) with median as reference -----
    def _aggregate_probs(arr, trim=0.10):
        if not arr:
            return None
        a = np.sort(np.array(arr, dtype=float))
        n = len(a)
        k = int(n * trim) if n >= 10 else 0
        core = a[k:n - k] if k > 0 else a
        return float(core.mean())

    agg = _aggregate_probs(probs, trim=0.10)
    med = float(np.median(probs)) if probs else None

    verdict = None
    if agg is not None:
        verdict = "fake" if agg >= FAKE_THRESHOLD else "real"

    return {
        "prob_fake": (round(agg, 4) if agg is not None else None),
        "verdict": verdict,
        "message": f"samples={len(probs)} frames, agg=trimmed_mean@10% (med={med:.4f})"
                   if med is not None else f"samples={len(probs)} frames, agg=trimmed_mean@10%"
    }

