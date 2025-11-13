from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from redis import Redis
from rq import Queue
import uuid, os
import io, csv, time
from torchvision import transforms
from fastapi import Response
from .schemas import JobStatus
from .jobs import enqueue_analysis, get_job_result
from .db import init_db, upsert_new_job, update_job_row, recent_jobs  # NEW



# --- Config & dirs ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
ART_DIR = os.getenv("ART_DIR", "artifacts")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

# --- App ---
app = FastAPI(title="Deepfake API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# init db
init_db()

# serve artifacts
app.mount("/artifacts", StaticFiles(directory=ART_DIR), name="artifacts")

redis = Redis.from_url(REDIS_URL)
q = Queue("analysis", connection=redis)

@app.get("/api/health")
def health():
    try:
        redis.ping()
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

# ---------- Jobs ----------
@app.post("/api/jobs", response_model=JobStatus)
async def create_job(kind: str = Form(...), file: UploadFile = File(...)):
    if kind not in ("image", "video"):
        raise HTTPException(400, "kind must be image or video")

    # Save upload
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not ext:
        ext = ".mp4" if kind == "video" else ".png"
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(500, f"Failed to save upload: {e}")

    # Log in DB (queued)
    upsert_new_job(job_id, kind, save_path)

    # Enqueue analysis
    rq_job = enqueue_analysis(q, job_id, kind, save_path)
    return JobStatus(job_id=rq_job.id, status="queued")

@app.get("/api/jobs/recent")
def api_recent(limit: int = 20):
    try:
        lim = max(1, min(int(limit), 100))
        items = recent_jobs(lim)
        return {"items": items}
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(e).__name__}: {e}"}
        )

@app.get("/api/jobs/export.csv")
def export_jobs_csv(limit: int = 100):
    """
    Export the most recent jobs as a CSV file (for reporting).
    """
    try:
        lim = max(1, min(int(limit), 1000))
        items = recent_jobs(lim)  # from db.py
        buf = io.StringIO()
        w = csv.writer(buf)
        # header
        w.writerow(["job_id", "kind", "status", "verdict", "prob_fake", "created_at", "updated_at"])
        # rows
        for r in items:
            w.writerow([
                r.get("job_id", ""),
                r.get("kind", ""),
                r.get("status", ""),
                r.get("verdict", "") if r.get("verdict") is not None else "",
                r.get("prob_fake", ""),
                r.get("created_at", ""),
                r.get("updated_at", ""),
            ])
        csv_bytes = buf.getvalue()
        filename = f"deepfake_jobs_{int(time.time())}.csv"
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"CSV export failed: {e}"})

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    # Query current job status from Redis/RQ
    js = get_job_result(redis, job_id)
    # Mirror to DB
    try:
        update_job_row(job_id, js.status, js.verdict, js.prob_fake, js.message)
    except Exception:
        # Do not break API if db write fails
        pass
    return js


# ---------- Explainability (Grad-CAM for images) ----------
@app.get("/api/jobs/{job_id}/explain")
def explain_job(job_id: str):
    """
    Generate & return a Grad-CAM overlay.
    - Image jobs: run directly on the uploaded image.
    - Video jobs: grab a middle frame and run CAM on that frame.
    """
    if DEMO_MODE:
        raise HTTPException(status_code=501, detail="Explain unavailable in demo mode (no model weights)")

    # Find upload path by job_id
    upload_path = None
    for fn in os.listdir(UPLOAD_DIR):
        if fn.startswith(job_id):
            upload_path = os.path.join(UPLOAD_DIR, fn)
            break
    if not upload_path:
        raise HTTPException(404, "Upload not found for this job_id")

    # Lazy imports
    try:
        from PIL import Image
        import cv2
        from .model_loader import get_model
        from .gradcam import gradcam_overlay
    except Exception as e:
        raise HTTPException(500, f"Grad-CAM deps missing: {e}")

    # Load model (we only need the model object)
    kind, model, meta = get_model()
    if kind != "pytorch":
        raise HTTPException(501, "Grad-CAM demo is implemented for PyTorch models only.")

    _, ext = os.path.splitext(upload_path.lower())
    is_video = ext in (".mp4",".mov",".avi",".mkv")

    try:
        if not is_video:
            # IMAGE path
            pil = Image.open(upload_path).convert("RGB")
            overlay = gradcam_overlay(pil, model)
        else:
            # VIDEO path -> sample the middle frame
            cap = cv2.VideoCapture(upload_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total <= 0:
                cap.release()
                raise HTTPException(400, "Could not read frames for Grad-CAM")
            mid = max(0, total // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid))
            ok, bgr = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(400, "Failed to decode frame for Grad-CAM")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            overlay = gradcam_overlay(pil, model)

        out_path = os.path.join(ART_DIR, f"{job_id}_cam.png")
        overlay.save(out_path)
        return FileResponse(out_path, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate Grad-CAM: {e}")


# ---------- Top-K Grad-CAM for VIDEOS ----------
@app.get("/api/jobs/{job_id}/topk-explain")
def topk_explain(job_id: str, k: int = 3):
    if DEMO_MODE:
        return JSONResponse(status_code=501, content={"detail": "Explain disabled in demo mode."})

    upload_path = None
    for fn in os.listdir(UPLOAD_DIR):
        if fn.startswith(job_id):
            upload_path = os.path.join(UPLOAD_DIR, fn)
            break
    if not upload_path:
        raise HTTPException(404, "Upload not found for this job_id")

    _, ext = os.path.splitext(upload_path.lower())
    if ext not in (".mp4", ".mov", ".avi", ".mkv"):
        raise HTTPException(400, "Top-K explain is for video jobs only")

    try:
        from PIL import Image
        import numpy as np
        from .model_loader import get_model
        from .gradcam import gradcam_overlay
        from .model_runner import _sample_frames_cv2, _batch_infer_pytorch
        kind, model, meta = get_model()
        if kind != "pytorch":
            return JSONResponse(status_code=501, content={"detail": "Top-K explain currently supports PyTorch only."})
        fake_idx = int(meta.get("fake_class_index", 1))

        frames = _sample_frames_cv2(upload_path, n=int(os.getenv("SAMPLE_FRAMES", "24")))
        if not frames:
            raise HTTPException(500, "Could not read frames (cv2 missing or corrupt video)")
        pil_frames = [Image.fromarray(f) for f in frames]

        probs = _batch_infer_pytorch(
            model, pil_frames, device=os.getenv("DEVICE", "cpu"),
            batch_size=int(os.getenv("BATCH_SIZE", "8")),
            fake_class_index=fake_idx
        )
        idxs = np.argsort(probs)[::-1][:max(1, int(k))]
        idxs = [int(i) for i in idxs]

        urls = []
        for rank, i in enumerate(idxs, start=1):
            try:
                _, overlay = gradcam_overlay(pil_frames[i], model, fake_class_index=fake_idx)
                out_path = os.path.join(ART_DIR, f"{job_id}_cam_{rank}.png")
                overlay.save(out_path)
                urls.append(f"/artifacts/{os.path.basename(out_path)}")
            except Exception:
                continue

        return {"topk": len(urls), "frames": idxs, "urls": urls}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate Top-K Grad-CAM: {e}")

# api/app.py
@app.get("/api/model")
def model_info():
    return {
        "arch": os.getenv("MODEL_ARCH"),
        "weights": os.getenv("WEIGHTS_PATH"),
        "invert_probs": os.getenv("INVERT_PROBS"),
        "threshold": os.getenv("FAKE_THRESHOLD"),
        "face_detect": os.getenv("FACE_DETECT"),
        "sample_frames": os.getenv("SAMPLE_FRAMES"),
    }
