import os, time, sys, json, requests

API = os.environ.get("API", "http://127.0.0.1:8000")

def submit(path, kind="video"):
    with open(path, "rb") as f:
        r = requests.post(f"{API}/api/jobs", files={"file": (os.path.basename(path), f)}, data={"kind": kind})
    r.raise_for_status()
    return r.json()["job_id"]

def poll(job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = requests.get(f"{API}/api/jobs/{job_id}").json()
        if s["status"] in ("finished","failed"):
            return s
        time.sleep(0.5)
    raise TimeoutError(job_id)

def eval_dir(root, label):
    ok=0; tot=0
    for fn in os.listdir(root):
        if not fn.lower().endswith((".mp4",".mov",".mkv",".avi")): continue
        tot += 1
        p = os.path.join(root, fn)
        jid = submit(p, "video")
        res = poll(jid)
        pred = res.get("verdict")
        print(f"{fn:30s} -> {pred} (prob_fake={res.get('prob_fake')})")
        if pred == label: ok += 1
    return ok, tot

if __name__ == "__main__":
    sets = [
        ("data/videos/val/real", "real"),
        ("data/videos/val/fake", "fake"),
    ]
    total_ok=0; total=0
    for folder, label in sets:
        if not os.path.isdir(folder): continue
        ok, tot = eval_dir(folder, label)
        total_ok += ok; total += tot
        print(f"[{label}] {ok}/{tot} correct")
    if total>0:
        print(f"OVERALL: {total_ok}/{total} = {total_ok/total:.3f}")
