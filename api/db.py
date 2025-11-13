# api/db.py
import os, sqlite3, time
from typing import List, Dict, Optional

DB_PATH = os.getenv("DB_PATH", "data/app.db")
os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)

DDL = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id     TEXT PRIMARY KEY,
  kind       TEXT NOT NULL,
  path       TEXT NOT NULL,
  status     TEXT NOT NULL,
  verdict    TEXT,
  prob_fake  REAL,
  message    TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
"""

def _conn():
  return sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)

def init_db():
  with _conn() as cx:
    for stmt in DDL.strip().split(";"):
      if stmt.strip():
        cx.execute(stmt)

def upsert_new_job(job_id: str, kind: str, path: str):
  now = int(time.time())
  with _conn() as cx:
    cx.execute("""
      INSERT INTO jobs(job_id, kind, path, status, created_at, updated_at)
      VALUES(?,?,?,?,?,?)
      ON CONFLICT(job_id) DO UPDATE SET
        kind=excluded.kind, path=excluded.path, updated_at=excluded.updated_at
    """, (job_id, kind, path, "queued", now, now))

def update_job_row(job_id: str, status: str, verdict: Optional[str], prob_fake: Optional[float], message: Optional[str]):
  now = int(time.time())
  with _conn() as cx:
    cx.execute("""
      UPDATE jobs
         SET status=?,
             verdict=?,
             prob_fake=?,
             message=?,
             updated_at=?
       WHERE job_id=?
    """, (status, verdict, prob_fake, message, now, job_id))

def recent_jobs(limit: int = 20) -> List[Dict]:
  with _conn() as cx:
    cur = cx.execute("""
      SELECT job_id, kind, status, verdict, prob_fake, created_at, updated_at
        FROM jobs
    ORDER BY created_at DESC
       LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
  return [
    {
      "job_id": r[0], "kind": r[1], "status": r[2],
      "verdict": r[3], "prob_fake": r[4],
      "created_at": r[5], "updated_at": r[6],
    } for r in rows
  ]
