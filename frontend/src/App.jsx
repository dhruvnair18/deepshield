import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const MAX_MB = Number(import.meta.env.VITE_MAX_MB || 200);
const MAX_BYTES = MAX_MB * 1024 * 1024;

function Spinner() {
  return (
    <div
      style={{
        display: "inline-block",
        width: 16,
        height: 16,
        border: "2px solid rgba(255,255,255,0.15)",
        borderTopColor: "#ffffff",
        borderRadius: "50%",
        animation: "spin 0.9s linear infinite",
      }}
    />
  );
}
const injectKeyframes = () => {
  const id = "spin-keyframes";
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
  style.id = id;
  style.textContent =
    `@keyframes spin { from { transform: rotate(0deg);} to {transform: rotate(360deg);} }`;
  document.head.appendChild(style);
};
injectKeyframes();

export default function App() {
  const [kind, setKind] = useState("video");
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [explainUrl, setExplainUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [recent, setRecent] = useState([]);              // <-- moved INSIDE component
  const pollRef = useRef(null);

  const accept = useMemo(
    () => (kind === "image" ? "image/*" : "video/*"),
    [kind]
  );

  const sampledFrames = useMemo(() => {
    if (!job?.message) return null;
    const m = /samples=(\d+)/.exec(job.message);
    return m ? Number(m[1]) : null;
  }, [job]);

  const probPct = useMemo(() => {
    if (job?.prob_fake == null) return null;
    return Math.round(Number(job.prob_fake) * 1000) / 10;
  }, [job]);

  const verdictColor = useMemo(() => {
    if (!job?.verdict) return "#94a3b8";
    return job.verdict === "fake" ? "#ef4444" : "#22c55e";
  }, [job]);

  const progressBar = (p) => (
    <div style={{ background: "rgba(255,255,255,0.08)", borderRadius: 8, height: 10 }}>
      <div
        style={{
          width: `${Math.max(0, Math.min(100, p))}%`,
          height: "100%",
          borderRadius: 8,
          background: p >= 50 ? "#ef4444" : "#22c55e",
          boxShadow: "0 0 10px rgba(34,197,94,0.25)",
        }}
      />
    </div>
  );

  const reset = () => {
    setError(null);
    setExplainUrl(null);
    setJob(null);
    setStatus(null);
    setFile(null);
  };

  const onFileChange = (e) => {
    setError(null);
    const f = e.target.files?.[0] || null;
    if (!f) {
      setFile(null);
      return;
    }
    if (f.size > MAX_BYTES) {
      setFile(null);
      setError(`File too large. Limit is ${MAX_MB} MB.`);
      return;
    }
    if (kind === "image" && !f.type.startsWith("image/")) {
      setFile(null);
      setError("Please choose an image file.");
      return;
    }
    if (kind === "video" && !f.type.startsWith("video/")) {
      setFile(null);
      setError("Please choose a video file.");
      return;
    }
    setFile(f);
  };

  const upload = async () => {
    setError(null);
    setExplainUrl(null);
    if (!file) return;
    try {
      setUploading(true);
      const fd = new FormData();
      fd.append("kind", kind);
      fd.append("file", file);
      const res = await fetch(`${API_BASE}/api/jobs`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Upload failed");
      setJob(data);
      setStatus("queued");
      poll(data.job_id);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setUploading(false);
    }
  };

  const poll = (jobId) => {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        const s = await r.json();
        setJob(s);
        if (s.status === "finished" || s.status === "failed") {
          clearInterval(pollRef.current);
          setStatus(s.status);
        }
      } catch (e) {
        clearInterval(pollRef.current);
        setError("Lost connection while polling. Please try again.");
      }
    }, 800);
  };

  const getExplain = async () => {
    if (!job?.job_id) return;
    const url = `${API_BASE}/api/jobs/${job.job_id}/explain`;
    try {
      const r = await fetch(url);
      if (r.status === 200) {
        setExplainUrl(url + `?t=${Date.now()}`);
      } else {
        const msg = await r.text();
        setExplainUrl(null);
        setError(
          r.status === 501
            ? "Explain is disabled in demo mode (no model weights)."
            : `Explain failed (${r.status}): ${msg}`
        );
      }
    } catch {
      setError("Explain request failed. Is the API running?");
    }
  };

  // ---- Recent list helpers ----
  const loadRecent = async () => {
    try {
      const r = await fetch(`${API_BASE}/api/jobs/recent?limit=10`);
      const data = await r.json();
      if (r.ok) setRecent(data.items || []);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadRecent();
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (job?.status === "finished" || job?.status === "failed") {
      loadRecent();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status]);

  // ---------- UI ----------
  return (
    <div
  style={{
    minHeight: "100vh",
    width: "100vw",
    display: "flex",
    justifyContent: "center",
    background: "radial-gradient(circle at top, #0f172a 0%, #0b1220 60%, #090d1a 100%)",
    color: "#e5e7eb",
    overflowY: "auto",
    padding: "40px 24px", // more top padding
    boxSizing: "border-box",
  }}
>

      <div
        style={{
          width: "100%",
          maxWidth: 780,
          background: "rgba(17, 24, 39, 0.75)",
          backdropFilter: "blur(6px)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 18,
          boxShadow: "0 10px 40px rgba(0,0,0,0.45)",
          padding: 20,
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 12,
          }}
        >
          <div>
            <h1 style={{ margin: 0, fontSize: 26, letterSpacing: 0.2 }}>
              Deepfake Detector
            </h1>
            <div style={{ fontSize: 12.5, color: "#94a3b8", marginTop: 4 }}>
              FastAPI • Redis • RQ • EfficientNet-B4 • React
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <span
              style={{
                padding: "6px 10px",
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(253, 230, 138, 0.15)",
                color: "#fde68a",
                borderRadius: 999,
                fontSize: 12.5,
              }}
            >
              Review build
            </span>
            <span
              style={{
                padding: "6px 10px",
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(99, 102, 241, 0.15)",
                color: "#c7d2fe",
                borderRadius: 999,
                fontSize: 12.5,
              }}
            >
              {kind === "image" ? "Image" : "Video"} mode
            </span>
          </div>
        </div>

        {/* Upload Card */}
        <div
          style={{
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 14,
            background: "rgba(2,6,23,0.45)",
            padding: 16,
          }}
        >
          <div style={{ display: "grid", gap: 12 }}>
            <label style={{ fontSize: 14, color: "#cbd5e1" }}>
              Type:&nbsp;
              <select
                value={kind}
                onChange={(e) => {
                  setKind(e.target.value);
                  setFile(null);
                  setError(null);
                }}
                style={{
                  background: "#0b1220",
                  color: "#e5e7eb",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 8,
                  padding: "6px 10px",
                }}
              >
                <option value="image">Image</option>
                <option value="video">Video</option>
              </select>
            </label>

            <div>
              <input
                type="file"
                accept={accept}
                onChange={onFileChange}
                style={{
                  width: "100%",
                  background: "#0b1220",
                  color: "#e5e7eb",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 10,
                  padding: "10px 12px",
                }}
              />
              <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 6 }}>
                Max size: {MAX_MB} MB
              </div>
            </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <button
                onClick={upload}
                disabled={!file || uploading}
                style={{
                  padding: "10px 14px",
                  borderRadius: 10,
                  border: "1px solid rgba(59,130,246,0.35)",
                  background:
                    !file || uploading
                      ? "rgba(59,130,246,0.25)"
                      : "rgba(59,130,246,0.45)",
                  color: "#eaf2ff",
                  cursor: !file || uploading ? "not-allowed" : "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                {uploading ? <Spinner /> : null}
                {uploading ? "Uploading…" : "Analyze"}
              </button>

              <button
                onClick={reset}
                style={{
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.12)",
                  background: "rgba(255,255,255,0.06)",
                  color: "#e5e7eb",
                  cursor: "pointer",
                }}
              >
                Analyze another
              </button>
            </div>

            {error && (
              <div
                style={{
                  marginTop: 4,
                  padding: 12,
                  borderRadius: 10,
                  background: "rgba(239,68,68,0.1)",
                  border: "1px solid rgba(239,68,68,0.35)",
                  color: "#fecaca",
                }}
              >
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Result Card */}
        {job && (
          <div
            style={{
              marginTop: 16,
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 14,
              background: "rgba(2,6,23,0.45)",
              padding: 16,
            }}
          >
            <h3 style={{ marginTop: 0, marginBottom: 8, color: "#e5e7eb" }}>
              Result
            </h3>
            <div
              style={{
                fontSize: 13,
                color: "#94a3b8",
                wordBreak: "break-all",
              }}
            >
              id: {job.job_id}
            </div>

            <div style={{ marginTop: 8 }}>
              Status:&nbsp;
              <span
                style={{
                  padding: "4px 10px",
                  borderRadius: 999,
                  border: "1px solid rgba(255,255,255,0.12)",
                  background: "rgba(255,255,255,0.06)",
                  color: "#e5e7eb",
                }}
              >
                {status || job.status}{" "}
                {job.status === "queued" ? (
                  <>
                    &nbsp;
                    <Spinner />
                  </>
                ) : null}
              </span>
            </div>

            {job.status === "finished" && (
              <>
                <div style={{ marginTop: 14, display: "grid", gap: 12 }}>
                  <div
                    style={{
                      display: "flex",
                      gap: 12,
                      alignItems: "center",
                      flexWrap: "wrap",
                    }}
                  >
                    <span
                      style={{
                        padding: "6px 10px",
                        borderRadius: 999,
                        background: "rgba(255,255,255,0.05)",
                        border: `1px solid ${verdictColor}80`,
                        color: verdictColor,
                        fontWeight: 700,
                        letterSpacing: 0.4,
                      }}
                    >
                      {job.verdict?.toUpperCase()}
                    </span>

                    {probPct != null && (
                      <div style={{ minWidth: 240, flex: 1 }}>
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            fontSize: 12,
                            color: "#94a3b8",
                          }}
                        >
                          <div>Probability(fake)</div>
                          <div>
                            <b style={{ color: "#e5e7eb" }}>{probPct}%</b>
                          </div>
                        </div>
                        {progressBar(probPct)}
                      </div>
                    )}

                    {sampledFrames != null && (
                      <span
                        style={{
                          fontSize: 12,
                          color: "#cbd5e1",
                          padding: "6px 10px",
                          background: "rgba(255,255,255,0.06)",
                          border: "1px solid rgba(255,255,255,0.12)",
                          borderRadius: 999,
                        }}
                      >
                        {kind === "video"
                          ? `Video • sampled ${sampledFrames} frames`
                          : "Image"}
                      </span>
                    )}
                  </div>

                  {job.message && (
                    <div style={{ fontSize: 12, color: "#94a3b8" }}>
                      {job.message}
                    </div>
                  )}

                  <div>
                    <button
                      onClick={getExplain}
                      style={{
                        padding: "8px 12px",
                        borderRadius: 8,
                        border: "1px solid rgba(255,255,255,0.12)",
                        background: "rgba(255,255,255,0.06)",
                        color: "#e5e7eb",
                        cursor: "pointer",
                      }}
                    >
                      View Heatmap (Grad-CAM)
                    </button>
                  </div>

                  {explainUrl && (
                    <div>
                      <img
                        alt="Grad-CAM"
                        src={explainUrl}
                        style={{
                          maxWidth: "100%",
                          borderRadius: 12,
                          border: "1px solid rgba(255,255,255,0.12)",
                        }}
                      />
                    </div>
                  )}
                </div>
              </>
            )}

            {job.status === "failed" && (
              <div
                style={{
                  marginTop: 12,
                  padding: 12,
                  borderRadius: 10,
                  background: "rgba(239,68,68,0.08)",
                  border: "1px solid rgba(239,68,68,0.3)",
                  color: "#fecaca",
                }}
              >
                <div style={{ fontWeight: 700, marginBottom: 6 }}>
                  Analysis failed
                </div>
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    margin: 0,
                    fontSize: 12,
                    color: "#fca5a5",
                  }}
                >
                  {(job.message || "").slice(0, 1200) || "No error message"}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Recent analyses */}
        <button
  onClick={() => {
    const url = `${API_BASE}/api/jobs/export.csv?limit=100`;
    const a = document.createElement("a");
    a.href = url;
    a.download = "";      // use server filename
    document.body.appendChild(a);
    a.click();
    a.remove();
  }}
  style={{
    padding: "6px 10px",
    borderRadius: 8,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(59,130,246,0.25)",
    color: "#eaf2ff",
    cursor: "pointer",
  }}
>
  Download report (CSV)
</button>

        <div
          style={{
            marginTop: 16,
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 14,
            background: "rgba(2,6,23,0.45)",
            padding: 16,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <h3 style={{ margin: 0, color: "#e5e7eb" }}>Recent analyses</h3>
            <button
              onClick={() => loadRecent()}
              style={{
                padding: "6px 10px",
                borderRadius: 8,
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.06)",
                color: "#e5e7eb",
                cursor: "pointer",
              }}
            >
              Refresh
            </button>
          </div>

          <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
            {recent.length === 0 && (
              <div style={{ fontSize: 12, color: "#94a3b8" }}>No jobs yet.</div>
            )}
            {recent.map((r) => (
              <div
                key={r.job_id}
                onClick={async () => {
                  try {
                    const res = await fetch(`${API_BASE}/api/jobs/${r.job_id}`);
                    const data = await res.json();
                    setJob(data);
                    setStatus(data.status);
                    setError(null);
                    setExplainUrl(null);
                  } catch {
                    // ignore
                  }
                }}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto",
                  gap: 8,
                  padding: 10,
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.08)",
                  background: "rgba(255,255,255,0.04)",
                  cursor: "pointer",
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 12,
                      color: "#cbd5e1",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {r.job_id}
                  </div>
                  <div style={{ fontSize: 12, color: "#94a3b8" }}>
                    {r.kind} • {r.status}
                    {r.verdict ? ` • ${r.verdict}` : ""}
                  </div>
                </div>
                <div
                  style={{
                    alignSelf: "center",
                    fontSize: 12,
                    color: "#94a3b8",
                    textAlign: "right",
                  }}
                >
                  {r.prob_fake != null ? `${Math.round(r.prob_fake * 100)}%` : "—"}
                </div>
                <div
                  style={{
                    alignSelf: "center",
                    fontSize: 12,
                    color: "#94a3b8",
                    textAlign: "right",
                  }}
                >
                  {new Date(
                    (r.updated_at || r.created_at) * 1000
                  ).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div
          style={{
            marginTop: 18,
            textAlign: "center",
            fontSize: 12,
            color: "#94a3b8",
          }}
        >
          API: <code style={{ color: "#e5e7eb" }}>{API_BASE}</code>
        </div>
      </div>
    </div>
  );
}
