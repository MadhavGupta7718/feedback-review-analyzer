import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type BatchInfo, type BatchJob } from "../api/client";
import { useApi } from "../api/useApi";
import { Card, ErrorState, Loading, PageHeader } from "../components/ui";
import { fmtDate, fmtInt } from "../lib/format";

export function Batches() {
  const nav = useNavigate();
  const list = useApi(() => api.batches());
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<BatchJob | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const poll = useCallback(async (jobId: string) => {
    for (let i = 0; i < 600; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      const j = await api.batchJob(jobId);
      setJob(j);
      if (j.status === "done" || j.status === "error") return j;
    }
    throw new Error("Upload timed out");
  }, []);

  async function onFile(file: File | null) {
    if (!file) return;
    setErr(null);
    setBusy(true);
    setJob(null);
    try {
      const started = await api.uploadBatch(file);
      setJob(started);
      const done = await poll(started.job_id);
      if (done.status === "error") throw new Error(done.error || "Pipeline failed");
      await list.reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function view(b: BatchInfo) {
    setErr(null);
    try {
      await api.activateBatch(b.batch_id);
      nav("/");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  if (list.loading) return <Loading />;
  if (list.error || !list.data) return <ErrorState message={list.error ?? "No data"} onRetry={list.reload} />;

  const batches = list.data.batches;

  return (
    <div className="page">
      <PageHeader
        title="Upload & batches"
        description="Upload a review CSV. Each file becomes its own analytics database. Accuracy lives on Model Validation (Amazon holdout), not on uploads."
      />

      <Card title="Upload CSV" subtitle="Needs a text/review column. Dates optional — without dates, Radar and Drift stay unavailable.">
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          hidden
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />
        <button className="btn btn-primary" disabled={busy} type="button" onClick={() => inputRef.current?.click()}>
          {busy ? "Running pipeline…" : "Choose CSV file"}
        </button>
        {job && (
          <p className="muted small top-gap">
            Job {job.job_id}: <strong>{job.status}</strong>
            {job.filename ? ` · ${job.filename}` : ""}
            {job.error ? ` · ${job.error}` : ""}
          </p>
        )}
        {err && <p className="banner banner-bad top-gap">{err}</p>}
      </Card>

      <Card title="Your batches" subtitle={`${batches.length} upload(s)`}>
        {batches.length === 0 ? (
          <p className="muted">No uploads yet. Add a CSV to analyse themes, sentiment and evidence.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th className="num">Reviews</th>
                <th>Dates</th>
                <th>Created</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {batches.map((b) => (
                <tr key={b.batch_id}>
                  <td>
                    <div className="theme-name">{b.name || b.filename}</div>
                    <div className="muted small mono">{b.batch_id}</div>
                    {b.active && <span className="badge">active</span>}
                  </td>
                  <td className="num">{b.reviews != null ? fmtInt(b.reviews) : "–"}</td>
                  <td>{b.dates_available ? "yes" : "missing"}</td>
                  <td className="muted small">{fmtDate(b.created_at_utc)}</td>
                  <td>
                    <button type="button" className="btn btn-small btn-primary" onClick={() => view(b)}>
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
