import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import type { DriftMetric } from "../api/types";
import { Card, DriftBadge, ErrorState, Kpi, Loading, PageHeader } from "../components/ui";
import { fmtDate, fmtInt, fmtNum, fmtRatio } from "../lib/format";

const DRIFT_LABELS: Record<string, string> = {
  sentiment: "Sentiment distribution",
  theme: "Theme distribution",
  volume: "Review volume",
  review_length: "Review length",
};

function driftDetail(key: string, m: DriftMetric, names: Record<string, string>) {
  if (key === "volume") return `${fmtNum(m.reference_per_day, 1)} → ${fmtNum(m.current_per_day, 1)} reviews/day`;
  if (key === "review_length") return `mean ${fmtNum(m.reference_mean, 1)} → ${fmtNum(m.current_mean, 1)} chars · p=${fmtNum(m.p_value, 4)}`;
  return (m.top_changes ?? [])
    .slice(0, 2)
    .map((c) => `${names[c.category] ?? c.category} ${c.change_pp > 0 ? "+" : ""}${c.change_pp} pp`)
    .join(" · ");
}

export function DataHealth() {
  const health = useApi(() => api.dataHealth());
  const drift = useApi(() => api.drift());
  const models = useApi(() => api.modelInfo());
  const themes = useApi(() => api.themes());

  if (health.loading) return <Loading />;
  if (health.error || !health.data) return <ErrorState message={health.error ?? "No data"} onRetry={health.reload} />;
  const h = health.data;
  const rejected = Object.values(h.rejected).reduce((a, b) => a + b, 0);
  const piiData = Object.entries(h.pii_redactions)
    .map(([type, n]) => ({ type, n }))
    .sort((a, b) => b.n - a.n);
  const themeNames: Record<string, string> = Object.fromEntries((themes.data?.themes ?? []).map((t) => [t.theme_id, t.name]));

  return (
    <div className="page">
      <PageHeader title="Data Health" description="Ingestion quality, personal-data redaction, traceability, drift and model status." />

      <div className="kpi-grid">
        <Kpi label="Input rows" value={fmtInt(h.input_rows)} />
        <Kpi label="Processed" value={fmtInt(h.processed_reviews)} tone="good" />
        <Kpi label="Rejected" value={fmtInt(rejected)} hint={Object.entries(h.rejected).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ") || "none"} />
        <Kpi label="Duplicates removed" value={fmtInt(h.ingestion_duplicates_removed)} hint="same text, time, platform, version" />
        <Kpi label="Encoding repaired" value={fmtInt(h.mojibake_repaired)} hint="mojibake fixed" />
        <Kpi label="Rows with PII" value={fmtInt(h.rows_with_pii)} hint="all redacted before storage" tone="warn" />
      </div>

      <div className="grid-2">
        <Card title="PII redacted by type" subtitle={`Recall on planted synthetic PII: ${fmtRatio(h.pii_audit.overall_recall)} · false-positive rows ${fmtRatio(h.pii_audit.false_positive_row_rate)}`}>
          <div className="chart-sm">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={piiData} layout="vertical" margin={{ left: 30 }}>
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="type" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="n" fill="#7c3aed" name="redactions" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Evidence traceability" subtitle="Automated audit run by the pipeline">
          {h.traceability ? (
            <dl className="kv">
              <dt>Status</dt>
              <dd>
                <span className={`badge ${h.traceability.status === "PASS" ? "badge-drift-none" : "badge-drift-significant"}`}>{h.traceability.status}</span>
              </dd>
              <dt>Evidence links checked</dt>
              <dd>{fmtInt(h.traceability.links_checked)}</dd>
              <dt>Problems</dt>
              <dd>{h.traceability.problems.length}</dd>
              <dt>Residual PII patterns</dt>
              <dd>{h.traceability.reviews_with_residual_pii_pattern}</dd>
            </dl>
          ) : (
            <p className="muted">Not available.</p>
          )}
          <h3 className="section-title">Theme coverage</h3>
          <p className="small">
            {h.theme_stats.n_clusters} themes (merged from {h.theme_stats.n_micro_clusters} clusters). HDBSCAN noise {fmtRatio(h.theme_stats.hdbscan_noise_fraction)};{" "}
            {fmtInt(h.theme_stats.reassigned_by_nearest_centroid)} reassigned to the nearest theme, {fmtInt(h.theme_stats.unassigned)} ({fmtRatio(h.theme_stats.unassigned_fraction)}) left
            unassigned.
          </p>
          <h3 className="section-title">Model accuracy</h3>
          <p className="small">
            {h.dataset_validation?.note ?? "Upload analytics only. See Model Validation for Amazon holdout accuracy."}
          </p>
          {h.dates_available === false && (
            <p className="banner top-gap">{h.dates_message || "Drift needs review dates; none were found in this dataset."}</p>
          )}
        </Card>
      </div>

      <Card
        title="Drift: current vs previous window"
        subtitle={
          drift.data?.current_vs_previous?.reference_window
            ? `${fmtDate(drift.data.current_vs_previous.reference_window[0])} → ${fmtDate(drift.data.current_vs_previous.current_window[1])}`
            : undefined
        }
        actions={drift.data?.current_vs_previous?.overall_status && drift.data.current_vs_previous.overall_status !== "unavailable" ? (
          <DriftBadge status={drift.data.current_vs_previous.overall_status} />
        ) : undefined}
      >
        {drift.loading && <Loading />}
        {drift.error && <ErrorState message={drift.error} onRetry={drift.reload} />}
        {drift.data?.status === "unavailable" || h.dates_available === false ? (
          <p className="muted">{drift.data?.message || h.dates_message || "This view needs review dates; none were found in this dataset."}</p>
        ) : null}
        {drift.data && drift.data.status !== "unavailable" && h.dates_available !== false && (
          <>
            <table className="table">
              <thead>
                <tr>
                  <th>Signal</th>
                  <th>Method</th>
                  <th className="num">Value</th>
                  <th>Status</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(drift.data.current_vs_previous.metrics).map(([k, m]) => (
                  <tr key={k}>
                    <td className="theme-name">{DRIFT_LABELS[k] ?? k}</td>
                    <td className="small">{m.metric}</td>
                    <td className="num">{fmtNum(m.value, 4)}</td>
                    <td>
                      <DriftBadge status={m.status} />
                    </td>
                    <td className="small muted">{driftDetail(k, m, themeNames)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h3 className="section-title">Weekly drift vs baseline (first two weeks)</h3>
            <div className="chart-sm">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={drift.data.weekly_vs_baseline.map((w) => ({ week: w.week_start, sentiment: w.sentiment.value, theme: w.theme.value }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <ReferenceLine y={0.1} stroke="#d97706" strokeDasharray="4 4" label={{ value: "moderate", fontSize: 10, fill: "#d97706" }} />
                  <ReferenceLine y={0.25} stroke="#dc2626" strokeDasharray="4 4" label={{ value: "significant", fontSize: 10, fill: "#dc2626" }} />
                  <Line type="monotone" dataKey="theme" stroke="#4f46e5" name="theme PSI" strokeWidth={2} />
                  <Line type="monotone" dataKey="sentiment" stroke="#16a34a" name="sentiment PSI" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <details>
              <summary className="small">Method and thresholds</summary>
              <pre className="pre">{drift.data.current_vs_previous.method_note}</pre>
            </details>
            <p className="muted small">{drift.data.note}</p>
          </>
        )}
      </Card>

      <Card title="Models" subtitle={models.data ? `Offline verification: ${models.data.verification_status ?? "–"}${models.data.verification_offline ? " (network blocked during test)" : ""}` : undefined}>
        {models.loading && <Loading />}
        {models.error && <ErrorState message={models.error} onRetry={models.reload} />}
        {models.data && (
          <>
            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>State</th>
                  <th className="num">Disk (GB)</th>
                  <th className="num">Load (s)</th>
                  <th>Checks</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(models.data.verification)
                  .filter(([name]) => name.includes("/"))
                  .map(([name, v]) => (
                    <tr key={name}>
                      <td className="mono small">{name}</td>
                      <td>
                        <span className={`badge ${v.state === "LOADED" ? "badge-drift-none" : "badge-drift-significant"}`}>{v.state}</span>
                      </td>
                      <td className="num">{fmtNum(v.disk_gb, 2)}</td>
                      <td className="num">{fmtNum(v.load_seconds, 1)}</td>
                      <td className="small">
                        {v.checks ? `${Object.values(v.checks).filter(Boolean).length}/${Object.keys(v.checks).length} passed` : "–"}
                        {v.plan && ` · ${v.plan.device} ${v.plan.quantization ?? ""}`}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {models.data.hardware_at_pipeline_run && (
              <p className="muted small">
                Pipeline hardware: {String(models.data.hardware_at_pipeline_run.gpu_name ?? "CPU")} · {String(models.data.hardware_at_pipeline_run.gpu_total_vram_gb ?? "–")} GB VRAM ·{" "}
                {String(models.data.hardware_at_pipeline_run.system_ram_total_gb ?? "–")} GB RAM · total {models.data.performance.total_seconds ?? "–"} s for{" "}
                {fmtInt(models.data.performance.n_reviews)} reviews
              </p>
            )}
          </>
        )}
      </Card>
    </div>
  );
}
