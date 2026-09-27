import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { Card, DriftBadge, ErrorState, Kpi, Loading, PageHeader, StatusBadge } from "../components/ui";
import { fmtDate, fmtInt, fmtPct, fmtRatio, SENTIMENT_COLORS } from "../lib/format";

export function Overview() {
  const metrics = useApi(() => api.metrics());
  const themes = useApi(() => api.themes());

  if (metrics.loading) return <Loading />;
  if (metrics.error || !metrics.data) return <ErrorState message={metrics.error ?? "No data"} onRetry={metrics.reload} />;

  const { overview: ov, dataset } = metrics.data;
  const sentimentData = (["negative", "neutral", "positive"] as const).map((k) => ({
    name: k,
    value: ov.sentiment_counts[k],
    pct: ov.sentiment_pct[k],
  }));
  const topComplaints = (themes.data?.themes ?? [])
    .filter((t) => t.is_complaint)
    .sort((a, b) => b.negative_count - a.negative_count)
    .slice(0, 8)
    .map((t) => ({ name: t.name, negative: t.negative_count, theme_id: t.theme_id }));

  return (
    <div className="page">
      <PageHeader
        title="Executive Overview"
        description={
          <>
            {fmtInt(ov.total_reviews)} reviews from {fmtDate(ov.date_range?.[0])} to {fmtDate(ov.date_range?.[1])} · {dataset?.name}
          </>
        }
      />
      {dataset?.kind === "synthetic" && (
        <div className="banner" role="note">
          <strong>Demonstration data.</strong> {dataset.note}
        </div>
      )}

      <div className="kpi-grid">
        <Kpi label="Reviews analysed" value={fmtInt(ov.total_reviews)} hint={`avg rating ${ov.avg_rating ?? "–"}`} />
        <Kpi label="Negative" value={fmtPct(ov.sentiment_pct.negative)} tone="bad" hint={`${fmtInt(ov.sentiment_counts.negative)} reviews`} />
        <Kpi label="Positive" value={fmtPct(ov.sentiment_pct.positive)} tone="good" hint={`${fmtInt(ov.sentiment_counts.positive)} reviews`} />
        <Kpi label="Themes" value={ov.n_themes} hint={`${ov.n_complaint_themes} complaint themes`} />
        <Kpi label="Emerging issues" value={ov.emerging.length} tone={ov.emerging.length ? "warn" : undefined} hint="NEW or EMERGING, last 14 days" />
        <Kpi label="PII redacted" value={fmtInt(ov.pii_redactions_total)} hint="before storage" />
      </div>

      <div className="grid-2">
        <Card title="Emerging complaints" subtitle="Current 14-day window vs the previous 14 days" actions={<Link to="/radar" className="btn btn-small">Open radar</Link>}>
          {ov.emerging.length === 0 ? (
            <p className="muted">No complaint theme meets the emerging thresholds.</p>
          ) : (
            <ul className="issue-list">
              {ov.emerging.map((e) => (
                <li key={e.theme_id} className="issue-row">
                  <div>
                    <div className="issue-name">{e.name}</div>
                    <div className="muted small">
                      {e.previous_mentions} → {e.current_mentions} mentions · {fmtRatio(e.negative_ratio, 0)} negative
                    </div>
                  </div>
                  <div className="issue-right">
                    {e.status !== "NEW" && <span className="growth">{e.growth_label}</span>}
                    <StatusBadge status={e.status} />
                    <Link to={`/radar?issue=${e.theme_id}`} className="btn btn-small btn-primary">
                      View why
                    </Link>
                  </div>
                </li>
              ))}
            </ul>
          )}
          {ov.top_complaint && (
            <p className="muted small top-gap">
              Largest complaint theme by negative volume: <strong>{ov.top_complaint.name}</strong> ({fmtInt(ov.top_complaint.negative_count)} negative of{" "}
              {fmtInt(ov.top_complaint.size)} mentions).
            </p>
          )}
        </Card>

        <Card title="Overall sentiment" subtitle={<>Drift status: <DriftBadge status={ov.drift_status} /></>}>
          <div className="chart-sm">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={sentimentData} dataKey="value" nameKey="name" innerRadius="55%" outerRadius="85%" paddingAngle={2}>
                  {sentimentData.map((d) => (
                    <Cell key={d.name} fill={SENTIMENT_COLORS[d.name]} />
                  ))}
                </Pie>
                <Tooltip formatter={(v, n, p) => [`${fmtInt(Number(v))} (${fmtPct((p.payload as { pct: number }).pct)})`, String(n)]} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="legend">
            {sentimentData.map((d) => (
              <span key={d.name}>
                <i style={{ background: SENTIMENT_COLORS[d.name] }} /> {d.name} {fmtPct(d.pct)}
              </span>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Top complaint themes" subtitle="Negative reviews per complaint theme (all time in this batch)">
        {themes.loading ? (
          <Loading />
        ) : themes.error ? (
          <ErrorState message={themes.error} onRetry={themes.reload} />
        ) : (
          <div className="chart-md">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topComplaints} layout="vertical" margin={{ left: 40, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="negative" fill={SENTIMENT_COLORS.negative} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
      <p className="muted small">
        Generated {metrics.data.generated_at_utc?.slice(0, 19).replace("T", " ")} UTC · pipeline {metrics.data.pipeline_seconds ?? "–"} s on {metrics.data.device ?? "–"}. Numbers
        describe feedback volume and sentiment; they do not establish causes.
      </p>
    </div>
  );
}
