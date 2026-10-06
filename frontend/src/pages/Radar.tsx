import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import type { RadarStatus } from "../api/types";
import { ReviewCard } from "../components/ReviewCard";
import { ReviewDrawer } from "../components/ReviewDrawer";
import { Card, ErrorState, Loading, PageHeader, StatusBadge } from "../components/ui";
import { fmtDate, fmtInt, fmtRatio } from "../lib/format";

const FILTERS: { label: string; statuses: RadarStatus[] | null }[] = [
  { label: "Actionable", statuses: ["NEW", "EMERGING", "STABLE", "DECLINING"] },
  { label: "Emerging only", statuses: ["NEW", "EMERGING"] },
  { label: "All", statuses: null },
];

/** Full status list with thresholds per status — not the separate param cards. */
function classificationRulesText(params: Record<string, number> | null | undefined): string {
  const mentions = params?.effective_min_current_mentions ?? params?.min_current_mentions ?? 30;
  const evidence = params?.effective_min_evidence_reviews ?? params?.min_evidence_reviews ?? 3;
  const neg = params?.min_negative_ratio ?? 0.5;
  const grow = params?.min_growth_pct ?? 50;
  const decline = params?.decline_pct ?? 25;
  const soft = params?.soft_decline_pct ?? 15;
  const negPct = Math.round(Number(neg) * 100);
  return [
    "evaluated in this order",
    "  NO_DATA                current == 0 and previous == 0",
    `  NOT_A_COMPLAINT        negative_ratio (current, or whole batch if current is empty) < ${neg} (${negPct}%)`,
    `  INSUFFICIENT_EVIDENCE  current < ${mentions}  or  negative evidence reviews < ${evidence}`,
    `  NEW                    previous == 0 and current >= ${mentions}`,
    `  EMERGING               growth_pct >= ${grow}%`,
    `  DECLINING              growth_pct <= -${decline}%`,
    `                         OR (growth_pct <= -${soft}% AND weekly trend is falling AND previous >= ${mentions})`,
    "  STABLE                 otherwise",
  ].join("\n");
}

export function Radar() {
  const { data, error, loading, reload } = useApi(() => api.issues());
  const [params, setParams] = useSearchParams();
  const [filter, setFilter] = useState(0);
  const selected = params.get("issue");

  if (loading) return <Loading />;
  if (error || !data) return <ErrorState message={error ?? "No data"} onRetry={reload} />;

  if (data.dates_available === false || data.status === "unavailable") {
    return (
      <div className="page">
        <PageHeader title="Complaint Radar" description="Growing complaints need review timestamps." />
        <div className="banner" role="note">
          {data.message || "This view needs review dates; none were found in this dataset."}
        </div>
      </div>
    );
  }

  const wanted = FILTERS[filter].statuses;
  const issues = data.issues.filter((i) => !wanted || wanted.includes(i.status));
  const w = data.window;

  return (
    <div className="page">
      <PageHeader
        title="Complaint Radar"
        description={
          w ? (
            <>
              Current window {fmtDate(w.current_start)} → {fmtDate(w.end)} ({fmtInt(w.current_reviews)} reviews) vs previous {fmtDate(w.previous_start)} →{" "}
              {fmtDate(w.current_start)} ({fmtInt(w.previous_reviews)} reviews).
            </>
          ) : (
            "Compares the latest window with the previous one."
          )
        }
      />
      <div className="toolbar">
        {FILTERS.map((f, i) => (
          <button key={f.label} className={`btn btn-small ${filter === i ? "btn-primary" : ""}`} onClick={() => setFilter(i)}>
            {f.label}
          </button>
        ))}
      </div>
      <div className={selected ? "split" : ""}>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Complaint</th>
                <th>Status</th>
                <th className="num">Previous</th>
                <th className="num">Current</th>
                <th className="num">Growth</th>
                <th className="num">% neg</th>
                <th className="num">Priority</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {issues.map((i) => (
                <tr key={i.theme_id} className={selected === i.theme_id ? "selected" : ""}>
                  <td className="theme-name">{i.name}</td>
                  <td>
                    <StatusBadge status={i.status} />
                  </td>
                  <td className="num">{fmtInt(i.previous_mentions)}</td>
                  <td className="num">{fmtInt(i.current_mentions)}</td>
                  <td className="num">{i.growth_label}</td>
                  <td className="num">{fmtRatio(i.negative_ratio, 0)}</td>
                  <td className="num">{i.priority.toFixed(1)}</td>
                  <td>
                    <button className="btn btn-small btn-primary" onClick={() => setParams({ issue: i.theme_id })}>
                      View why
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {selected && <WhyPanel themeId={selected} onClose={() => setParams({})} />}
      </div>
      {(data.rules || data.params) && (
        <Card title="Classification rules" subtitle="Evaluated in order; the first matching rule sets the status.">
          <pre className="pre">{classificationRulesText(data.params)}</pre>
        </Card>
      )}
    </div>
  );
}

function WhyPanel({ themeId, onClose }: { themeId: string; onClose: () => void }) {
  const issue = useApi(() => api.issue(themeId), [themeId]);
  const evidence = useApi(() => api.evidence(themeId), [themeId]);
  const [openReview, setOpenReview] = useState<string | null>(null);
  const d = issue.data;

  const n = d?.weekly_counts.length ?? 0;
  const weekly = (d?.weekly_counts ?? []).map((v, i) => ({ week: `W${i + 1}`, n: v, inWindow: i >= n - 2 }));
  const radarEvidence = (evidence.data?.evidence ?? []).filter((e) => e.evidence_kind === "radar_evidence");

  return (
    <Card
      className="detail"
      title={
        <>
          Why is “{d?.name ?? themeId}” {d ? <StatusBadge status={d.status} /> : null}?
        </>
      }
      actions={
        <button className="btn btn-small" onClick={onClose} aria-label="Close explanation">
          ✕
        </button>
      }
    >
      {issue.loading && <Loading />}
      {issue.error && <ErrorState message={issue.error} onRetry={issue.reload} />}
      {d && (
        <div data-testid="why-panel">
          <h3 className="section-title">Reasons</h3>
          <ul className="reasons">
            {d.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
          <h3 className="section-title">Calculation</h3>
          <dl className="kv">
            <dt>Previous window</dt>
            <dd>
              {fmtDate(d.calculation.window.previous[0])} → {fmtDate(d.calculation.window.previous[1])}: <strong>{d.previous_mentions}</strong> mentions
            </dd>
            <dt>Current window</dt>
            <dd>
              {fmtDate(d.calculation.window.current[0])} → {fmtDate(d.calculation.window.current[1])}: <strong>{d.current_mentions}</strong> mentions
            </dd>
            <dt>Growth</dt>
            <dd className="mono">{d.calculation.growth}</dd>
            <dt>Negative share</dt>
            <dd className="mono">{d.calculation.negative_ratio}</dd>
            <dt>Priority</dt>
            <dd className="mono">{d.calculation.priority}</dd>
            <dt>Formula</dt>
            <dd className="mono small">{d.formula}</dd>
            <dt>Thresholds</dt>
            <dd className="small">
              {Object.entries(d.calculation.thresholds)
                .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`)
                .join(" · ")}
            </dd>
          </dl>
          <h3 className="section-title">Weekly mentions (last two weeks = current window)</h3>
          <div className="chart-xs">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weekly}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="week" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="n" name="mentions">
                  {weekly.map((w) => (
                    <Cell key={w.week} fill={w.inWindow ? "#dc2626" : "#a5b4fc"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <h3 className="section-title">Segment associations</h3>
          {d.associations.length === 0 ? (
            <p className="muted small">No app-version or platform segment is over-represented (lift ≥ 1.3, support ≥ 20) in the current window.</p>
          ) : (
            <ul className="reasons">
              {d.associations.map((a) => (
                <li key={`${a.segment}-${a.value}`}>{a.statement}</li>
              ))}
            </ul>
          )}
          <h3 className="section-title">Evidence: current-window negative reviews</h3>
          {evidence.loading && <Loading />}
          {evidence.error && <ErrorState message={evidence.error} onRetry={evidence.reload} />}
          {radarEvidence.map((r) => (
            <ReviewCard key={r.review_id} review={r} onOpen={setOpenReview} tag={`evidence #${r.rank + 1}`} />
          ))}
          {evidence.data && radarEvidence.length === 0 && <p className="muted small">No radar evidence stored for this theme.</p>}
          <p className="muted small">
            These figures describe how often the complaint appears; they are an emerging pattern, not proof of a cause.{" "}
            <Link to={`/evidence?theme=${themeId}`}>All reviews in this theme</Link>
          </p>
        </div>
      )}
      {openReview && <ReviewDrawer reviewId={openReview} onClose={() => setOpenReview(null)} />}
    </Card>
  );
}
