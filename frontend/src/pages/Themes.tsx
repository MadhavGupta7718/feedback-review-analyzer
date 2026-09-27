import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import type { Theme } from "../api/types";
import { ReviewCard } from "../components/ReviewCard";
import { ReviewDrawer } from "../components/ReviewDrawer";
import { Card, Empty, ErrorState, Loading, PageHeader, SentimentBar, StatusBadge } from "../components/ui";
import { fmtInt, fmtNum, fmtPct } from "../lib/format";

type SortKey = "size" | "negative_pct" | "priority" | "name";

export function Themes() {
  const { data, error, loading, reload } = useApi(() => api.themes());
  const [params, setParams] = useSearchParams();
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortKey>("size");
  const [complaintsOnly, setComplaintsOnly] = useState(false);
  const selected = params.get("theme");

  const themes = useMemo(() => {
    let list: Theme[] = data?.themes ?? [];
    if (complaintsOnly) list = list.filter((t) => t.is_complaint);
    if (q.trim()) {
      const ql = q.toLowerCase();
      list = list.filter((t) => t.name.toLowerCase().includes(ql) || t.keywords.some((k) => k.includes(ql)));
    }
    return [...list].sort((a, b) => (sort === "name" ? a.name.localeCompare(b.name) : (b[sort] ?? 0) - (a[sort] ?? 0)));
  }, [data, q, sort, complaintsOnly]);

  if (loading) return <Loading />;
  if (error || !data) return <ErrorState message={error ?? "No data"} onRetry={reload} />;

  return (
    <div className="page">
      <PageHeader title="Themes" description={`${data.count} themes discovered by clustering sentence embeddings. Every theme links to real, redacted reviews.`} />
      <div className="toolbar">
        <input className="input" placeholder="Search theme or keyword…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search themes" />
        <select className="input" value={sort} onChange={(e) => setSort(e.target.value as SortKey)} aria-label="Sort themes">
          <option value="size">Sort: size</option>
          <option value="negative_pct">Sort: % negative</option>
          <option value="priority">Sort: priority</option>
          <option value="name">Sort: name</option>
        </select>
        <label className="check">
          <input type="checkbox" checked={complaintsOnly} onChange={(e) => setComplaintsOnly(e.target.checked)} /> complaints only
        </label>
      </div>
      <div className={selected ? "split" : ""}>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Theme</th>
                <th className="num">Reviews</th>
                <th>Sentiment</th>
                <th className="num">% neg</th>
                <th>Radar</th>
              </tr>
            </thead>
            <tbody>
              {themes.map((t) => (
                <tr
                  key={t.theme_id}
                  className={`clickable ${selected === t.theme_id ? "selected" : ""}`}
                  onClick={() => setParams({ theme: t.theme_id })}
                >
                  <td>
                    <div className="theme-name">{t.name}</div>
                    <div className="muted small">{t.keywords.slice(0, 5).join(", ")}</div>
                  </td>
                  <td className="num">{fmtInt(t.size)}</td>
                  <td style={{ minWidth: 120 }}>
                    <SentimentBar negative={t.negative_count} neutral={t.neutral_count} positive={t.positive_count} />
                  </td>
                  <td className="num">{fmtPct(t.negative_pct)}</td>
                  <td>
                    <StatusBadge status={t.radar_status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {themes.length === 0 && <Empty>No theme matches.</Empty>}
        </div>
        {selected && <ThemeDetailPanel themeId={selected} onClose={() => setParams({})} />}
      </div>
    </div>
  );
}

function ThemeDetailPanel({ themeId, onClose }: { themeId: string; onClose: () => void }) {
  const { data, error, loading, reload } = useApi(() => api.theme(themeId), [themeId]);
  const [openReview, setOpenReview] = useState<string | null>(null);

  return (
    <Card
      className="detail"
      title={data?.name ?? themeId}
      subtitle={data ? `${fmtInt(data.size)} reviews · coherence ${fmtNum(data.coherence, 2)} · ${themeId}` : themeId}
      actions={
        <button className="btn btn-small" onClick={onClose} aria-label="Close theme">
          ✕
        </button>
      }
    >
      {loading && <Loading />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {data && (
        <>
          <div className="chips">
            {data.keywords.map((k) => (
              <span key={k} className="chip">
                {k}
              </span>
            ))}
          </div>
          <h3 className="section-title">Weekly mentions</h3>
          <div className="chart-xs">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.weekly_counts.map((n, i) => ({ week: `W${i + 1}`, n }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="week" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="n" stroke="#4f46e5" strokeWidth={2} dot={false} name="mentions" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {data.segments.app_version.length > 0 && (
            <>
              <h3 className="section-title">By app version</h3>
              <div className="chart-xs">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.segments.app_version}>
                    <XAxis dataKey="value" tick={{ fontSize: 11 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="n" fill="#6366f1" name="reviews" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
          {data.radar && data.is_complaint && (
            <p className="small">
              Radar: <StatusBadge status={data.radar.status} /> {data.radar.previous_mentions} → {data.radar.current_mentions} mentions ({data.radar.growth_label}){" "}
              <Link to={`/radar?issue=${themeId}`}>View why</Link>
            </p>
          )}
          <h3 className="section-title">Representative reviews</h3>
          <p className="muted small">The most central reviews of this theme (closest to the theme centroid).</p>
          {data.representative_reviews.map((r) => (
            <ReviewCard key={r.review_id} review={r} onOpen={setOpenReview} />
          ))}
          <Link className="btn btn-small" to={`/evidence?theme=${themeId}`}>
            Browse all reviews in this theme
          </Link>
        </>
      )}
      {openReview && <ReviewDrawer reviewId={openReview} onClose={() => setOpenReview(null)} />}
    </Card>
  );
}
