import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { ReviewCard } from "../components/ReviewCard";
import { ReviewDrawer } from "../components/ReviewDrawer";
import { Empty, ErrorState, Loading, PageHeader } from "../components/ui";
import { fmtInt } from "../lib/format";

const PAGE = 20;

export function Evidence() {
  const [params, setParams] = useSearchParams();
  const theme = params.get("theme") ?? "";
  const sentiment = params.get("sentiment") ?? "";
  const q = params.get("q") ?? "";
  const offset = Number(params.get("offset") ?? 0) || 0;
  const [draft, setDraft] = useState(q);
  const [openReview, setOpenReview] = useState<string | null>(null);

  useEffect(() => setDraft(q), [q]);

  const themes = useApi(() => api.themes());
  const reviews = useApi(() => api.reviews({ theme_id: theme, sentiment, q, limit: PAGE, offset }), [theme, sentiment, q, offset]);

  const update = (patch: Record<string, string>) => {
    const next = new URLSearchParams(params);
    Object.entries(patch).forEach(([k, v]) => (v ? next.set(k, v) : next.delete(k)));
    if (!("offset" in patch)) next.delete("offset");
    setParams(next);
  };

  const total = reviews.data?.total ?? 0;

  return (
    <div className="page">
      <PageHeader
        title="Evidence"
        description="Search the redacted verbatims behind every number. Filters combine; click a review ID for its full record."
      />
      <form
        className="toolbar"
        onSubmit={(e) => {
          e.preventDefault();
          update({ q: draft.trim().slice(0, 60) });
        }}
      >
        <select className="input" value={theme} onChange={(e) => update({ theme: e.target.value })} aria-label="Filter by theme">
          <option value="">All themes</option>
          {(themes.data?.themes ?? [])
            .slice()
            .sort((a, b) => a.name.localeCompare(b.name))
            .map((t) => (
              <option key={t.theme_id} value={t.theme_id}>
                {t.name} ({t.size})
              </option>
            ))}
        </select>
        <select className="input" value={sentiment} onChange={(e) => update({ sentiment: e.target.value })} aria-label="Filter by sentiment">
          <option value="">All sentiment</option>
          <option value="negative">Negative</option>
          <option value="neutral">Neutral</option>
          <option value="positive">Positive</option>
        </select>
        <input className="input grow" placeholder="Search text…" value={draft} maxLength={60} onChange={(e) => setDraft(e.target.value)} aria-label="Search review text" />
        <button className="btn btn-primary" type="submit">
          Search
        </button>
      </form>

      {reviews.loading && <Loading />}
      {reviews.error && <ErrorState message={reviews.error} onRetry={reviews.reload} />}
      {reviews.data && (
        <>
          <p className="muted small">
            {fmtInt(total)} matching reviews{total > 0 && ` · showing ${offset + 1}–${Math.min(offset + PAGE, total)}`}
          </p>
          {reviews.data.reviews.length === 0 ? (
            <Empty>No reviews match these filters.</Empty>
          ) : (
            reviews.data.reviews.map((r) => <ReviewCard key={r.review_id} review={r} onOpen={setOpenReview} tag={r.theme_name ?? "unassigned"} />)
          )}
          <div className="pager">
            <button className="btn btn-small" disabled={offset === 0} onClick={() => update({ offset: String(Math.max(0, offset - PAGE)) })}>
              ← Previous
            </button>
            <button className="btn btn-small" disabled={offset + PAGE >= total} onClick={() => update({ offset: String(offset + PAGE) })}>
              Next →
            </button>
          </div>
        </>
      )}
      {openReview && <ReviewDrawer reviewId={openReview} onClose={() => setOpenReview(null)} />}
    </div>
  );
}
