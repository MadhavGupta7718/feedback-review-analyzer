import { useEffect } from "react";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { fmtDateTime, fmtNum, fmtRatio } from "../lib/format";
import { ErrorState, Loading, SentimentBadge } from "./ui";

export function ReviewDrawer({ reviewId, onClose }: { reviewId: string; onClose: () => void }) {
  const { data, error, loading, reload } = useApi(() => api.review(reviewId), [reviewId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-label={`Review ${reviewId}`} onClick={(e) => e.stopPropagation()}>
        <header className="drawer-header">
          <h2 className="mono">{reviewId}</h2>
          <button className="btn btn-small" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        {loading && <Loading />}
        {error && <ErrorState message={error} onRetry={reload} />}
        {data && (
          <div className="drawer-body">
            <p className="review-text big">“{data.text}”</p>
            <dl className="kv">
              <dt>Sentiment</dt>
              <dd>
                <SentimentBadge sentiment={data.sentiment} /> {fmtRatio(data.confidence, 1)} confidence
              </dd>
              <dt>Probabilities</dt>
              <dd>
                neg {fmtNum(data.p_negative)} · neu {fmtNum(data.p_neutral)} · pos {fmtNum(data.p_positive)}
              </dd>
              <dt>Theme</dt>
              <dd>
                {data.theme_name ?? "unassigned"} {data.theme_assignment && <span className="muted">({data.theme_assignment})</span>}
                {data.theme_similarity != null && <span className="muted"> · similarity {fmtNum(data.theme_similarity)}</span>}
              </dd>
              <dt>Created</dt>
              <dd>{fmtDateTime(data.created_at)}</dd>
              <dt>Rating</dt>
              <dd>{data.rating ?? "–"}</dd>
              <dt>Platform / version</dt>
              <dd>
                {data.platform ?? "–"} / {data.app_version ?? "–"}
              </dd>
              <dt>PII redactions</dt>
              <dd>{data.pii_redactions}</dd>
              <dt>Source</dt>
              <dd>{data.source}</dd>
              <dt>Used as evidence</dt>
              <dd>
                {data.used_as_evidence.length === 0
                  ? "no"
                  : data.used_as_evidence.map((e) => `${e.theme_id} (${e.kind} #${e.rank + 1})`).join(", ")}
              </dd>
            </dl>
            <p className="muted small">Text shown is the redacted verbatim stored by the pipeline; the raw input is never served.</p>
          </div>
        )}
      </aside>
    </div>
  );
}
