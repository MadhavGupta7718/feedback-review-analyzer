import type { Review } from "../api/types";
import { fmtDate, fmtRatio } from "../lib/format";
import { SentimentBadge } from "./ui";

export function ReviewCard({ review, onOpen, tag }: { review: Review; onOpen?: (id: string) => void; tag?: string }) {
  return (
    <article className="review" data-testid="review-card">
      <p className="review-text">“{review.text}”</p>
      <div className="review-meta">
        <SentimentBadge sentiment={review.sentiment} />
        <span title="model confidence">{fmtRatio(review.confidence, 0)} conf.</span>
        {review.rating != null && <span>{"★".repeat(review.rating)}{"☆".repeat(Math.max(0, 5 - review.rating))}</span>}
        <span>{fmtDate(review.created_at)}</span>
        {review.platform && <span>{review.platform}</span>}
        {review.app_version && <span>v{review.app_version}</span>}
        {review.pii_redactions > 0 && <span className="pii-tag" title="personal data was redacted">{review.pii_redactions} PII redacted</span>}
        {tag && <span className="tag">{tag}</span>}
        {onOpen ? (
          <button className="link-btn mono" onClick={() => onOpen(review.review_id)} aria-label={`Open review ${review.review_id}`}>
            {review.review_id}
          </button>
        ) : (
          <span className="mono">{review.review_id}</span>
        )}
      </div>
    </article>
  );
}
