import type { ReactNode } from "react";
import type { DriftStatus, RadarStatus, Sentiment } from "../api/types";

export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-header">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="card-subtitle">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Kpi({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: "bad" | "good" | "warn" }) {
  return (
    <div className={`kpi ${tone ? `kpi-${tone}` : ""}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {hint && <div className="kpi-hint">{hint}</div>}
    </div>
  );
}

const STATUS_LABEL: Record<RadarStatus, string> = {
  NEW: "New",
  EMERGING: "Emerging",
  STABLE: "Stable",
  DECLINING: "Declining",
  INSUFFICIENT_EVIDENCE: "Insufficient evidence",
  NOT_A_COMPLAINT: "Not a complaint",
  NO_DATA: "No data",
};

export function StatusBadge({ status }: { status: RadarStatus }) {
  return <span className={`badge badge-${status.toLowerCase()}`}>{STATUS_LABEL[status] ?? status}</span>;
}

export function SentimentBadge({ sentiment }: { sentiment: Sentiment }) {
  return <span className={`badge badge-sent-${sentiment}`}>{sentiment}</span>;
}

export function DriftBadge({ status }: { status: DriftStatus }) {
  if (status === "unavailable") return <span className="badge badge-drift-unavailable">dates required</span>;
  return <span className={`badge badge-drift-${status}`}>{status === "none" ? "no drift" : `${status} drift`}</span>;
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state state-loading" role="status">
      <span className="spinner" aria-hidden /> {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state state-error" role="alert">
      <strong>Could not load data.</strong> <span>{message}</span>
      {onRetry && (
        <button className="btn btn-small" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="state state-empty">{children}</div>;
}

export function SentimentBar({ negative, neutral, positive }: { negative: number; neutral: number; positive: number }) {
  const total = negative + neutral + positive || 1;
  const w = (n: number) => `${(n / total) * 100}%`;
  return (
    <div className="sentbar" title={`negative ${negative} · neutral ${neutral} · positive ${positive}`}>
      <span className="sentbar-neg" style={{ width: w(negative) }} />
      <span className="sentbar-neu" style={{ width: w(neutral) }} />
      <span className="sentbar-pos" style={{ width: w(positive) }} />
    </div>
  );
}

export function PageHeader({ title, description, children }: { title: string; description?: ReactNode; children?: ReactNode }) {
  return (
    <div className="page-header">
      <div>
        <h1>{title}</h1>
        {description && <p className="page-desc">{description}</p>}
      </div>
      {children}
    </div>
  );
}
