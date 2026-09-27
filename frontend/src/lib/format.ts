export const fmtInt = (n: number | null | undefined) => (n == null ? "–" : n.toLocaleString("en-US"));

export const fmtPct = (n: number | null | undefined, digits = 1) => (n == null ? "–" : `${n.toFixed(digits)}%`);

export const fmtRatio = (n: number | null | undefined, digits = 1) => (n == null ? "–" : `${(n * 100).toFixed(digits)}%`);

export const fmtNum = (n: number | null | undefined, digits = 3) => (n == null ? "–" : n.toFixed(digits));

export const fmtDate = (iso: string | null | undefined) => (iso ? iso.slice(0, 10) : "–");

export const fmtDateTime = (iso: string | null | undefined) => (iso ? iso.replace("T", " ").slice(0, 16) : "–");

export const SENTIMENT_COLORS = { negative: "#dc2626", neutral: "#94a3b8", positive: "#16a34a" } as const;

export const DRIFT_COLORS = { none: "#16a34a", moderate: "#d97706", significant: "#dc2626" } as const;
