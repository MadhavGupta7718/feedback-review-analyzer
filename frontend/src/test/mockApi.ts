import { vi } from "vitest";
import brief from "./fixtures/brief.json";
import dataHealth from "./fixtures/data_health.json";
import drift from "./fixtures/drift.json";
import evidence from "./fixtures/evidence.json";
import health from "./fixtures/health.json";
import issue from "./fixtures/issue.json";
import issues from "./fixtures/issues.json";
import metrics from "./fixtures/metrics.json";
import modelInfo from "./fixtures/model_info.json";
import review from "./fixtures/review.json";
import reviews from "./fixtures/reviews.json";
import sentiment from "./fixtures/sentiment.json";
import theme from "./fixtures/theme.json";
import themes from "./fixtures/themes.json";

export const fixtures = { brief, dataHealth, drift, evidence, health, issue, issues, metrics, modelInfo, review, reviews, sentiment, theme, themes };

type Override = (url: string, init?: RequestInit) => unknown | undefined;

function route(path: string, init?: RequestInit): unknown {
  const p = path.split("?")[0];
  if (p === "/health") return health;
  if (p === "/metrics") return metrics;
  if (p === "/themes") return themes;
  if (/^\/themes\/theme_\d{3}$/.test(p)) return theme;
  if (p === "/issues") return issues;
  if (/^\/issues\/theme_\d{3}\/evidence$/.test(p)) return evidence;
  if (/^\/issues\/theme_\d{3}$/.test(p)) return issue;
  if (p === "/reviews") return reviews;
  if (/^\/reviews\/[\w-]+$/.test(p)) return review;
  if (p === "/sentiment/validation") return sentiment;
  if (p === "/drift") return drift;
  if (p === "/data-health") return dataHealth;
  if (p === "/model-info") return modelInfo;
  if (p === "/product-brief" && init?.method === "POST") return brief;
  return undefined;
}

export function mockFetch(override?: Override) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    const path = url.pathname + url.search;
    const body = override?.(path, init) ?? route(path, init);
    if (body === undefined) return new Response(JSON.stringify({ detail: "Not found" }), { status: 404 });
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}
