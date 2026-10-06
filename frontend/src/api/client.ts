import type {
  DataHealth,
  DriftResponse,
  EvidenceItem,
  IssueDetail,
  IssuesResponse,
  MetricsResponse,
  ModelInfo,
  ProductBrief,
  Review,
  ReviewDetail,
  SentimentValidation,
  Theme,
  ThemeDetail,
} from "./types";

export interface BatchInfo {
  batch_id: string;
  name?: string;
  filename?: string;
  db_path?: string;
  created_at_utc?: string;
  reviews?: number | null;
  dates_available?: boolean | null;
  n_themes?: number | null;
  active?: boolean;
}

export interface BatchJob {
  job_id: string;
  batch_id: string;
  status: string;
  filename?: string;
  error?: string | null;
  started_at_utc?: string;
  finished_at_utc?: string;
  batch?: BatchInfo;
}

export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError("Cannot reach the analytics API. Is the backend running?", 0);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail && typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export interface ReviewQuery {
  theme_id?: string;
  sentiment?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

export const api = {
  health: () => request<{ status: string; reviews: number; dataset: string }>("/health"),
  metrics: () => request<MetricsResponse>("/metrics"),
  themes: () => request<{ count: number; themes: Theme[] }>("/themes"),
  theme: (id: string) => request<ThemeDetail>(`/themes/${enc(id)}`),
  issues: () => request<IssuesResponse>("/issues"),
  issue: (id: string) => request<IssueDetail>(`/issues/${enc(id)}`),
  evidence: (id: string) => request<{ theme_id: string; count: number; evidence: EvidenceItem[] }>(`/issues/${enc(id)}/evidence`),
  reviews: (query: ReviewQuery) => {
    const params = new URLSearchParams();
    Object.entries(query).forEach(([k, v]) => {
      if (v !== undefined && v !== "" && v !== null) params.set(k, String(v));
    });
    return request<{ total: number; limit: number; offset: number; reviews: Review[] }>(`/reviews?${params}`);
  },
  review: (id: string) => request<ReviewDetail>(`/reviews/${enc(id)}`),
  sentimentValidation: () => request<SentimentValidation>("/model/evaluation"),
  drift: () => request<DriftResponse>("/drift"),
  dataHealth: () => request<DataHealth>("/data-health"),
  modelInfo: () => request<ModelInfo>("/model-info"),
  productBrief: (engine: "auto" | "qwen" | "template") =>
    request<ProductBrief>("/product-brief", { method: "POST", body: JSON.stringify({ engine }) }),
  batches: () => request<{ batches: BatchInfo[]; active_db: string | null }>("/batches"),
  uploadBatch: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<BatchJob>("/batches/upload", { method: "POST", body });
  },
  batchJob: (jobId: string) => request<BatchJob>(`/batches/jobs/${enc(jobId)}`),
  activateBatch: (batchId: string) => request<BatchInfo>(`/batches/${enc(batchId)}/activate`, { method: "POST" }),
};
