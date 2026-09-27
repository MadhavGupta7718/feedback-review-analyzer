export type Sentiment = "negative" | "neutral" | "positive";
export type RadarStatus =
  | "NEW"
  | "EMERGING"
  | "STABLE"
  | "DECLINING"
  | "INSUFFICIENT_EVIDENCE"
  | "NOT_A_COMPLAINT"
  | "NO_DATA";
export type DriftStatus = "none" | "moderate" | "significant";

export interface DatasetInfo {
  name: string;
  kind: string;
  seed?: number;
  note?: string;
}

export interface EmergingSummary {
  theme_id: string;
  name: string;
  status: RadarStatus;
  current_mentions: number;
  previous_mentions: number;
  growth_pct: number | null;
  growth_label: string;
  negative_ratio: number;
  priority: number;
}

export interface Overview {
  total_reviews: number;
  sentiment_counts: Record<Sentiment, number>;
  sentiment_pct: Record<Sentiment, number>;
  n_themes: number;
  n_complaint_themes: number;
  top_complaint: { theme_id: string; name: string; negative_count: number; size: number; negative_pct: number } | null;
  emerging: EmergingSummary[];
  pii_redactions_total: number;
  drift_status: DriftStatus;
  date_range: [string, string];
  avg_rating: number | null;
}

export interface MetricsResponse {
  overview: Overview;
  dataset: DatasetInfo;
  generated_at_utc: string;
  pipeline_seconds: number | null;
  device: string | null;
}

export interface Theme {
  theme_id: string;
  name: string;
  keywords: string[];
  size: number;
  core_size: number;
  coherence: number;
  negative_count: number;
  neutral_count: number;
  positive_count: number;
  negative_pct: number;
  weekly_counts: number[];
  is_complaint: boolean;
  avg_rating: number | null;
  radar_status: RadarStatus;
  growth_pct: number | null;
  growth_label: string;
  priority: number;
}

export interface Review {
  review_id: string;
  created_at: string | null;
  rating: number | null;
  platform: string | null;
  app_version: string | null;
  text: string;
  sentiment: Sentiment;
  confidence: number;
  theme_id: string | null;
  theme_assignment: string | null;
  pii_redactions: number;
  source: string;
  theme_name?: string | null;
}

export interface ReviewDetail extends Review {
  p_negative: number;
  p_neutral: number;
  p_positive: number;
  theme_similarity: number | null;
  used_as_evidence: { theme_id: string; kind: string; rank: number }[];
}

export interface Association {
  segment: string;
  value: string;
  theme_share: number;
  overall_share: number;
  lift: number;
  support: number;
  statement: string;
}

export interface Issue {
  theme_id: string;
  name: string;
  status: RadarStatus;
  current_mentions: number;
  previous_mentions: number;
  growth_pct: number | null;
  growth_label: string;
  negative_mentions_current: number;
  negative_ratio: number;
  negative_ratio_all: number;
  total_mentions: number;
  acceleration_pp: number | null;
  trend: string;
  weekly_counts: number[];
  evidence_review_ids: string[];
  priority: number;
  associations: Association[];
  reasons: string[];
}

export interface Calculation {
  window: { current: [string, string]; previous: [string, string] };
  growth: string;
  negative_ratio: string;
  priority: string;
  thresholds: Record<string, number>;
}

export interface RadarWindow {
  current_start: string;
  end: string;
  previous_start: string;
  current_reviews: number;
  previous_reviews: number;
  has_previous_window: boolean;
}

export interface IssuesResponse {
  count: number;
  window: RadarWindow | null;
  formula: string | null;
  params: Record<string, number> | null;
  rules: string | null;
  issues: Issue[];
}

export interface IssueDetail extends Issue {
  calculation: Calculation;
  formula: string | null;
  window: RadarWindow | null;
  theme: { theme_id: string; name: string; keywords: string[]; size: number; coherence: number } | null;
}

export interface ThemeDetail extends Theme {
  representative_reviews: Review[];
  radar: (Issue & { calculation: Calculation }) | null;
  segments: { app_version: { value: string; n: number }[]; platform: { value: string; n: number }[] };
}

export interface EvidenceItem extends Review {
  evidence_kind: "representative" | "radar_evidence";
  rank: number;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface BinaryMetrics {
  n: number;
  accuracy: number;
  negative: ClassMetrics;
  positive: ClassMetrics;
  macro_f1: number;
  coverage?: number;
}

export interface BenchmarkRow {
  device: string;
  dtype: string;
  batch_size: number;
  n: number;
  seconds: number;
  reviews_per_sec: number;
  peak_gpu_mem_gb?: number | null;
  oom: boolean;
}

export interface SentimentValidation {
  evaluation_date_utc: string;
  model: string;
  model_revision: string;
  dataset: string;
  device: string;
  gpu?: string | null;
  sample: { seed: number; size: number; label_counts: Record<string, number>; selection: string };
  ground_truth_note: string;
  methodology: string;
  metrics: {
    binary_forced: BinaryMetrics;
    strict_3class: BinaryMetrics;
    abstain: BinaryMetrics;
    neutral_prediction_rate: number;
    confusion_true2_pred3: Record<"negative" | "positive", Record<Sentiment, number>>;
  };
  mean_confidence: number;
  reproducibility?: { identical_labels_between_runs: boolean; max_abs_prob_diff: number };
  cpu_vs_gpu_label_agreement?: number;
  benchmark?: BenchmarkRow[];
  synthetic_3class?: {
    note: string;
    n: number;
    accuracy: number;
    per_class: Record<Sentiment, ClassMetrics>;
    macro_f1: number;
    confusion: Record<Sentiment, Record<Sentiment, number>>;
  };
}

export interface DriftMetric {
  metric?: string;
  value: number;
  status: DriftStatus;
  js_distance?: number;
  p_value?: number;
  reference_n?: number;
  current_n?: number;
  reference_per_day?: number;
  current_per_day?: number;
  reference_mean?: number;
  current_mean?: number;
  top_changes?: { category: string; reference_share: number; current_share: number; change_pp: number }[];
}

export interface DriftBlock {
  thresholds: Record<string, number>;
  metrics: Record<"sentiment" | "theme" | "volume" | "review_length", DriftMetric>;
  overall_status: DriftStatus;
  method_note: string;
  reference_window: [string, string];
  current_window: [string, string];
}

export interface WeeklyDrift {
  week: number;
  week_start: string;
  n: number;
  sentiment: { value: number; status: DriftStatus };
  theme: { value: number; status: DriftStatus };
  volume: { value: number; status: DriftStatus };
  review_length: { value: number; status: DriftStatus };
  overall_status: DriftStatus;
}

export interface DriftResponse {
  current_vs_previous: DriftBlock;
  weekly_vs_baseline: WeeklyDrift[];
  note: string;
}

export interface DataHealth {
  dataset: DatasetInfo;
  input_rows: number;
  processed_reviews: number;
  rejected: Record<string, number>;
  ingestion_duplicates_removed: number;
  identical_text_rows_kept: number;
  missing_timestamps: number;
  mojibake_repaired: number;
  html_entities_decoded: number;
  pii_redactions: Record<string, number>;
  rows_with_pii: number;
  theme_stats: {
    n_clusters: number;
    n_micro_clusters: number;
    hdbscan_noise_fraction: number;
    reassigned_by_nearest_centroid: number;
    unassigned: number;
    unassigned_fraction: number;
    params: Record<string, number | string | null>;
  };
  traceability: { links_checked: number; problems: unknown[]; reviews_with_residual_pii_pattern: number; status: string } | null;
  pii_audit: { overall_recall: number | null; recall_by_type: Record<string, string> | null; false_positive_row_rate: number | null };
  sentiment140_validation: {
    rows: number | null;
    status: string | null;
    label_distribution: Record<string, number> | null;
    duplicate_ids: number | null;
    date_min: string | null;
    date_max: string | null;
  };
}

export interface ModelInfo {
  pipeline_models: Record<string, Record<string, unknown>>;
  verification: Record<string, { state: string; revision?: string; disk_gb?: number; load_seconds?: number; checks?: Record<string, boolean>; plan?: Record<string, string>; gpu_memory_allocated_gb?: number }>;
  verification_status: string | null;
  verification_offline: boolean | null;
  hardware_at_pipeline_run: Record<string, unknown> | null;
  performance: { stages?: Record<string, { seconds: number; rss_mb: number }>; total_seconds?: number; device?: string; n_reviews?: number; sentiment_reviews_per_sec?: number; peak_gpu_mem_gb?: number; peak_rss_mb?: number };
  brief_engine: { brief_mode: string; qwen_installed: boolean; cuda_available: boolean; qwen_loaded: boolean; last_error: string | null; live_generation_possible: boolean };
  precomputed_qwen_brief: boolean;
}

export type GenerationPath = "qwen_live" | "qwen_precomputed" | "template";

export interface ProductBrief {
  generation_path: GenerationPath;
  generated_at_utc: string;
  executive_summary: string;
  top_complaints: string[];
  emerging_complaints: string[];
  evidence: { theme: string; review_id: string; text: string }[];
  possible_associations: string[];
  suggested_investigation_areas: string[];
  caveats: string[];
  fallback_reasons?: string[];
  model?: string;
  validation?: { passed: boolean; errors: string[]; checks?: Record<string, boolean> };
}
