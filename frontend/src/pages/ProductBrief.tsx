import { useState } from "react";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import type { GenerationPath, ProductBrief as Brief } from "../api/types";
import { ReviewDrawer } from "../components/ReviewDrawer";
import { Card, ErrorState, PageHeader } from "../components/ui";

type Engine = "auto" | "qwen" | "template";

const PATH_LABEL: Record<GenerationPath, { label: string; detail: string }> = {
  qwen_live: { label: "Qwen2.5-3B (live)", detail: "Written just now by the local Qwen model from the deterministic fact sheet, then validated." },
  qwen_precomputed: { label: "Qwen2.5-3B (precomputed)", detail: "Written offline by the local Qwen model, validated, and stored with the analytics." },
  template: { label: "Deterministic template", detail: "Assembled directly from the analytics without a language model." },
};

function Section({ title, items }: { title: string; items: string[] }) {
  if (!items?.length) return null;
  return (
    <>
      <h3 className="section-title">{title}</h3>
      <ul className="brief-list">
        {items.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ul>
    </>
  );
}

export function ProductBrief() {
  const status = useApi(() => api.modelInfo());
  const [engine, setEngine] = useState<Engine>("auto");
  const [brief, setBrief] = useState<Brief | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [openReview, setOpenReview] = useState<string | null>(null);

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      setBrief(await api.productBrief(engine));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const eng = status.data?.brief_engine;
  const path = brief ? PATH_LABEL[brief.generation_path] : null;

  return (
    <div className="page">
      <PageHeader
        title="Product Brief"
        description="A one-page summary for product teams. Numbers, theme names and quotes come from the deterministic analytics; the language model only writes the prose."
      />
      <Card>
        <div className="toolbar">
          <label className="small" htmlFor="engine">
            Engine
          </label>
          <select id="engine" className="input" value={engine} onChange={(e) => setEngine(e.target.value as Engine)}>
            <option value="auto">Auto (Qwen if available, else fallback)</option>
            <option value="qwen">Qwen (with fallback)</option>
            <option value="template">Template only</option>
          </select>
          <button className="btn btn-primary" onClick={generate} disabled={busy}>
            {busy ? "Generating…" : "Generate product brief"}
          </button>
          {eng && (
            <span className="muted small">
              Qwen {eng.qwen_installed ? "installed" : "not installed"} · GPU {eng.cuda_available ? "available" : "unavailable"} ·{" "}
              {status.data?.precomputed_qwen_brief ? "precomputed brief stored" : "no precomputed brief"}
            </span>
          )}
        </div>
        {busy && engine !== "template" && <p className="muted small">Live generation on a laptop GPU can take one to three minutes (model load, generation and up to one validated retry). Hosts without a GPU return the stored brief immediately.</p>}
      </Card>

      {error && <ErrorState message={error} onRetry={generate} />}

      {brief && path && (
        <Card
          title="Product brief"
          subtitle={`Generated ${brief.generated_at_utc?.slice(0, 19).replace("T", " ")} UTC`}
          actions={
            <span className={`badge badge-path-${brief.generation_path}`} data-testid="generation-path">
              {path.label}
            </span>
          }
        >
          <p className="muted small">
            {path.detail}
            {brief.written_by_model && ` Model-written sections: ${brief.written_by_model.map((s) => s.replace(/_/g, " ")).join(", ")}; every other section is deterministic.`}
            {brief.validation?.attempts != null &&
              ` Attempts: ${Array.isArray(brief.validation.attempts) ? brief.validation.attempts.length : brief.validation.attempts}.`}
          </p>
          {brief.fallback_reasons && brief.fallback_reasons.length > 0 && (
            <div className="banner banner-warn" role="note">
              <strong>Fallback used.</strong> {brief.fallback_reasons.join(" · ")}
            </div>
          )}
          {brief.validation && (
            <p className="small">
              Validation: {brief.validation.passed ? "passed" : "failed"}
              {brief.validation.checks &&
                ` (${Object.entries(brief.validation.checks)
                  .map(([k, v]) => `${k.replace(/_/g, " ")} ${v ? "✓" : "✗"}`)
                  .join(", ")})`}
            </p>
          )}
          <h3 className="section-title">Executive summary</h3>
          <p className="brief-summary">{brief.executive_summary}</p>
          <Section title="Top complaints" items={brief.top_complaints} />
          <Section title="Emerging complaints" items={brief.emerging_complaints} />
          {brief.evidence?.length > 0 && (
            <>
              <h3 className="section-title">Evidence</h3>
              {brief.evidence.map((e) => (
                <blockquote key={e.review_id} className="quote">
                  “{e.text}”
                  <footer>
                    {e.theme} ·{" "}
                    <button className="link-btn mono" onClick={() => setOpenReview(e.review_id)}>
                      {e.review_id}
                    </button>
                  </footer>
                </blockquote>
              ))}
            </>
          )}
          <Section title="Possible associations" items={brief.possible_associations} />
          <Section title="Suggested investigation areas" items={brief.suggested_investigation_areas} />
          <Section title="Caveats" items={brief.caveats} />
        </Card>
      )}
      {openReview && <ReviewDrawer reviewId={openReview} onClose={() => setOpenReview(null)} />}
    </div>
  );
}
