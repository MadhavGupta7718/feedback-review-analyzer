import { api } from "../api/client";
import { useApi } from "../api/useApi";
import type { Sentiment } from "../api/types";
import { Card, ErrorState, Kpi, Loading, PageHeader } from "../components/ui";
import { fmtInt, fmtNum, fmtRatio } from "../lib/format";

function Confusion({ rows, cols, data }: { rows: string[]; cols: Sentiment[]; data: Record<string, Record<string, number>> }) {
  const max = Math.max(...rows.flatMap((r) => cols.map((c) => data[r]?.[c] ?? 0)), 1);
  return (
    <table className="confusion" aria-label="Confusion matrix">
      <thead>
        <tr>
          <th>true ↓ / predicted →</th>
          {cols.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r}>
            <th>{r}</th>
            {cols.map((c) => {
              const v = data[r]?.[c] ?? 0;
              return (
                <td key={c} style={{ background: `rgba(79, 70, 229, ${0.08 + (v / max) * 0.6})` }}>
                  {fmtInt(v)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SentimentValidation() {
  const { data, error, loading, reload } = useApi(() => api.sentimentValidation());
  if (loading) return <Loading />;
  if (error || !data) return <ErrorState message={error ?? "No model evaluation yet. Run scripts/finetune_amazon_sentiment.py"} onRetry={reload} />;

  const m = data.metrics;
  const per = m.per_class ?? m.three_class?.per_class;
  const conf3 = m.confusion_3x3 ?? m.three_class?.confusion;

  return (
    <div className="page">
      <PageHeader
        title="Model Validation"
        description={
          <>
            Global metrics for the <strong>Amazon-trained</strong> 3-class sentiment model on its held-out TEST split
            ({fmtInt(data.sample.size)} reviews). Not tied to any uploaded batch.
          </>
        }
      />
      <div className="banner" role="note">
        {data.ground_truth_note} Uploaded CSVs do not get their own accuracy page.
      </div>
      <div className="kpi-grid">
        <Kpi label="Test accuracy" value={m.headline_accuracy != null ? fmtRatio(m.headline_accuracy) : "–"} hint="argmax vs teacher text labels" />
        <Kpi label="Macro recall" value={m.macro_recall != null ? fmtRatio(m.macro_recall) : "–"} hint="mean of neg/neu/pos recall" tone="warn" />
        <Kpi label="Neg recall" value={per ? fmtRatio(per.negative.recall) : "–"} tone="bad" />
        <Kpi label="Neu recall" value={per ? fmtRatio(per.neutral.recall) : "–"} />
        <Kpi label="Pos recall" value={per ? fmtRatio(per.positive.recall) : "–"} tone="good" />
      </div>

      {per && (
        <Card title="Per-class (TEST)">
          <table className="table">
            <thead>
              <tr>
                <th>Class</th>
                <th className="num">Precision</th>
                <th className="num">Recall</th>
                <th className="num">F1</th>
                <th className="num">Support</th>
              </tr>
            </thead>
            <tbody>
              {(["negative", "neutral", "positive"] as const).map((c) => (
                <tr key={c}>
                  <td>{c}</td>
                  <td className="num">{fmtNum(per[c].precision, 3)}</td>
                  <td className="num">{fmtNum(per[c].recall, 3)}</td>
                  <td className="num">{fmtNum(per[c].f1, 3)}</td>
                  <td className="num">{fmtInt(per[c].support)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {conf3 && (
        <Card title="Confusion matrix (TEST)" subtitle="Ground truth from star ratings vs predicted">
          <Confusion rows={["negative", "neutral", "positive"]} cols={["negative", "neutral", "positive"]} data={conf3} />
        </Card>
      )}

      <Card title="Methodology">
        <pre className="pre">{data.methodology}</pre>
        <p className="muted small">
          Model <span className="mono">{data.model}</span>
          {data.evaluation_date_utc ? ` · evaluated ${data.evaluation_date_utc.slice(0, 10)}` : ""}
          {data.device ? ` · ${data.device}` : ""}
        </p>
      </Card>
    </div>
  );
}
