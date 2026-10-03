import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import type { BinaryMetrics, Sentiment } from "../api/types";
import { Card, ErrorState, Kpi, Loading, PageHeader } from "../components/ui";
import { fmtInt, fmtNum, fmtRatio } from "../lib/format";

function MetricsTable({ rows }: { rows: { name: string; m: BinaryMetrics; note: string }[] }) {
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Scoring</th>
          <th className="num">n</th>
          <th className="num">Accuracy</th>
          <th className="num">Macro F1</th>
          <th className="num">Neg P / R</th>
          <th className="num">Pos P / R</th>
          <th className="num">Coverage</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ name, m, note }) => (
          <tr key={name}>
            <td>
              <div className="theme-name">{name}</div>
              <div className="muted small">{note}</div>
            </td>
            <td className="num">{fmtInt(m.n)}</td>
            <td className="num strong">{m.accuracy != null ? fmtRatio(m.accuracy) : "–"}</td>
            <td className="num">{m.macro_f1 != null ? fmtNum(m.macro_f1) : "–"}</td>
            <td className="num">
              {fmtNum(m.negative.precision, 2)} / {fmtNum(m.negative.recall, 2)}
            </td>
            <td className="num">
              {fmtNum(m.positive.precision, 2)} / {fmtNum(m.positive.recall, 2)}
            </td>
            <td className="num">{m.coverage != null ? fmtRatio(m.coverage) : "100%"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

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
  if (error || !data) return <ErrorState message={error ?? "No data"} onRetry={reload} />;

  const m = data.metrics;
  const per = m.per_class ?? m.three_class?.per_class;
  const be = data.batch_evaluation;
  const bench = (data.benchmark ?? []).filter((b) => !b.oom).map((b) => ({ label: `${b.device} bs${b.batch_size}`, rps: b.reviews_per_sec }));
  const conf3 = m.confusion_3x3;
  const conf2 = m.confusion_true2_pred3;

  return (
    <div className="page">
      <PageHeader
        title="Sentiment Validation"
        description={
          <>
            <span className="mono">{data.model}</span> scored on{" "}
            <strong>this database only</strong> ({fmtInt(data.sample.size)} labelled reviews · {data.sample.selection}).
          </>
        }
      />
      <div className="banner" role="note">
        {data.ground_truth_note} Metrics never mix in another dataset. Switch the analytics DB to see another batch.
      </div>
      <div className="kpi-grid">
        <Kpi label="Accuracy (3-class)" value={m.headline_accuracy != null ? fmtRatio(m.headline_accuracy) : fmtRatio(m.strict_3class?.accuracy ?? m.three_class?.accuracy ?? 0)} hint="argmax label vs ground truth" />
        <Kpi label="Macro recall" value={m.macro_recall != null ? fmtRatio(m.macro_recall) : "–"} hint="mean of neg/neu/pos recall" tone="warn" />
        <Kpi label="Neg recall" value={per ? fmtRatio(per.negative.recall) : "–"} tone="bad" />
        <Kpi label="Neu recall" value={per ? fmtRatio(per.neutral.recall) : "–"} />
        <Kpi label="Pos recall" value={per ? fmtRatio(per.positive.recall) : "–"} tone="good" />
        <Kpi label="Neutral predictions" value={fmtRatio(m.neutral_prediction_rate)} />
      </div>

      {be?.overall_recall && (
        <Card title="Overall recall (this batch)" subtitle="Each score is computed only from reviews stored in this database">
          <div className="kpi-grid">
            <Kpi label="Sentiment macro recall" value={be.overall_recall.sentiment_macro_recall != null ? fmtRatio(be.overall_recall.sentiment_macro_recall) : "–"} />
            <Kpi label="Theme recall" value={be.overall_recall.theme_recall != null ? fmtRatio(be.overall_recall.theme_recall) : "–"} hint="planted themes recovered" />
            <Kpi label="Radar recall" value={be.overall_recall.radar_recall != null ? fmtRatio(be.overall_recall.radar_recall) : "–"} hint="planted temporal patterns" />
            <Kpi label="PII recall" value={be.overall_recall.pii_recall != null ? fmtRatio(be.overall_recall.pii_recall) : "–"} />
            <Kpi label="Mean available" value={be.overall_recall.mean_available_recalls != null ? fmtRatio(be.overall_recall.mean_available_recalls) : "–"} tone="warn" />
          </div>
        </Card>
      )}

      {m.binary_forced && m.binary_forced.accuracy != null && (
        <Card title="Binary scoring (non-neutral ground truth)">
          <MetricsTable
            rows={[
              { name: "Binary forced", m: m.binary_forced, note: "positive iff P(pos) > P(neg); neutral truth rows excluded" },
              ...(m.strict_3class ? [{ name: "Strict 3-class (neg/pos rows)", m: m.strict_3class, note: "neutral prediction counts wrong" }] : []),
            ]}
          />
        </Card>
      )}

      <div className="grid-2">
        {conf3 && (
          <Card title="Confusion matrix (3-class)" subtitle="Ground truth vs predicted for this batch">
            <Confusion rows={["negative", "neutral", "positive"]} cols={["negative", "neutral", "positive"]} data={conf3} />
          </Card>
        )}
        {!conf3 && conf2 && (
          <Card title="Confusion matrix" subtitle="Binary truth vs 3-class prediction">
            <Confusion rows={["negative", "positive"]} cols={["negative", "neutral", "positive"]} data={conf2} />
          </Card>
        )}
        {bench.length > 0 && (
          <Card title="Throughput" subtitle="Reviews per second by device and batch size">
            <div className="chart-sm">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={bench}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="rps" fill="#4f46e5" name="reviews/s" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}
      </div>

      <Card title="Methodology">
        <pre className="pre">{data.methodology}</pre>
        <p className="muted small">
          Revision <span className="mono">{data.model_revision?.slice(0, 12)}</span> · device {data.device}
          {data.gpu ? ` (${data.gpu})` : ""} · evaluated {data.evaluation_date_utc?.slice(0, 10)}
          {data.ground_truth_source && ` · labels: ${data.ground_truth_source}`}
        </p>
      </Card>
    </div>
  );
}
