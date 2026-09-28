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
            <td className="num strong">{fmtRatio(m.accuracy)}</td>
            <td className="num">{fmtNum(m.macro_f1)}</td>
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
  const bench = (data.benchmark ?? []).filter((b) => !b.oom).map((b) => ({ label: `${b.device} bs${b.batch_size}`, rps: b.reviews_per_sec }));
  const syn = data.synthetic_3class;
  const study = data.accuracy_study;

  return (
    <div className="page">
      <PageHeader
        title="Sentiment Validation"
        description={
          <>
            <span className="mono">{data.model}</span> evaluated on a labelled {data.dataset} sample of {fmtInt(data.sample.size)} tweets ({data.sample.selection}, seed{" "}
            {data.sample.seed}).
          </>
        }
      />
      <div className="banner" role="note">
        {data.ground_truth_note} The model predicts three classes, so three scorings are reported separately.
      </div>
      <div className="kpi-grid">
        <Kpi label="Accuracy (binary)" value={fmtRatio(m.binary_forced.accuracy)} hint="headline, every example scored" />
        <Kpi label="Macro F1 (binary)" value={fmtNum(m.binary_forced.macro_f1)} />
        <Kpi label="Accuracy (abstain)" value={fmtRatio(m.abstain.accuracy)} hint={`coverage ${fmtRatio(m.abstain.coverage)}`} />
        <Kpi label="Strict 3-class" value={fmtRatio(m.strict_3class.accuracy)} hint="neutral counted wrong" />
        <Kpi label="Neutral predictions" value={fmtRatio(m.neutral_prediction_rate)} />
        <Kpi label="CPU/GPU agreement" value={data.cpu_vs_gpu_label_agreement != null ? fmtRatio(data.cpu_vs_gpu_label_agreement) : "–"} hint="same labels on 500 examples" />
      </div>
      <Card title="Metrics by scoring method">
        <MetricsTable
          rows={[
            { name: "Binary forced", m: m.binary_forced, note: "positive iff P(pos) > P(neg)" },
            ...(m.binary_threshold
              ? [
                  {
                    name: "Binary, tuned threshold",
                    m: m.binary_threshold,
                    note: `positive iff P(pos)/(P(pos)+P(neg)) > ${m.binary_threshold.threshold} (tuned on validation only)`,
                  },
                ]
              : []),
            { name: "Abstain on neutral", m: m.abstain, note: "neutral predictions excluded" },
            { name: "Strict 3-class", m: m.strict_3class, note: "neutral counted as wrong" },
          ]}
        />
      </Card>
      {study && (
        <Card title="Accuracy study" subtitle={study.split}>
          <table className="table">
            <thead>
              <tr>
                <th>Configuration</th>
                <th className="num">Val accuracy</th>
                <th className="num">Val macro F1</th>
                <th className="num">Test accuracy</th>
                <th className="num">Test macro F1</th>
              </tr>
            </thead>
            <tbody>
              {study.rows.map((r) => (
                <tr key={`${r.config}-${r.model}`}>
                  <td>
                    <div className="theme-name">{r.config}</div>
                    <div className="muted small mono">
                      {r.model}
                      {r.in_product ? " · used by the pipeline" : " · not used by the pipeline"}
                    </div>
                  </td>
                  <td className="num">{fmtRatio(r.val_accuracy)}</td>
                  <td className="num">{fmtNum(r.val_macro_f1)}</td>
                  <td className="num strong">{fmtRatio(r.test_accuracy)}</td>
                  <td className="num">{fmtNum(r.test_macro_f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">{study.note}</p>
        </Card>
      )}
      <div className="grid-2">
        <Card title="Confusion matrix (Sentiment140)" subtitle="Binary truth vs the model's 3-class prediction">
          <Confusion rows={["negative", "positive"]} cols={["negative", "neutral", "positive"]} data={m.confusion_true2_pred3} />
        </Card>
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
      </div>
      {syn && (
        <Card title="Synthetic product reviews (3-class)" subtitle={syn.note}>
          <p className="small">
            Accuracy <strong>{fmtRatio(syn.accuracy)}</strong> · macro F1 {fmtNum(syn.macro_f1)} on {fmtInt(syn.n)} reviews. Neutral recall is only{" "}
            <strong>{fmtRatio(syn.per_class.neutral.recall)}</strong>: mixed or lukewarm reviews are often scored positive or negative.
          </p>
          <Confusion rows={["negative", "neutral", "positive"]} cols={["negative", "neutral", "positive"]} data={syn.confusion} />
        </Card>
      )}
      <Card title="Methodology">
        <pre className="pre">{data.methodology}</pre>
        <p className="muted small">
          Revision <span className="mono">{data.model_revision?.slice(0, 12)}</span> · device {data.device}
          {data.gpu ? ` (${data.gpu})` : ""} · evaluated {data.evaluation_date_utc?.slice(0, 10)}
          {data.reproducibility && ` · reproducible: ${data.reproducibility.identical_labels_between_runs ? "yes" : "no"}`}
        </p>
      </Card>
    </div>
  );
}
