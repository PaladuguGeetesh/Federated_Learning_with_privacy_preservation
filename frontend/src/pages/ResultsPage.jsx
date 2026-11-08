import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import results from '../data/results.json'
import { Card, PageTitle, Legend, ChartTip, Stat } from '../components/ui.jsx'
import { int, pct, fixed } from '../format.js'
import { colorFor } from './Overview.jsx'

const METRICS = [
  ['accuracy', 'Accuracy', (v) => pct(v)],
  ['precision', 'Precision', (v) => pct(v)],
  ['recall', 'Recall', (v) => pct(v)],
  ['f1', 'F1 (attack)', (v) => pct(v)],
  ['roc_auc', 'ROC-AUC', (v) => fixed(v, 4)],
  ['pr_auc', 'PR-AUC', (v) => (v == null ? 'n/a' : fixed(v, 4))],
  ['fpr', 'False-alarm rate', (v) => pct(v, 3), true],
]

export default function ResultsPage() {
  const { models, test_set: ts } = results
  const bestOf = (key, lowerBetter) => {
    const vals = models.map((m) => m[key]).filter((v) => v != null)
    return lowerBetter ? Math.min(...vals) : Math.max(...vals)
  }
  const errors = models.map((m) => ({ model: m.model, 'Missed attacks': m.fn, 'False alarms': m.fp }))

  return (
    <>
      <PageTitle title="Results">
        All four models are scored once on the same held-out test set: {int(ts.flows)} flows ({int(ts.attacks)} attacks, {int(ts.benign)} benign).
      </PageTitle>
      <div className="stat-grid">
        {models.map((m) => (
          <Stat key={m.model} label={m.model} value={pct(m.f1)} hint={`attack F1 · ${int(m.fn)} missed · ${int(m.fp)} false alarms`} />
        ))}
      </div>

      <Card title="Test-set metrics" subtitle="Best value in each column is marked ★. Thresholds for the ANN and the logistic regression were chosen on validation data.">
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr><th>Model</th>{METRICS.map(([k, l]) => <th key={k} className="r">{l}</th>)}<th className="r">Threshold</th></tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.model}>
                  <td><span className="swatch" style={{ background: colorFor(m.model) }} /> {m.model}</td>
                  {METRICS.map(([k, , f, low]) => {
                    const best = m[k] != null && m[k] === bestOf(k, low)
                    return <td key={k} className={`r ${best ? 'best' : ''}`}>{f(m[k])}{best && ' ★'}</td>
                  })}
                  <td className="r muted">{fixed(m.threshold, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">The Random Forest’s PR-AUC was not recorded by its original evaluation, so it is shown as n/a.</p>
      </Card>

      <div className="grid-2">
        <Card title="Errors on the test set" subtitle="Counts of wrong decisions — lower is better">
          <Legend items={[{ label: 'Missed attacks (false negatives)', color: 'var(--s1)' }, { label: 'False alarms (false positives)', color: 'var(--s2)' }]} />
          <div className="chart">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={errors} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }} barGap={3}>
                <CartesianGrid horizontal={false} stroke="var(--grid)" />
                <XAxis type="number" tick={{ fill: 'var(--ink-2)', fontSize: 12 }} tickLine={false} axisLine={false} tickFormatter={int} />
                <YAxis type="category" dataKey="model" tick={{ fill: 'var(--ink-2)', fontSize: 12 }} tickLine={false} axisLine={{ stroke: 'var(--grid)' }} width={132} />
                <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTip format={(v) => int(v)} />} />
                <Bar dataKey="Missed attacks" fill="var(--s1)" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false} />
                <Bar dataKey="False alarms" fill="var(--s2)" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Federated versus centralized" subtitle="Gap in percentage points (centralized minus federated ANN)">
          <div className="table-wrap">
            <table className="tbl">
              <thead><tr><th>Metric</th><th className="r">Centralized ANN − Federated ANN</th><th className="r">Random Forest − Federated ANN</th></tr></thead>
              <tbody>
                {Object.keys(results.gaps_ann).map((k) => (
                  <tr key={k}>
                    <td style={{ textTransform: 'capitalize' }}>{k.replace('_', ' ').replace('roc auc', 'ROC-AUC').replace('pr auc', 'PR-AUC')}</td>
                    <td className="r">{signed(results.gaps_ann[k])}</td>
                    <td className="r">{results.gaps_rf[k] == null ? 'n/a' : signed(results.gaps_rf[k])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">A positive number means the first model is better. Federation costs under 0.1 pp of F1 against the same network trained centrally.</p>
        </Card>
      </div>

      <Card title="Confusion matrices" subtitle="Rows are the true class, columns the predicted class; each cell shows the number of flows and its share of the row">
        <div className="cm-grid">
          {models.map((m) => <Confusion key={m.model} m={m} />)}
        </div>
      </Card>

      <Card title="Notes">
        <ul className="notes">{results.notes.map((n) => <li key={n}>{n}</li>)}</ul>
      </Card>
    </>
  )
}

const signed = (v) => `${v > 0 ? '+' : ''}${v.toFixed(4)} pp`

function Confusion({ m }) {
  const rows = [
    { name: 'Benign', total: m.tn + m.fp, cells: [{ v: m.tn, ok: true }, { v: m.fp, ok: false }] },
    { name: 'Attack', total: m.fn + m.tp, cells: [{ v: m.fn, ok: false }, { v: m.tp, ok: true }] },
  ]
  return (
    <figure className="cm">
      <figcaption><span className="swatch" style={{ background: colorFor(m.model) }} /> {m.model}</figcaption>
      <div className="cm-table">
        <span />
        <span className="cm-h">Pred. benign</span><span className="cm-h">Pred. attack</span>
        {rows.map((r) => (
          <FragmentRow key={r.name} r={r} />
        ))}
      </div>
    </figure>
  )
}

function FragmentRow({ r }) {
  return (
    <>
      <span className="cm-h left">True {r.name.toLowerCase()}</span>
      {r.cells.map((c, i) => {
        const share = c.v / r.total
        return (
          <div key={i} className={`cm-cell ${c.ok ? 'ok' : 'err'}`} style={{ '--a': c.ok ? 0.1 + share * 0.3 : Math.min(0.1 + share * 40, 0.5) }}>
            <b>{int(c.v)}</b>
            <span>{pct(share, 2)}</span>
          </div>
        )
      })}
    </>
  )
}
