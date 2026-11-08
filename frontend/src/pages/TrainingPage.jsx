import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts'
import results from '../data/results.json'
import dataset from '../data/dataset.json'
import { Card, PageTitle, Legend, ChartTip, Stat } from '../components/ui.jsx'
import { int, fixed, annParams } from '../format.js'
import { RoundFlowCard } from '../components/Diagrams.jsx'

const { ann, lr, ann_local1: ann1 } = results.federated
const axis = { fill: 'var(--ink-2)', fontSize: 12 }

// merge per-round histories into one array of {round, a, b, ...}; round 0 is the untrained model
function merge(series, from = 1) {
  const out = []
  for (let r = from; r <= ann.rounds; r++) {
    const row = { round: r }
    for (const [key, hist, field] of series) row[key] = hist.find((h) => h.round === r)?.[field]
    out.push(row)
  }
  return out
}

function RoundChart({ data, lines, domain, ticks, fmt, height = 280 }) {
  return (
    <>
      <Legend items={lines.map((l) => ({ label: l.name, color: l.color }))} />
      <div className="chart">
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="round" tick={axis} tickLine={false} axisLine={{ stroke: 'var(--grid)' }} label={{ value: 'Communication round', position: 'insideBottom', offset: -2, fill: 'var(--ink-3)', fontSize: 12 }} height={40} />
            <YAxis domain={domain} ticks={ticks} tick={axis} tickLine={false} axisLine={false} tickFormatter={fmt} width={58} />
            <Tooltip content={<ChartTip labelPrefix="Round " format={(v) => fmt(v)} />} />
            {lines.map((l) => (
              <Line key={l.key} type="monotone" dataKey={l.key} name={l.name} stroke={l.color} strokeWidth={2} dot={{ r: 3, strokeWidth: 2, stroke: 'var(--surface)', fill: l.color }} activeDot={{ r: 5, stroke: 'var(--surface)', strokeWidth: 2 }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  )
}

export default function TrainingPage() {
  const r0 = ann.history[0]
  const last = ann.history[ann.history.length - 1]
  const client = dataset.splits.find((s) => s.file === 'client_A.parquet')
  const trials = results.optuna.trials
  const done = trials.filter((t) => t.state === 'COMPLETE').map((t) => ({ ...t, x: t.trial, y: t.pr_auc }))
  const pruned = trials.filter((t) => t.state === 'PRUNED').map((t) => ({ ...t, x: t.trial, y: t.pr_auc }))
  const p = results.optuna.params

  return (
    <>
      <PageTitle title="Training">
        Federated Averaging over {ann.rounds} communication rounds, plus the Optuna search that tuned the network. Validation data only — no test data is used here.
      </PageTitle>
      <div className="stat-grid">
        <Stat label="Communication rounds" value={ann.rounds} hint={`${ann.local_epochs} local epochs per round`} />
        <Stat label="Federated ANN, validation PR-AUC" value={fixed(last.val_pr_auc, 4)} hint={`from ${fixed(r0.val_pr_auc, 4)} untrained; best at round ${ann.best_round}`} />
        <Stat label="Training time" value={`${ann.seconds.toFixed(0)} s`} hint={`LR ${lr.seconds.toFixed(0)} s · ANN ${int(annParams(77, [312, 289, 201]))} parameters`} />
        <Stat label="Optuna search" value={`${trials.length} trials`} hint={`${done.length} completed, ${pruned.length} pruned · best trial ${results.optuna.best_trial}`} />
      </div>

      <RoundFlowCard />

      <div className="grid-2">
        <Card title="Validation PR-AUC per round" subtitle={`Higher is better. Round 0 (untrained) is not drawn: ANN ${fixed(r0.val_pr_auc, 4)}, LR ${fixed(lr.history[0].val_pr_auc, 4)}.`}>
          <RoundChart
            data={merge([['ann', ann.history, 'val_pr_auc'], ['ann1', ann1.history, 'val_pr_auc'], ['lr', lr.history, 'val_pr_auc']])}
            lines={[
              { key: 'ann', name: `Federated ANN (${ann.local_epochs} local epochs)`, color: 'var(--s1)' },
              { key: 'ann1', name: 'Federated ANN (1 local epoch)', color: 'var(--s2)' },
              { key: 'lr', name: 'Federated Logistic Regression', color: 'var(--s3)' },
            ]}
            domain={[0.92, 1]} ticks={[0.92, 0.94, 0.96, 0.98, 1]} fmt={(v) => fixed(v, 2)}
          />
        </Card>
        <Card title="Validation F1 per round" subtitle="Attack-class F1 at the default threshold 0.5">
          <RoundChart
            data={merge([['ann', ann.history, 'val_f1'], ['lr', lr.history, 'val_f1']])}
            lines={[{ key: 'ann', name: 'Federated ANN', color: 'var(--s1)' }, { key: 'lr', name: 'Federated Logistic Regression', color: 'var(--s3)' }]}
            domain={[0.8, 1]} ticks={[0.8, 0.85, 0.9, 0.95, 1]} fmt={(v) => fixed(v, 2)}
          />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Local training loss of each client" subtitle="Federated ANN, mean loss of local training in each round (90% of each client's rows; 10% is kept for validation)">
          <RoundChart
            data={ann.history.filter((h) => h.round >= 1).map((h) => ({ round: h.round, A: h.clients.A, B: h.clients.B, C: h.clients.C }))}
            lines={[{ key: 'A', name: 'Client A', color: 'var(--s1)' }, { key: 'B', name: 'Client B', color: 'var(--s2)' }, { key: 'C', name: 'Client C', color: 'var(--s3)' }]}
            domain={['auto', 'auto']} fmt={(v) => fixed(v, 4)}
          />
          <p className="muted small">The three curves overlap because the clients hold IID shares of the same training pool ({int(client.rows)} flows each).</p>
        </Card>
        <Card title="Optuna search" subtitle="Validation PR-AUC of each trial. Weak trials are stopped early (pruned).">
          <Legend items={[{ label: `Completed (${done.length})`, color: 'var(--s1)' }, { label: `Pruned (${pruned.length})`, color: 'var(--s2)' }]} />
          <div className="chart">
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis type="number" dataKey="x" name="Trial" domain={[0, trials.length - 1]} tick={axis} tickLine={false} axisLine={{ stroke: 'var(--grid)' }} label={{ value: 'Trial number', position: 'insideBottom', offset: -2, fill: 'var(--ink-3)', fontSize: 12 }} height={40} />
                <YAxis type="number" dataKey="y" name="PR-AUC" domain={[0.9, 1]} tick={axis} tickLine={false} axisLine={false} tickFormatter={(v) => fixed(v, 2)} width={50} />
                <ZAxis range={[70, 70]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<ScatterTip />} />
                <Scatter name="Completed" data={done} fill="var(--s1)" stroke="var(--surface)" strokeWidth={2} isAnimationActive={false} />
                <Scatter name="Pruned" data={pruned} fill="var(--s2)" stroke="var(--surface)" strokeWidth={2} shape="diamond" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Best hyperparameters" subtitle={`Trial ${results.optuna.best_trial}, validation PR-AUC ${fixed(results.optuna.best_value, 4)}`}>
          <dl className="kv">
            <div><dt>Hidden layers</dt><dd>{p.layer_0_size}, {p.layer_1_size}, {p.layer_2_size}</dd></div>
            <div><dt>Dropout</dt><dd>{p.dropout.toFixed(4)}</dd></div>
            <div><dt>Batch normalization</dt><dd>{p.batch_norm ? 'yes' : 'no'}</dd></div>
            <div><dt>Optimizer</dt><dd>{p.optimizer.toUpperCase()}</dd></div>
            <div><dt>Learning rate</dt><dd>{p.learning_rate.toExponential(3)}</dd></div>
            <div><dt>Weight decay</dt><dd>{p.weight_decay.toExponential(3)}</dd></div>
            <div><dt>Batch size / epochs</dt><dd>{p.batch_size} / {p.epochs}</dd></div>
            <div><dt>Gradient clipping</dt><dd>{p.grad_clip ? `norm ${p.grad_clip_norm.toFixed(2)}` : 'off'}</dd></div>
            <div><dt>Class weighting</dt><dd>{p.class_weighting}</dd></div>
          </dl>
        </Card>
        <Card title="Per-round table" subtitle="Federated ANN, validation data (threshold 0.5)">
          <div className="table-wrap tall short">
            <table className="tbl">
              <thead><tr><th className="r">Round</th><th className="r">Loss</th><th className="r">PR-AUC</th><th className="r">F1</th><th className="r">Recall</th><th className="r">Missed</th></tr></thead>
              <tbody>
                {ann.history.map((h) => (
                  <tr key={h.round} className={h.round === ann.best_round ? 'emph' : ''}>
                    <td className="r">{h.round === 0 ? '0 (untrained)' : h.round}{h.round === ann.best_round && ' ★'}</td>
                    <td className="r">{fixed(h.val_loss, 4)}</td><td className="r">{fixed(h.val_pr_auc, 4)}</td><td className="r">{fixed(h.val_f1, 4)}</td>
                    <td className="r">{fixed(h.val_attack_recall, 4)}</td><td className="r">{int(h.val_false_negatives)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">★ marks the round whose weights were kept (best validation PR-AUC).</p>
        </Card>
      </div>
    </>
  )
}

function ScatterTip({ active, payload }) {
  if (!active || !payload?.length) return null
  const t = payload[0].payload
  return (
    <div className="tip">
      <div className="tip-title">Trial {t.trial} · {t.state.toLowerCase()}</div>
      <div className="tip-row"><span>PR-AUC</span><b>{fixed(t.pr_auc, 4)}</b></div>
      <div className="tip-row"><span>Layers / dropout</span><b>{t.layers} / {t.dropout?.toFixed(3)}</b></div>
      <div className="tip-row"><span>Optimizer / batch</span><b>{t.optimizer} / {t.batch}</b></div>
      <div className="tip-row"><span>Run time</span><b>{t.minutes} min</b></div>
    </div>
  )
}
