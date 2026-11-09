import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../api.js'
import { Card, Stat, PageTitle, Legend, ChartTip } from '../components/ui.jsx'
import { int, fixed, pct } from '../format.js'
import { LogBox } from './LiveTestPage.jsx'

const axis = { fill: 'var(--ink-2)', fontSize: 12 }
const CLIENTS = ['A', 'B', 'C']
const mib = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(2)} MiB` : b >= 1024 ? `${(b / 1024).toFixed(1)} KiB` : `${b} B`)
const mb = (b) => `${(b / 1e6).toFixed(0)} MB`

/* ---------------------------------------------------------------- job hook with fast polling */
function useRun() {
  const [job, setJob] = useState(null)
  const [err, setErr] = useState('')
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])
  const poll = useCallback((id) => {
    api.job(id).then((j) => { setJob(j); if (j.status === 'running') timer.current = setTimeout(() => poll(id), 600) }).catch((e) => setErr(e.message))
  }, [])
  const start = useCallback(async (body) => {
    setErr(''); setJob(null)
    try { const j = await api.startJob(body); setJob(j); poll(j.id) } catch (e) { setErr(e.message) }
  }, [poll])
  const stop = useCallback(async () => { if (job) { try { setJob(await api.stopJob(job.id)); poll(job.id) } catch (e) { setErr(e.message) } } }, [job, poll])
  return { job, err, start, stop, running: job?.status === 'running' }
}

/* ---------------------------------------------------------------- page */
export default function MonitorPage() {
  const run = useRun()
  const [model, setModel] = useState('logistic_regression')
  const [rounds, setRounds] = useState(3)
  const [epochs, setEpochs] = useState(1)
  const [comm, setComm] = useState(null)
  const [hist, setHist] = useState([])
  const [offline, setOffline] = useState('')

  const loadHist = useCallback(() => api.history().then(setHist).catch(() => {}), [])
  useEffect(() => {
    api.comm().then(setComm).catch((e) => setOffline(e.message))
    loadHist()
  }, [loadHist])
  const status = run.job?.status
  useEffect(() => { if (status && status !== 'running') loadHist() }, [status, loadHist])

  const live = run.job?.rounds_live ?? []
  const R = run.job?.params?.rounds ?? rounds
  const key = run.job?.params?.model ?? model

  const series = useMemo(() => {
    const val = live.filter((r) => r.val).map((r) => ({ round: r.round, 'Loss': r.val.val_loss, 'PR-AUC': r.val.pr_auc, 'F1': r.val.f1, 'Accuracy': r.val.accuracy, 'Recall': r.val.recall, FN: r.val.fn, FP: r.val.fp }))
    const cli = live.filter((r) => r.fit).map((r) => ({ round: r.round, ...Object.fromEntries(r.fit.clients.map((c) => [`Client ${c.client}`, c.train_loss])) }))
    return { val, cli }
  }, [live])
  const lastVal = [...live].reverse().find((r) => r.val)?.val
  const done = live.filter((r) => r.val && r.round > 0).length

  if (offline) {
    return (
      <>
        <PageTitle title="Training monitor">This page needs the Python backend.</PageTitle>
        <Card title="Backend is not running" subtitle={offline}>
          <pre className="code">{`source venv/bin/activate
uvicorn backend.main:app --port 8000`}</pre>
        </Card>
      </>
    )
  }

  return (
    <>
      <PageTitle title="Training monitor">
        Start a short federated run (FedAvg over clients A, B and C), watch every round as it happens and stop it at any time. Runs write only to a temporary folder; your saved models and results are never changed.
      </PageTitle>

      <Card title="Run controls" subtitle="Real training, not a simulation of the screen: each step below is read from the running Flower server.">
        <div className="controls">
          <label>Model
            <select value={model} disabled={run.running} onChange={(e) => setModel(e.target.value)}>
              <option value="logistic_regression">Logistic Regression (about 1.5 s per round)</option>
              <option value="ann">ANN (about 5 s per round)</option>
            </select>
          </label>
          <label>Rounds <b>{rounds}</b><input type="range" min="1" max="5" value={rounds} disabled={run.running} onChange={(e) => setRounds(+e.target.value)} /></label>
          <label>Local epochs <b>{epochs}</b><input type="range" min="1" max="2" value={epochs} disabled={run.running} onChange={(e) => setEpochs(+e.target.value)} /></label>
          {!run.running
            ? <button className="btn primary" onClick={() => run.start({ kind: 'federated', model, rounds, local_epochs: epochs })}>▶ Start training</button>
            : <button className="btn danger" onClick={run.stop}>■ Stop</button>}
        </div>
        {run.err && <p className="error">{run.err}</p>}
        {run.job && (
          <p className={status === 'done' ? 'ok-line' : status === 'failed' ? 'error' : 'muted small'}>
            {status === 'running' && `Running… ${Math.round(run.job.elapsed || 0)} s · ${done} of ${R} rounds finished`}
            {status === 'done' && `✓ Finished ${run.job.result?.rounds} rounds in ${run.job.result?.seconds?.toFixed(1)} s · best round ${run.job.result?.best_round} (validation PR-AUC ${fixed(run.job.result?.best_val_pr_auc, 4)})`}
            {status === 'stopped' && 'Stopped by you. The rounds that finished are shown below.'}
            {status === 'failed' && 'The run failed. See the log at the bottom of the page.'}
          </p>
        )}
      </Card>

      <Card title="Live federated round" subtitle="Blue: the server sends the global weights and each client trains on its own data. Orange: clients return weights and sample counts. Green: the server validates the new global model.">
        <FedAnimation job={run.job} R={R} />
      </Card>

      <div className="stat-grid">
        <Stat label="Round" value={done ? `${done} / ${R}` : `– / ${R}`} hint={run.running ? 'in progress' : 'finished rounds'} />
        <Stat label="Validation loss" value={lastVal ? fixed(lastVal.val_loss, 4) : '–'} hint="global model, lower is better" />
        <Stat label="Validation F1 (attack)" value={lastVal ? pct(lastVal.f1) : '–'} hint={lastVal ? `accuracy ${pct(lastVal.accuracy)}` : 'threshold 0.5'} />
        <Stat label="Missed attacks" value={lastVal ? int(lastVal.fn) : '–'} hint={lastVal ? `${int(lastVal.fp)} false alarms` : 'on 178,545 validation flows'} />
      </div>

      <div className="grid-2">
        <Card title="Global model: loss and PR-AUC" subtitle="Validation scores after each round (round 0 is the untrained model).">
          <Chart data={series.val} R={R} lines={[{ k: 'Loss', c: 'var(--s2)' }, { k: 'PR-AUC', c: 'var(--s1)' }]} />
        </Card>
        <Card title="Global model: accuracy, F1 and recall" subtitle="Threshold 0.5, attack class is positive.">
          <Chart data={series.val} R={R} lines={[{ k: 'Accuracy', c: 'var(--s1)' }, { k: 'F1', c: 'var(--s3)' }, { k: 'Recall', c: 'var(--s4)' }]} />
        </Card>
      </div>
      <div className="grid-2">
        <Card title="Client-wise training loss" subtitle="Mean loss of each client’s local training in that round, from its own data.">
          <Chart data={series.cli} R={R} from={1} lines={CLIENTS.map((c, i) => ({ k: `Client ${c}`, c: ['var(--s1)', 'var(--s2)', 'var(--s3)'][i] }))} />
          <ClientTable live={live} />
        </Card>
        <Card title="Per-round results" subtitle="Validation data, 178,545 flows.">
          <div className="table-wrap tall short">
            <table className="tbl">
              <thead><tr><th className="r">Round</th><th className="r">Loss</th><th className="r">Acc.</th><th className="r">F1</th><th className="r">PR-AUC</th><th className="r">Missed</th><th className="r">False alarms</th></tr></thead>
              <tbody>
                {series.val.map((r) => (
                  <tr key={r.round}><td className="r">{r.round === 0 ? '0 (untrained)' : r.round}</td><td className="r">{fixed(r.Loss, 4)}</td><td className="r">{pct(r.Accuracy)}</td><td className="r">{pct(r.F1)}</td><td className="r">{fixed(r['PR-AUC'], 4)}</td><td className="r">{int(r.FN)}</td><td className="r">{int(r.FP)}</td></tr>
                ))}
                {!series.val.length && <tr><td colSpan={7} className="muted">Start a run to see results.</td></tr>}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <Confusion val={lastVal} />
      <Privacy comm={comm} modelKey={key} roundsDone={done} />
      <History hist={hist} onClear={() => api.clearHistory().then(loadHist)} />
      {run.job && <Card title="Server log"><LogBox lines={run.job.log.filter((l) => l.trim() && !/logging\.cc|raylet|\(raylet\)/.test(l)).slice(-30)} /></Card>}
    </>
  )
}

/* ---------------------------------------------------------------- charts and tables */
function Chart({ data, lines, R, from = 0 }) {
  return (
    <>
      <Legend items={lines.map((l) => ({ label: l.k, color: l.c }))} />
      <div className="chart">
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={data} margin={{ top: 6, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="round" type="number" domain={[from, Math.max(from + 1, R)]} allowDecimals={false} tick={axis} tickLine={false} axisLine={{ stroke: 'var(--grid)' }} />
            <YAxis domain={['auto', 'auto']} tick={axis} tickLine={false} axisLine={false} width={50} tickFormatter={(v) => v.toFixed(2)} />
            <Tooltip content={<ChartTip labelPrefix="Round " format={(v) => fixed(v, 4)} />} />
            {lines.map((l) => <Line key={l.k} dataKey={l.k} stroke={l.c} strokeWidth={2} dot={{ r: 3, strokeWidth: 2, stroke: 'var(--surface)', fill: l.c }} isAnimationActive={false} connectNulls />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  )
}

function ClientTable({ live }) {
  const fits = live.filter((r) => r.fit)
  if (!fits.length) return <p className="muted small">Client losses appear after the first round.</p>
  const last = fits[fits.length - 1].fit
  return (
    <div className="table-wrap">
      <table className="tbl">
        <thead><tr><th>Client</th><th className="r">Training rows</th><th className="r">Weight in FedAvg</th><th className="r">Loss (round {fits[fits.length - 1].round})</th></tr></thead>
        <tbody>
          {last.clients.map((c) => {
            const total = last.clients.reduce((a, x) => a + x.num_examples, 0)
            return <tr key={c.client}><td>Client {c.client}</td><td className="r">{int(c.num_examples)}</td><td className="r">{pct(c.num_examples / total, 1)}</td><td className="r">{fixed(c.train_loss, 4)}</td></tr>
          })}
        </tbody>
      </table>
    </div>
  )
}

function Confusion({ val }) {
  if (!val) return null
  const cells = [[val.tn, val.fp], [val.fn, val.tp]]
  const tot = [val.tn + val.fp, val.fn + val.tp]
  return (
    <Card title="Confusion matrix of the global model" subtitle="Latest round, validation flows, threshold 0.5. Rows are the true class.">
      <div className="cm-single">
        <div className="cm-table">
          <span /><span className="cm-h">Pred. benign</span><span className="cm-h">Pred. attack</span>
          {['Benign', 'Attack'].map((l, i) => (
            <FragmentRow key={l} label={l} cells={cells[i]} total={tot[i]} okIndex={i} />
          ))}
        </div>
        <dl className="kv">
          <div><dt>Precision</dt><dd>{pct(val.precision)}</dd></div><div><dt>Recall</dt><dd>{pct(val.recall)}</dd></div>
          <div><dt>F1 (attack)</dt><dd>{pct(val.f1)}</dd></div><div><dt>ROC-AUC</dt><dd>{fixed(val.roc_auc, 4)}</dd></div>
        </dl>
      </div>
    </Card>
  )
}
function FragmentRow({ label, cells, total, okIndex }) {
  return (
    <>
      <span className="cm-h left">True {label.toLowerCase()}</span>
      {cells.map((v, j) => {
        const ok = j === okIndex, share = total ? v / total : 0
        return <div key={j} className={`cm-cell ${ok ? 'ok' : 'err'}`} style={{ '--a': ok ? 0.1 + share * 0.3 : Math.min(0.1 + share * 40, 0.5) }}><b>{int(v)}</b><span>{pct(share, 2)}</span></div>
      })}
    </>
  )
}

/* ---------------------------------------------------------------- privacy and communication */
function Privacy({ comm, modelKey, roundsDone }) {
  if (!comm) return null
  const m = comm.models[modelKey] ?? comm.models.lr
  const perRound = m.bytes_per_client_per_round_each_way
  const raw = comm.clients[0].raw_feature_bytes
  const sentUp = perRound * 3 * roundsDone
  return (
    <Card title="Privacy and communication monitor" subtitle={`Computed from the ${m.name} weights and the client files, not estimated.`}>
      <div className="stat-grid">
        <Stat label="Training rows sent to server" value="0" hint="only weights and counts leave a client" tone="good" />
        <Stat label="Sent per client per round" value={mib(perRound)} hint={`${int(m.values)} values × 4 bytes, each way`} />
        <Stat label="Sent so far (this run)" value={mib(sentUp)} hint={`3 clients × ${roundsDone} round(s), uploads`} />
        <Stat label="Data kept on each client" value={mb(raw)} hint={`${int(comm.clients[0].train_rows)} rows × ${comm.n_features} features`} />
      </div>
      <p className="muted small">One round sends about {((raw / perRound)).toFixed(0)} times less data than the client’s feature data (as float32), and the rows themselves never move.</p>
      <div className="grid-2">
        <div>
          <h4 className="sub">Local versus global model</h4>
          <div className="table-wrap">
            <table className="tbl">
              <thead><tr><th>Parameter array</th><th className="r">Shape</th><th className="r">Values</th></tr></thead>
              <tbody>
                {m.layers.map((l) => <tr key={l.name}><td className="mono">{l.name}</td><td className="r">{l.shape.join(' × ')}</td><td className="r">{int(l.values)}</td></tr>)}
                <tr className="emph"><td><b>Total (identical on every client and the server)</b></td><td /><td className="r"><b>{int(m.values)}</b></td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h4 className="sub">What each client contributes</h4>
          <div className="table-wrap">
            <table className="tbl">
              <thead><tr><th>Client</th><th className="r">Local rows</th><th className="r">Training rows</th><th className="r">FedAvg weight</th></tr></thead>
              <tbody>{comm.clients.map((c, i) => <tr key={c.client}><td>Client {c.client}</td><td className="r">{int(c.rows)}</td><td className="r">{int(c.train_rows)}</td><td className="r">{pct(comm.weights[i].share, 1)}</td></tr>)}</tbody>
            </table>
          </div>
          <ul className="notes small">{comm.notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </div>
      </div>
    </Card>
  )
}

/* ---------------------------------------------------------------- run history */
function History({ hist, onClear }) {
  const fed = hist.filter((h) => h.kind === 'federated')
  return (
    <Card title="Run history" subtitle="Every run started from this page is kept in backend/.runtime/history.json (not in git)."
      right={fed.length ? <button className="btn" onClick={onClear}>Clear history</button> : null}>
      {!fed.length ? <div className="empty">No runs yet.</div> : (
        <div className="table-wrap">
          <table className="tbl">
            <thead><tr><th>Started</th><th>Model</th><th className="r">Rounds × epochs</th><th>Status</th><th className="r">Best round</th><th className="r">Val PR-AUC</th><th className="r">Final F1</th><th className="r">Missed</th><th className="r">Time</th></tr></thead>
            <tbody>
              {fed.map((h) => (
                <tr key={h.id}>
                  <td>{new Date(h.started * 1000).toLocaleString()}</td><td>{h.params.model === 'ann' ? 'ANN' : 'Logistic Regression'}</td>
                  <td className="r">{h.params.rounds} × {h.params.local_epochs}</td><td>{h.status}</td>
                  <td className="r">{h.result?.best_round ?? '–'}</td><td className="r">{h.result ? fixed(h.result.best_val_pr_auc, 4) : '–'}</td>
                  <td className="r">{h.result ? pct(h.result.final.val_f1) : '–'}</td><td className="r">{h.result ? int(h.result.final.val_false_negatives) : '–'}</td>
                  <td className="r muted">{h.seconds ? `${h.seconds.toFixed(1)} s` : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

/* ---------------------------------------------------------------- animation (driven by real server events) */
function steps(R) {
  const out = []
  out.push({ r: 0, s: 'init' })
  for (let r = 1; r <= R; r++) out.push({ r, s: 'train' }, { r, s: 'agg' }, { r, s: 'val' })
  return out
}

function FedAnimation({ job, R }) {
  const list = useMemo(() => steps(R), [R])
  const live = job?.rounds_live ?? []
  const avail = (st) => {
    const row = live.find((x) => x.round === st.r)
    if (st.s === 'init') return !!row?.val
    if (st.s === 'train') return !!row?.fit || (job?.phase?.round === st.r && job.phase.event === 'start') || job?.phase?.round > st.r
    if (st.s === 'agg') return !!row?.fit
    return !!row?.val
  }
  let latest = -1
  list.forEach((st, i) => { if (avail(st)) latest = i })
  const [idx, setIdx] = useState(-1)
  const prevId = useRef(null)
  useEffect(() => { if (job?.id !== prevId.current) { prevId.current = job?.id; setIdx(-1) } }, [job?.id])
  // play the real events one after another so that each step stays visible for a moment, even when rounds are fast
  useEffect(() => {
    if (idx >= latest) return
    const t = setTimeout(() => setIdx((i) => Math.min(i + 1, latest)), 650)
    return () => clearTimeout(t)
  }, [idx, latest])

  const cur = idx >= 0 ? list[idx] : null
  const finished = job && job.status !== 'running' && idx >= latest
  const active = cur && !finished ? cur.s : null
  const row = cur ? live.find((x) => x.round === cur.r) : null
  const fitNow = cur && (cur.s === 'agg' || cur.s === 'val') && row?.fit
  const label = !job ? 'Press “Start training”.'
    : !cur ? 'Starting the Flower simulation…'
    : cur.s === 'init' ? 'Round 0: the server scores the untrained global model.'
    : cur.s === 'train' ? `Round ${cur.r} of ${R}: the server sends the global weights; each client trains locally on its own rows.`
    : cur.s === 'agg' ? `Round ${cur.r} of ${R}: clients return weights and sample counts; the server averages them (FedAvg).`
    : `Round ${cur.r} of ${R}: the server validates the new global model.`

  const cy = [60, 150, 240]
  const lossOf = (c) => row?.fit?.clients.find((x) => x.client === c)
  return (
    <div className="anim">
      <svg viewBox="0 0 700 300" role="img" aria-label="Federated round animation">
        <defs>
          <marker id="ah-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--s1)" /></marker>
          <marker id="ah-o" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--s2)" /></marker>
          <marker id="ah-g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--s3)" /></marker>
        </defs>
        {CLIENTS.map((c, i) => {
          const f = lossOf(c)
          return (
            <g key={c}>
              <rect x="10" y={cy[i] - 38} width="170" height="76" rx="12" className={`node ${active === 'train' ? 'busy' : ''}`} />
              <text x="95" y={cy[i] - 14} className="t-title" textAnchor="middle">Client {c}</text>
              <text x="95" y={cy[i] + 4} className="t-sub" textAnchor="middle">{active === 'train' ? 'training locally…' : 'data stays here'}</text>
              <text x="95" y={cy[i] + 24} className="t-sub" textAnchor="middle">{fitNow && f ? `loss ${f.train_loss.toFixed(4)} · ${int(f.num_examples)} rows` : '535,633 rows'}</text>
              {/* server -> client (broadcast), then client -> server (upload): two parallel lines per client */}
              <line x1="268" y1={150 + (i - 1) * 22 - 7} x2="186" y2={cy[i] - 7} className={`arrow blue ${active === 'train' ? 'flow' : ''}`} markerEnd="url(#ah-b)" />
              <line x1="186" y1={cy[i] + 9} x2="268" y2={150 + (i - 1) * 22 + 9} className={`arrow orange ${active === 'agg' ? 'flow' : ''}`} markerEnd="url(#ah-o)" />
            </g>
          )
        })}
        <rect x="270" y="95" width="150" height="110" rx="14" className={`node server ${active === 'agg' ? 'busy' : ''}`} />
        <text x="345" y="125" className="t-title" textAnchor="middle">Server</text>
        <text x="345" y="146" className="t-sub" textAnchor="middle">FedAvg</text>
        <text x="345" y="164" className="t-sub" textAnchor="middle">{active === 'agg' ? 'averaging weights…' : 'weights only'}</text>
        <text x="345" y="184" className="t-sub" textAnchor="middle">{cur ? `round ${cur.r} / ${R}` : `0 / ${R}`}</text>
        <line x1="424" y1="150" x2="498" y2="150" className={`arrow green ${active === 'val' || active === 'init' ? 'flow' : ''}`} markerEnd="url(#ah-g)" />
        <rect x="500" y="95" width="190" height="110" rx="14" className={`node val ${active === 'val' || active === 'init' ? 'busy' : ''}`} />
        <text x="595" y="122" className="t-title" textAnchor="middle">Validation</text>
        <text x="595" y="142" className="t-sub" textAnchor="middle">178,545 flows</text>
        {row?.val ? (
          <>
            <text x="595" y="166" className="t-big" textAnchor="middle">F1 {(row.val.f1 * 100).toFixed(1)}%</text>
            <text x="595" y="186" className="t-sub" textAnchor="middle">PR-AUC {row.val.pr_auc.toFixed(4)} · missed {int(row.val.fn)}</text>
          </>
        ) : <text x="595" y="170" className="t-sub" textAnchor="middle">waiting…</text>}
      </svg>
      <div className="anim-label" aria-live="polite">{label}</div>
      <div className="anim-steps" aria-hidden>
        {list.map((s, i) => <span key={i} className={`dot ${i < idx ? 'past' : i === idx ? 'now' : ''} ${s.s}`} title={`round ${s.r}: ${s.s}`} />)}
      </div>
    </div>
  )
}
