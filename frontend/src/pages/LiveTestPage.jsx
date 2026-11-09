import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api.js'
import { Card, Chip, Stat, PageTitle, Segmented } from '../components/ui.jsx'
import { cell, int, pct, fixed } from '../format.js'

const MODEL_ORDER = ['Federated ANN', 'Centralized ANN', 'Centralized RF', 'Federated LR']
const COLOR = { 'Federated ANN': 'var(--s1)', 'Centralized ANN': 'var(--s2)', 'Centralized RF': 'var(--s3)', 'Federated LR': 'var(--s4)' }
const KEY_FEATURES = ['Protocol', 'Flow Duration', 'Total Fwd Packets', 'Flow Packets/s']

export { useJob, LogBox }
export default function LiveTestPage() {
  const [health, setHealth] = useState(null)
  const [err, setErr] = useState('')
  const [bank, setBank] = useState(null)
  const [thr, setThr] = useState({}) // per-model decision thresholds (start at the ones saved with the project)

  useEffect(() => {
    let live = true
    Promise.all([api.health(), api.samples()])
      .then(([h, s]) => {
        if (!live) return
        setHealth(h); setBank(s)
        setThr(Object.fromEntries(Object.entries(h.loaded).map(([k, v]) => [k, v.threshold])))
      })
      .catch((e) => live && setErr(e.message))
    return () => { live = false }
  }, [])

  if (err) return <Offline message={err} />
  if (!health || !bank) return <p className="muted">Connecting to the backend…</p>

  return (
    <>
      <PageTitle title="Live test">
        Score real network flows with the trained models, change the decision thresholds, re-run a quick evaluation and start short jobs. Every number here is computed by the Python backend when you press a button.
      </PageTitle>
      <div className="stat-grid">
        <Stat label="Backend" value="Online" tone="good" hint={`torch ${health.torch} · scikit-learn ${health.sklearn}`} />
        <Stat label="Models loaded" value={`${Object.keys(health.loaded).length} / 4`} hint={Object.keys(health.errors).length ? 'see errors in the API log' : 'RF, centralized ANN, federated ANN, federated LR'} />
        <Stat label="Sample flows" value={health.n_samples} hint="real flows from the held-out test file" />
        <Stat label="Features per flow" value={health.n_features} hint="same 77 columns as the dataset" />
      </div>
      <Thresholds thr={thr} setThr={setThr} saved={health.loaded} />
      <TrySample bank={bank} thr={thr} />
      <BatchTest bank={bank} thr={thr} />
      <QuickEval thr={thr} />
      <UploadFlows bank={bank} thr={thr} />
      <Jobs />
      <Card title="What cannot be re-run here" subtitle="By design">
        <ul className="notes">
          <li>The held-out test set is scored once per model (see Results). The sample flows above are single-row demonstrations; no score from them is used to choose a model or a threshold.</li>
          <li>Long tasks are not offered here: the Optuna search takes about 47 minutes and a full 15-round federated ANN run about 2.5 minutes. Use <code>python run_project.py</code> for those.</li>
        </ul>
      </Card>
    </>
  )
}

function Offline({ message }) {
  return (
    <>
      <PageTitle title="Live test">This page needs the Python backend.</PageTitle>
      <Card title="Backend is not running" subtitle={message}>
        <p className="muted small">Open a second terminal in the project folder and run:</p>
        <pre className="code">{`source venv/bin/activate
uvicorn backend.main:app --port 8000`}</pre>
        <p className="muted small">Then reload this page. The other pages work without the backend.</p>
      </Card>
    </>
  )
}

/* ---------------------------------------------------------------- thresholds */
function Thresholds({ thr, setThr, saved }) {
  const names = MODEL_ORDER.filter((m) => saved[m])
  return (
    <Card title="Decision thresholds" subtitle="A flow is called an attack when the model’s attack probability is at or above its threshold. Drag a slider: the tests below use these values.">
      <div className="thr-grid">
        {names.map((m) => (
          <div className="thr" key={m}>
            <div className="thr-head">
              <span><span className="swatch" style={{ background: COLOR[m] }} />{m}</span>
              <b>{thr[m]?.toFixed(3)}</b>
            </div>
            <input type="range" min="0.01" max="0.99" step="0.01" value={thr[m] ?? 0.5} onChange={(e) => setThr({ ...thr, [m]: +e.target.value })} aria-label={`Threshold for ${m}`} />
            <div className="thr-foot muted small">
              saved: {saved[m].threshold.toFixed(3)}
              {Math.abs(thr[m] - saved[m].threshold) > 0.0005 && <button className="link" onClick={() => setThr({ ...thr, [m]: saved[m].threshold })}>reset</button>}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

/* ---------------------------------------------------------------- single sample */
function TrySample({ bank, thr }) {
  const [filter, setFilter] = useState('all')
  const [selId, setSelId] = useState(null)
  const [edits, setEdits] = useState({})
  const [q, setQ] = useState('')
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [showFeat, setShowFeat] = useState(false)

  const list = bank.samples.filter((s) => filter === 'all' || (filter === 'normal') === (s.label === 0))
  const sel = bank.samples.find((s) => s.id === selId)
  const values = useMemo(() => (sel ? { ...sel.features, ...edits } : null), [sel, edits])
  const edited = Object.keys(edits).length

  const run = useCallback(async (id, ed, th) => {
    const s = bank.samples.find((x) => x.id === id)
    if (!s) return
    setBusy(true); setErr('')
    try {
      const body = Object.keys(ed).length ? { features: { ...s.features, ...ed }, thresholds: th } : { sample_id: id, thresholds: th }
      setRes(await api.predict(body))
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }, [bank])

  const pick = (s) => { setSelId(s.id); setEdits({}); setRes(null); setErr('') }
  // thresholds changed while a result is on screen: re-score immediately (cheap)
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }
    if (selId != null && res) { const t = setTimeout(() => run(selId, edits, thr), 200); return () => clearTimeout(t) }
  }, [thr]) // eslint-disable-line react-hooks/exhaustive-deps

  const shown = bank.feature_names.filter((f) => f.toLowerCase().includes(q.trim().toLowerCase()))

  return (
    <Card
      title="1 · Test one flow"
      subtitle="Pick a flow from the list (real rows, some normal, some attacks), optionally change feature values, then press Test."
      right={<Segmented value={filter} onChange={setFilter} options={[{ value: 'all', label: `All ${bank.samples.length}` }, { value: 'normal', label: 'Normal' }, { value: 'attack', label: 'Attack' }]} />}
    >
      <div className="try">
        <ul className="picker" role="listbox" aria-label="Sample flows">
          {list.map((s) => (
            <li key={s.id}>
              <button className={`pick ${s.id === selId ? 'on' : ''}`} role="option" aria-selected={s.id === selId} onClick={() => pick(s)}>
                <span className="pick-top"><Chip attack={!!s.label}>{s.type}</Chip><span className="mono muted">#{s.id.toLocaleString('en-US')}</span></span>
                <span className="pick-facts">
                  {KEY_FEATURES.map((k) => <span key={k}><i>{k}</i>{cell(s.features[k])}</span>)}
                </span>
              </button>
            </li>
          ))}
        </ul>

        <div className="try-main">
          {!sel && <div className="empty">← Choose a flow, then press “Test this flow”. Its true class is hidden until you test it.</div>}
          {sel && (
            <>
              <div className="try-head">
                <div>
                  <div className="try-title">Flow <span className="mono">#{sel.id.toLocaleString('en-US')}</span></div>
                  <div className="muted small">Traffic type and true label are revealed with the result{edited ? ' (not shown once you edit values: the label no longer applies)' : ''}.</div>
                </div>
                <div className="try-actions">
                  <button className="btn" onClick={() => setShowFeat(!showFeat)}>{showFeat ? 'Hide' : 'Edit'} features{edited ? ` (${edited} changed)` : ''}</button>
                  {edited > 0 && <button className="btn" onClick={() => { setEdits({}); setRes(null) }}>Reset</button>}
                  <button className="btn primary" disabled={busy} onClick={() => run(sel.id, edits, thr)}>{busy ? 'Testing…' : 'Test this flow'}</button>
                </div>
              </div>
              {err && <p className="error">{err}</p>}
              {showFeat && (
                <div className="feat">
                  <input className="search" placeholder="Filter features…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter features" />
                  <div className="table-wrap tall short">
                    <table className="tbl">
                      <thead><tr><th>Feature</th><th className="r">Value</th></tr></thead>
                      <tbody>
                        {shown.map((f) => (
                          <tr key={f} className={f in edits ? 'emph' : ''}>
                            <td>{f}</td>
                            <td className="r">
                              <input className="num" type="number" step="any" value={values[f] ?? ''} aria-label={f}
                                onChange={(e) => { const v = e.target.value; setEdits((p) => { const n = { ...p }; if (v === '' || +v === sel.features[f]) delete n[f]; else n[f] = +v; return n }) }} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              {res && <Verdicts res={res} sel={sel} edited={edited > 0} />}
            </>
          )}
        </div>
      </div>
    </Card>
  )
}

function Verdicts({ res, sel, edited }) {
  const truth = edited ? null : res.truth
  return (
    <div className="verdicts">
      <div className="truth">
        {truth == null ? <span className="muted">Edited flow: no ground truth.</span> : (
          <>True class: <Chip attack={truth === 1}>{truth === 1 ? 'Attack' : 'Benign'}</Chip> · traffic type <b>{sel.type}</b></>
        )}
      </div>
      <div className="vgrid">
        {MODEL_ORDER.filter((m) => res.models[m]).map((m) => {
          const r = res.models[m]
          const ok = truth == null ? null : r.pred === truth
          return (
            <div key={m} className={`vcard ${ok === false ? 'wrong' : ''}`}>
              <div className="vhead"><span><span className="swatch" style={{ background: COLOR[m] }} />{m}</span><span className="muted small">{r.ms.toFixed(1)} ms</span></div>
              <div className="vmain">
                <Chip attack={r.pred === 1}>{r.verdict}</Chip>
                {ok != null && <span className={`badge ${ok ? 'ok' : 'bad'}`}>{ok ? '✓ correct' : '✗ wrong'}</span>}
              </div>
              <div className="prob"><span>P(attack)</span><b>{(r.prob * 100).toFixed(2)}%</b></div>
              <div className="meter" aria-hidden>
                <span className="meter-fill" style={{ width: `${r.prob * 100}%`, background: COLOR[m] }} />
                <span className="meter-thr" style={{ left: `${r.threshold * 100}%` }} title={`threshold ${r.threshold.toFixed(3)}`} />
              </div>
              <div className="muted small">threshold {r.threshold.toFixed(3)}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- all samples */
function BatchTest({ bank, thr }) {
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const run = useCallback(async (th) => {
    setBusy(true); setErr('')
    try { setRes(await api.predictBatch({ thresholds: th })) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }, [])
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }
    if (res) { const t = setTimeout(() => run(thr), 250); return () => clearTimeout(t) }
  }, [thr]) // eslint-disable-line react-hooks/exhaustive-deps
  const byId = Object.fromEntries(bank.samples.map((s) => [s.id, s]))
  return (
    <Card title="2 · Test all sample flows" subtitle="Scores the whole list with the four models at the thresholds above and compares each verdict with the true class."
      right={<button className="btn primary" disabled={busy} onClick={() => run(thr)}>{busy ? 'Testing…' : `Test all ${bank.samples.length} flows`}</button>}>
      {err && <p className="error">{err}</p>}
      {!res && !err && <div className="empty">Press the button to score every sample.</div>}
      {res && (
        <>
          <div className="stat-grid">
            {MODEL_ORDER.filter((m) => res.summary[m]).map((m) => {
              const s = res.summary[m]
              return <Stat key={m} label={m} value={`${s.correct} / ${s.total}`} hint={`${s.fn} missed attacks · ${s.fp} false alarms`} />
            })}
          </div>
          <p className="muted small">All {res.ids.length} flows scored in {res.total_ms.toFixed(0)} ms. This list has more attacks (and more rare attack types) than real traffic, so these counts are not the project’s accuracy: see Results for the full held-out test.</p>
          <div className="table-wrap tall">
            <table className="tbl">
              <thead><tr><th>Flow</th><th>True class</th>{MODEL_ORDER.map((m) => <th key={m} className="r">{m}</th>)}</tr></thead>
              <tbody>
                {res.ids.map((id, i) => (
                  <tr key={id}>
                    <td><span className="mono muted">#{id.toLocaleString('en-US')}</span> {byId[id].type}</td>
                    <td><Chip attack={res.truth[i] === 1}>{res.truth[i] === 1 ? 'Attack' : 'Benign'}</Chip></td>
                    {MODEL_ORDER.map((m) => {
                      const r = res.rows[m]; const ok = r.pred[i] === res.truth[i]
                      return <td key={m} className={`r ${ok ? '' : 'bad-cell'}`}>{ok ? '✓' : '✗'} <span className="muted">{(r.prob[i] * 100).toFixed(1)}%</span></td>
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  )
}

/* ---------------------------------------------------------------- quick evaluation */
function QuickEval({ thr }) {
  const [n, setN] = useState(20000)
  const [seed, setSeed] = useState(1)
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const run = useCallback(async (nn, sd, th) => {
    setBusy(true); setErr('')
    try { setRes(await api.quickEval({ n: nn, seed: sd, thresholds: th })) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }, [])
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }
    if (res) { const t = setTimeout(() => run(n, seed, thr), 250); return () => clearTimeout(t) }
  }, [thr]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Card title="3 · Quick re-evaluation on validation rows"
      subtitle="Draws N random validation rows, scores them with all four models and computes the metrics again. After the first run the scores are cached, so moving a threshold slider updates the table at once."
      right={<button className="btn primary" disabled={busy} onClick={() => run(n, seed, thr)}>{busy ? 'Running…' : 'Run evaluation'}</button>}>
      <div className="controls">
        <label>Rows <b>{int(n)}</b>
          <input type="range" min="1000" max="100000" step="1000" value={n} onChange={(e) => setN(+e.target.value)} />
        </label>
        <label>Random seed
          <input className="num" type="number" min="0" value={seed} onChange={(e) => setSeed(Math.max(0, +e.target.value || 0))} />
        </label>
      </div>
      {err && <p className="error">{err}</p>}
      {!res && !err && <div className="empty">Choose the number of rows and press Run evaluation.</div>}
      {res && (
        <>
          <p className="muted small">{int(res.n)} of {int(res.validation_rows_total)} validation rows ({int(res.attacks)} attacks, {int(res.benign)} benign) · finished in {res.seconds.toFixed(2)} s</p>
          <div className="table-wrap">
            <table className="tbl">
              <thead><tr><th>Model</th><th className="r">Threshold</th><th className="r">Accuracy</th><th className="r">Precision</th><th className="r">Recall</th><th className="r">F1</th><th className="r">ROC-AUC</th><th className="r">Missed</th><th className="r">False alarms</th><th className="r">Time</th></tr></thead>
              <tbody>
                {MODEL_ORDER.filter((m) => res.models[m]).map((m) => {
                  const r = res.models[m], x = r.metrics
                  return (
                    <tr key={m}>
                      <td><span className="swatch" style={{ background: COLOR[m] }} />{m}{r.saw_these_rows_in_training && <sup title="The Random Forest was trained on these rows, so its score is optimistic"> *</sup>}</td>
                      <td className="r">{r.threshold.toFixed(3)}</td><td className="r">{pct(x.accuracy)}</td><td className="r">{pct(x.precision)}</td><td className="r">{pct(x.recall)}</td>
                      <td className="r"><b>{pct(x.f1)}</b></td><td className="r">{fixed(x.roc_auc, 4)}</td><td className="r">{int(x.fn)}</td><td className="r">{int(x.fp)}</td>
                      <td className="r muted">{r.cached ? 'cached' : `${r.seconds.toFixed(2)} s`}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="muted small">* {res.note}</p>
          <div className="cm-grid">
            {MODEL_ORDER.filter((m) => res.models[m]).map((m) => <ConfusionMini key={m} name={m} x={res.models[m].metrics} />)}
          </div>
        </>
      )}
    </Card>
  )
}

function ConfusionMini({ name, x }) {
  const rows = [
    { l: 'Benign', total: x.tn + x.fp, cells: [{ v: x.tn, ok: true }, { v: x.fp, ok: false }] },
    { l: 'Attack', total: x.fn + x.tp, cells: [{ v: x.fn, ok: false }, { v: x.tp, ok: true }] },
  ]
  return (
    <figure className="cm">
      <figcaption><span className="swatch" style={{ background: COLOR[name] }} /> {name}</figcaption>
      <div className="cm-table">
        <span /><span className="cm-h">Pred. benign</span><span className="cm-h">Pred. attack</span>
        {rows.map((r) => (
          <Fragment key={r.l}>
            <span className="cm-h left">True {r.l.toLowerCase()}</span>
            {r.cells.map((c, i) => {
              const share = r.total ? c.v / r.total : 0
              return (
                <div key={i} className={`cm-cell ${c.ok ? 'ok' : 'err'}`} style={{ '--a': c.ok ? 0.1 + share * 0.3 : Math.min(0.1 + share * 40, 0.5) }}>
                  <b>{int(c.v)}</b><span>{pct(share, 2)}</span>
                </div>
              )
            })}
          </Fragment>
        ))}
      </div>
    </figure>
  )
}

/* ---------------------------------------------------------------- upload your own flows */
function parseCsv(text) {
  const lines = text.replace(/\r/g, '').split('\n').filter((l) => l.trim())
  if (lines.length < 2) throw new Error('The file needs a header row and at least one data row.')
  const head = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''))
  return { head, rows: lines.slice(1).map((l) => l.split(',').map((v) => v.trim().replace(/^"|"$/g, ''))) }
}

function UploadFlows({ bank, thr }) {
  const [res, setRes] = useState(null)
  const [meta, setMeta] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  const downloadSample = () => {
    const head = [...bank.feature_names, 'label']
    const csv = [head.join(','), ...bank.samples.slice(0, 12).map((s) => [...bank.feature_names.map((f) => s.features[f]), s.label].join(','))].join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); a.download = 'sample_flows.csv'; a.click()
    URL.revokeObjectURL(a.href)
  }

  const onFile = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setErr(''); setRes(null); setBusy(true)
    try {
      const { head, rows } = parseCsv(await file.text())
      const missing = bank.feature_names.filter((f) => !head.includes(f))
      if (missing.length) throw new Error(`Missing ${missing.length} feature column(s), for example “${missing[0]}”. Download the sample CSV to see the format.`)
      if (rows.length > 2000) throw new Error('At most 2,000 rows per upload.')
      const labelCol = head.findIndex((h) => h.toLowerCase() === 'label')
      const feats = rows.map((r) => Object.fromEntries(bank.feature_names.map((f) => [f, Number(r[head.indexOf(f)])])))
      if (feats.some((r) => Object.values(r).some((v) => !Number.isFinite(v)))) throw new Error('Some feature values are not numbers.')
      let labels = null
      if (labelCol >= 0) labels = rows.map((r) => (/^(0|benign|normal)$/i.test(r[labelCol]) ? 0 : 1))
      setMeta({ name: file.name, n: rows.length, labels })
      setRes(await api.predictRows({ rows: feats, labels, thresholds: thr }))
    } catch (x) { setErr(x.message) } finally { setBusy(false); e.target.value = '' }
  }

  return (
    <Card title="4 · Test your own flows" subtitle="Upload a CSV with the 77 feature columns (and optionally a label column: 0/1 or Benign/attack name). Up to 2,000 rows."
      right={<div className="try-actions"><button className="btn" onClick={downloadSample}>Download sample CSV</button><label className="btn primary file-btn">{busy ? 'Scoring…' : 'Choose CSV file'}<input type="file" accept=".csv,text/csv" onChange={onFile} hidden /></label></div>}>
      {err && <p className="error">{err}</p>}
      {!res && !err && <div className="empty">No file yet. Use the sample CSV as a template.</div>}
      {res && meta && (
        <>
          <p className="muted small">{meta.name}: {int(meta.n)} flows scored at the thresholds above.</p>
          {meta.labels ? (
            <div className="stat-grid">
              {MODEL_ORDER.filter((m) => res.summary[m]).map((m) => {
                const s = res.summary[m]
                return <Stat key={m} label={m} value={`${s.correct} / ${s.total}`} hint={`${s.fn} missed attacks · ${s.fp} false alarms`} />
              })}
            </div>
          ) : (
            <div className="stat-grid">
              {MODEL_ORDER.map((m) => <Stat key={m} label={m} value={`${res.rows[m].pred.filter((v) => v === 1).length} attacks`} hint={`of ${meta.n} flows flagged`} />)}
            </div>
          )}
          <div className="table-wrap tall short">
            <table className="tbl">
              <thead><tr><th className="r">#</th>{meta.labels && <th>True</th>}{MODEL_ORDER.map((m) => <th key={m} className="r">{m}</th>)}</tr></thead>
              <tbody>
                {Array.from({ length: Math.min(meta.n, 50) }, (_, i) => (
                  <tr key={i}>
                    <td className="r muted">{i + 1}</td>
                    {meta.labels && <td><Chip attack={meta.labels[i] === 1}>{meta.labels[i] === 1 ? 'Attack' : 'Benign'}</Chip></td>}
                    {MODEL_ORDER.map((m) => {
                      const r = res.rows[m]; const bad = meta.labels && r.pred[i] !== meta.labels[i]
                      return <td key={m} className={`r ${bad ? 'bad-cell' : ''}`}>{r.pred[i] ? 'Attack' : 'Benign'} <span className="muted">{(r.prob[i] * 100).toFixed(1)}%</span></td>
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {meta.n > 50 && <p className="muted small">Showing the first 50 of {int(meta.n)} rows.</p>}
        </>
      )}
    </Card>
  )
}

/* ---------------------------------------------------------------- background jobs */
function useJob() {
  const [job, setJob] = useState(null)
  const [err, setErr] = useState('')
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])
  const poll = useCallback((id) => {
    api.job(id).then((j) => { setJob(j); if (j.status === 'running') timer.current = setTimeout(() => poll(id), 1000) }).catch((e) => setErr(e.message))
  }, [])
  const start = useCallback(async (body) => {
    setErr(''); setJob(null)
    try { const j = await api.startJob(body); setJob(j); poll(j.id) } catch (e) { setErr(e.message) }
  }, [poll])
  return { job, err, start, running: job?.status === 'running' }
}

function LogBox({ lines }) {
  const ref = useRef(null)
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight }, [lines])
  return <pre className="log" ref={ref}>{lines.length ? lines.join('\n') : 'Waiting for output…'}</pre>
}

function Jobs() {
  const tests = useJob()
  return (
    <Card title="5 · Run the automated tests" subtitle="The project’s 80 tests (synthetic data only; they never touch the real data). Takes about 3 seconds. For federated training runs, see the Training monitor tab."
      right={<button className="btn primary" disabled={tests.running} onClick={() => tests.start({ kind: 'tests' })}>{tests.running ? 'Running…' : 'Run tests'}</button>}>
      {tests.err && <p className="error">{tests.err}</p>}
      {tests.job && (
        <>
          {tests.job.status === 'done' && tests.job.result?.passed != null && <p className="ok-line">✓ {tests.job.result.passed} passed, {tests.job.result.failed} failed in {tests.job.result.pytest_seconds}s</p>}
          {tests.job.status === 'failed' && <p className="error">The tests did not pass. See the log.</p>}
          <LogBox lines={tests.job.log} />
        </>
      )}
      {!tests.job && !tests.err && <div className="empty">Press Run tests.</div>}
    </Card>
  )
}
