import { useMemo, useState } from 'react'
import dataset from '../data/dataset.json'
import { Card, Stat, Chip, PageTitle, Segmented } from '../components/ui.jsx'
import { int, pct, cell } from '../format.js'
import { DataPipelineCard } from '../components/Diagrams.jsx'

export default function DatasetPage() {
  const total = dataset.rows
  const attacks = dataset.splits[0].attacks
  return (
    <>
      <PageTitle title="Dataset">
        {dataset.name}: network flows described by {dataset.n_features} numeric features plus a binary label.
      </PageTitle>
      <div className="stat-grid">
        <Stat label="Flows" value={int(total)} hint={`${int(dataset.raw_rows)} raw − ${int(dataset.duplicates_removed)} duplicates`} />
        <Stat label="Feature columns" value={dataset.n_features} hint="+ 1 label column = 78" />
        <Stat label="Benign flows" value={int(total - attacks)} hint={pct((total - attacks) / total)} />
        <Stat label="Attack flows" value={int(attacks)} hint={`${pct(attacks / total)} · ${dataset.attack_types.length - 1} attack types`} />
      </div>
      <ClassDistribution />
      <DataPipelineCard />
      <Splits />
      <Samples />
      <Columns />
    </>
  )
}

function ClassDistribution() {
  const [scale, setScale] = useState('log')
  const max = dataset.attack_types[0].count
  const w = (c) => (scale === 'log' ? Math.max(Math.log10(c) / Math.log10(max), 0.01) : c / max) * 100
  return (
    <Card
      title="Class distribution"
      subtitle="Flows per traffic type after cleaning (the dataset is imbalanced)"
      right={<Segmented value={scale} onChange={setScale} options={[{ value: 'log', label: 'Log scale' }, { value: 'linear', label: 'Linear' }]} />}
    >
      <ul className="dist">
        {dataset.attack_types.map((a) => (
          <li key={a.label}>
            <span className="dist-label">{a.label}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${w(a.count)}%`, background: a.binary ? 'var(--s2)' : 'var(--s1)' }} /></span>
            <b className="dist-n">{int(a.count)}</b>
            <span className="dist-pct">{pct(a.share, 3)}</span>
          </li>
        ))}
      </ul>
      <p className="muted small">Blue is normal traffic (label 0), orange is an attack (label 1). All 14 attack types are merged into one Attack class for the binary task.</p>
    </Card>
  )
}

function Splits() {
  return (
    <Card title="Data splits" subtitle="Stratified 80 / 20 split with seed 42; the test set is never used for training or tuning">
      <div className="table-wrap">
        <table className="tbl">
          <thead><tr><th>File</th><th>Role</th><th className="r">Flows</th><th className="r">Benign</th><th className="r">Attack</th><th className="r">Attack share</th></tr></thead>
          <tbody>
            {dataset.splits.map((s) => (
              <tr key={s.file}>
                <td className="mono">{s.file}</td><td>{s.description}</td>
                <td className="r">{int(s.rows)}</td><td className="r">{int(s.benign)}</td><td className="r">{int(s.attacks)}</td>
                <td className="r">{pct(s.attacks / s.rows)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function Samples() {
  const [view, setView] = useState('rows')
  const [q, setQ] = useState('')
  const cols = dataset.columns.map((c) => c.name)
  const shown = useMemo(() => cols.filter((c) => c.toLowerCase().includes(q.trim().toLowerCase())), [q])
  const s = dataset.samples
  const idShort = (n) => `#${n.toLocaleString('en-US')}`

  return (
    <Card
      title="Sample rows"
      subtitle={`${s.length} real flows: ${s.filter((x) => !x.label).length} normal and ${s.filter((x) => x.label).length} attacks. ${dataset.sample_note}`}
      right={<Segmented value={view} onChange={setView} options={[{ value: 'rows', label: 'Rows' }, { value: 'features', label: 'Features' }]} />}
    >
      <input className="search" placeholder="Filter feature columns…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter feature columns" />
      <div className="table-wrap tall">
        {view === 'rows' ? (
          <table className="tbl samples">
            <thead>
              <tr>
                <th className="sticky s0">Row</th><th className="sticky s1">Traffic type</th><th className="sticky s2">label</th>
                {shown.map((c) => <th key={c} className="r">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {s.map((r) => (
                <tr key={r.id}>
                  <td className="sticky s0 mono">{idShort(r.id)}</td>
                  <td className="sticky s1"><Chip attack={!!r.label}>{r.type}</Chip></td>
                  <td className="sticky s2 r"><b>{r.label}</b></td>
                  {shown.map((c) => <td key={c} className="r">{cell(r.values[c])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="tbl samples">
            <thead>
              <tr>
                <th className="sticky s0">Feature</th>
                {s.map((r) => <th key={r.id} className="r"><Chip attack={!!r.label}>{r.type}</Chip><div className="mono muted small">{idShort(r.id)}</div></th>)}
              </tr>
            </thead>
            <tbody>
              <tr className="emph"><td className="sticky s0"><b>label</b></td>{s.map((r) => <td key={r.id} className="r"><b>{r.label}</b></td>)}</tr>
              {shown.map((c) => (
                <tr key={c}>
                  <td className="sticky s0">{c}</td>
                  {s.map((r) => <td key={r.id} className="r">{cell(r.values[c])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <p className="muted small">Showing {shown.length} of {cols.length} feature columns. Scroll the table sideways or down to see every value.</p>
    </Card>
  )
}

function Columns() {
  const [q, setQ] = useState('')
  const attacks = dataset.splits[0].attacks
  const rows = useMemo(() => {
    const all = [
      ...dataset.columns,
      { index: dataset.n_features + 1, name: 'label', dtype: 'int64', min: 0, max: 1, mean: attacks / dataset.rows, benign_mean: 0, attack_mean: 1 },
    ]
    return all.filter((c) => c.name.toLowerCase().includes(q.trim().toLowerCase()))
  }, [q])
  return (
    <Card title="Dataset columns" subtitle="All 77 feature columns and the label, with value ranges and the average value for benign and attack flows">
      <input className="search" placeholder="Search a column name…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search column names" />
      <div className="table-wrap tall">
        <table className="tbl">
          <thead>
            <tr>
              <th className="r">#</th><th>Column name</th><th>Type</th><th className="r">Min</th><th className="r">Max</th>
              <th className="r">Mean (all)</th><th className="r">Mean (benign)</th><th className="r">Mean (attack)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.name} className={c.name === 'label' ? 'emph' : ''}>
                <td className="r muted">{c.index}</td>
                <td>{c.name}{c.name === 'label' && <span className="muted small"> (target: Benign = 0, Attack = 1)</span>}</td>
                <td className="mono">{c.dtype}</td>
                <td className="r">{cell(c.min)}</td><td className="r">{cell(c.max)}</td><td className="r">{cell(c.mean)}</td>
                <td className="r">{cell(c.benign_mean)}</td><td className="r">{cell(c.attack_mean)}</td>
              </tr>
            ))}
            {!rows.length && <tr><td colSpan={8} className="muted">No column matches “{q}”.</td></tr>}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
