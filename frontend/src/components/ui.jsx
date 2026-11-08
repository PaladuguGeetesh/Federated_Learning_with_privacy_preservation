export function Card({ title, subtitle, right, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <header className="card-head">
          <div>
            {title && <h3>{title}</h3>}
            {subtitle && <p className="muted">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, hint, tone }) {
  return (
    <div className={`stat ${tone || ''}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  )
}

export function Chip({ attack, children }) {
  return <span className={`chip ${attack ? 'chip-attack' : 'chip-benign'}`}>{children}</span>
}

export function Legend({ items }) {
  return (
    <ul className="legend">
      {items.map((i) => (
        <li key={i.label}>
          <span className="swatch" style={{ background: i.color }} />
          {i.label}
        </li>
      ))}
    </ul>
  )
}

export function PageTitle({ title, children }) {
  return (
    <div className="page-title">
      <h2>{title}</h2>
      {children && <p className="muted">{children}</p>}
    </div>
  )
}

export function Segmented({ options, value, onChange }) {
  return (
    <div className="segmented" role="group">
      {options.map((o) => (
        <button key={o.value} className={o.value === value ? 'on' : ''} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

// Tooltip shared by all charts. `format` turns a value into text.
export function ChartTip({ active, payload, label, labelPrefix = '', format = (v) => v }) {
  if (!active || !payload?.length) return null
  return (
    <div className="tip">
      <div className="tip-title">{labelPrefix}{label}</div>
      {payload.map((p) => (
        <div className="tip-row" key={p.dataKey + p.name}>
          <span className="swatch" style={{ background: p.color || p.stroke || p.fill }} />
          <span>{p.name}</span>
          <b>{format(p.value, p)}</b>
        </div>
      ))}
    </div>
  )
}
