import { useEffect, useState } from 'react'
import Overview from './pages/Overview.jsx'
import ArchitecturePage from './pages/ArchitecturePage.jsx'
import DatasetPage from './pages/DatasetPage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'
import TrainingPage from './pages/TrainingPage.jsx'

const TABS = [
  { id: 'overview', label: 'Overview', Page: Overview },
  { id: 'architecture', label: 'Architecture', Page: ArchitecturePage },
  { id: 'dataset', label: 'Dataset', Page: DatasetPage },
  { id: 'results', label: 'Results', Page: ResultsPage },
  { id: 'training', label: 'Training', Page: TrainingPage },
]

const fromHash = () => {
  const h = window.location.hash.replace('#', '')
  return TABS.some((t) => t.id === h) ? h : 'overview'
}

export default function App() {
  const [tab, setTab] = useState(fromHash)
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('theme') || '' } catch { return '' }
  })

  useEffect(() => {
    const on = () => setTab(fromHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  useEffect(() => {
    const root = document.documentElement
    if (theme) root.setAttribute('data-theme', theme)
    else root.removeAttribute('data-theme')
    try { theme ? localStorage.setItem('theme', theme) : localStorage.removeItem('theme') } catch { /* ignore */ }
  }, [theme])

  const go = (id) => {
    window.location.hash = id
    setTab(id)
    window.scrollTo({ top: 0 })
  }
  const dark = theme ? theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  const { Page } = TABS.find((t) => t.id === tab)

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="logo" aria-hidden>
              <svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 2l8 3v6c0 5-3.4 8.6-8 11-4.6-2.4-8-6-8-11V5z" fill="currentColor" /></svg>
            </span>
            <div>
              <div className="brand-title">Federated Malware Detection</div>
              <div className="brand-sub">Privacy-preserving intrusion detection · CICIDS2017</div>
            </div>
          </div>
          <nav className="tabs" aria-label="Sections">
            {TABS.map((t) => (
              <button key={t.id} className={t.id === tab ? 'on' : ''} onClick={() => go(t.id)} aria-current={t.id === tab ? 'page' : undefined}>
                {t.label}
              </button>
            ))}
          </nav>
          <button className="theme-btn" onClick={() => setTheme(dark ? 'light' : 'dark')} aria-label="Toggle light or dark theme" title="Toggle theme">
            {dark ? '☀' : '☾'}
          </button>
        </div>
      </header>
      <main className="page">
        <Page go={go} />
      </main>
      <footer className="footer">
        K Karthik · Manjari M Varma · P Prerana Acharya · Paladugu Geetesh — Guide: Dr. Nagarathna C R, Dept. of MEL, BMS College of Engineering
      </footer>
    </div>
  )
}
