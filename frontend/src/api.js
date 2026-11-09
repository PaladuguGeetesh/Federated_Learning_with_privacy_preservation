// Thin client for the Python backend (see backend/main.py). All paths go through the Vite proxy.
async function call(path, body) {
  let res
  try {
    res = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  } catch {
    throw new Error('Cannot reach the backend. Start it with: uvicorn backend.main:app --port 8000')
  }
  if (!res.ok) {
    let detail = res.statusText
    try { const j = await res.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail) } catch { /* ignore */ }
    if (res.status >= 500 && res.status !== 503 && detail === res.statusText) detail = 'Backend is not running (start it with uvicorn backend.main:app --port 8000)'
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  health: () => call('/api/health'),
  samples: () => call('/api/samples'),
  predict: (body) => call('/api/predict', body),
  predictBatch: (body) => call('/api/predict-batch', body),
  quickEval: (body) => call('/api/quick-eval', body),
  startJob: (body) => call('/api/jobs', body),
  job: (id) => call(`/api/jobs/${id}`),
  stopJob: (id) => call(`/api/jobs/${id}/stop`, {}),
  history: () => call('/api/history'),
  clearHistory: () => fetch('/api/history', { method: 'DELETE' }).then((r) => r.json()),
  comm: () => call('/api/comm'),
  predictRows: (body) => call('/api/predict-rows', body),
}
