# Dashboard UI (React + Vite)

Shows the dataset columns, sample flows, results and training curves of the federated malware-detection project.

```bash
npm install      # first time only
npm run dev      # http://localhost:5173
npm start        # same, and opens the browser
npm run build    # production build in dist/
npm run preview  # serve the build on http://localhost:4173
npm run data     # re-export src/data/*.json from the project files (uses ../venv)
```

Needs Node.js 18 or newer (nothing else). Data files: `src/data/dataset.json` and `src/data/results.json`, written by `scripts/export_data.py`.
