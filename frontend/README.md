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

## Live test and Training monitor pages

The Live test and Training monitor tabs need the Python backend (it loads the trained models). From the project root, in a second terminal:

```bash
source venv/bin/activate
uvicorn backend.main:app --port 8000
```

`npm run dev` forwards `/api` to port 8000 (see `vite.config.js`). The other tabs work without the backend.
