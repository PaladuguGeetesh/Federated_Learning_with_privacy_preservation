import dataset from '../data/dataset.json'
import results from '../data/results.json'
import { Card, Stat, PageTitle } from '../components/ui.jsx'
import { int, pct, annParams } from '../format.js'

export default function Overview({ go }) {
  const fedAnn = results.models.find((m) => m.model === 'Federated ANN')
  const cenAnn = results.models.find((m) => m.model === 'Centralized ANN (tuned)')
  const rf = results.models.find((m) => m.model === 'Centralized RF')
  const train = dataset.splits.find((s) => s.file === 'train.parquet')
  const test = dataset.splits.find((s) => s.file === 'test.parquet')
  const client = dataset.splits.find((s) => s.file === 'client_A.parquet')
  const fed = results.federated.ann

  const steps = [
    { n: 1, t: 'Raw CICIDS2017', v: int(dataset.raw_rows), d: '8 parquet files, 77 flow features' },
    { n: 2, t: 'Clean and de-duplicate', v: int(dataset.rows), d: `${int(dataset.duplicates_removed)} duplicates removed; Benign = 0, Attack = 1` },
    { n: 3, t: 'Stratified 80 / 20 split', v: `${int(train.rows)} / ${int(test.rows)}`, d: 'training pool / held-out test set' },
    { n: 4, t: 'Three organizations', v: `3 × ${int(client.rows)}`, d: 'clients A, B and C hold IID parts of the training pool' },
    { n: 5, t: 'Federated training', v: `${fed.rounds} rounds`, d: `FedAvg, ${fed.local_epochs} local epochs per round; only weights are shared` },
    { n: 6, t: 'One test evaluation', v: `${int(results.test_set.flows)} flows`, d: 'thresholds and checkpoints chosen on validation data only' },
  ]

  return (
    <>
      <section className="hero">
        <div>
          <span className="eyebrow">Project Work 2 · 24AM7PWPW2</span>
          <h1>Collaborative malware detection without sharing network data</h1>
          <p>
            Three simulated organizations train one intrusion detector together with Federated Averaging. Raw
            network flows never leave a client; only model weights and sample counts are exchanged. Every number
            in this dashboard comes from the saved results of the project.
          </p>
          <div className="hero-actions">
            <button className="btn primary" onClick={() => go('results')}>See the results</button>
            <button className="btn" onClick={() => go('dataset')}>Browse the dataset</button>
          </div>
        </div>
        <div className="hero-card">
          <div className="hero-card-label">Federated ANN on {int(results.test_set.flows)} unseen flows</div>
          <div className="hero-big">{pct(fedAnn.accuracy)}</div>
          <div className="muted">accuracy</div>
          <div className="hero-row">
            <div><b>{pct(fedAnn.f1)}</b><span>attack F1</span></div>
            <div><b>{fedAnn.roc_auc.toFixed(4)}</b><span>ROC-AUC</span></div>
            <div><b>{int(fedAnn.fn)}</b><span>missed attacks</span></div>
          </div>
        </div>
      </section>

      <div className="stat-grid">
        <Stat label="Network flows" value={int(dataset.rows)} hint="after removing duplicates" />
        <Stat label="Features per flow" value={dataset.n_features} hint="numeric CICIDS2017 columns" />
        <Stat label="Attack share" value={pct(dataset.splits[0].attacks / dataset.rows, 2)} hint={`${int(dataset.splits[0].attacks)} attack flows`} />
        <Stat label="Gap to centralized ANN" value={`${results.gaps_ann.f1.toFixed(3)} pp`} hint="attack F1, same data and settings" tone="good" />
      </div>

      <Card title="How the pipeline works" subtitle="From the raw files to the final comparison">
        <ol className="steps">
          {steps.map((s) => (
            <li key={s.n}>
              <span className="step-n">{s.n}</span>
              <div>
                <div className="step-t">{s.t}</div>
                <div className="step-v">{s.v}</div>
                <div className="muted small">{s.d}</div>
              </div>
            </li>
          ))}
        </ol>
      </Card>

      <div className="grid-2">
        <Card title="Four models, one test set" subtitle="Attack F1 on the held-out test set">
          <ul className="bars">
            {[rf, cenAnn, fedAnn, results.models.find((m) => m.model === 'Federated LR')].map((m) => (
              <li key={m.model}>
                <span className="bar-label">{m.model}</span>
                <span className="bar-track"><span className="bar-fill" style={{ width: `${m.f1 * 100}%`, background: colorFor(m.model) }} /></span>
                <b className="bar-val">{pct(m.f1)}</b>
              </li>
            ))}
          </ul>
          <p className="muted small">
            In this IID setting federation costs very little: {results.gaps_ann.accuracy.toFixed(3)} pp of accuracy and {results.gaps_ann.f1.toFixed(3)} pp of
            F1 against the centralized ANN. The Random Forest is strongest ({pct(rf.f1)}); the linear federated model is clearly weaker.
          </p>
        </Card>
        <Card title="The tuned neural network" subtitle="Architecture found by an Optuna search (trial 14 of 40)" right={<button className="btn small-btn" onClick={() => go('architecture')}>Full diagrams →</button>}>
          <div className="arch">
            {[['Input', 77], ['Hidden 1', 312], ['Hidden 2', 289], ['Hidden 3', 201], ['Output', 1]].map(([n, u], i, a) => (
              <div className="arch-item" key={n}>
                <div className="arch-box"><b>{u}</b><span>{n}</span></div>
                {i < a.length - 1 && <span className="arch-arrow" aria-hidden>→</span>}
              </div>
            ))}
          </div>
          <dl className="kv">
            <div><dt>Trainable parameters</dt><dd>{int(annParams(77, [312, 289, 201]))}</dd></div>
            <div><dt>Activation / dropout</dt><dd>ReLU / {results.optuna.params.dropout.toFixed(3)}</dd></div>
            <div><dt>Optimizer</dt><dd>{results.optuna.params.optimizer.toUpperCase()}, lr {results.optuna.params.learning_rate.toExponential(2)}</dd></div>
            <div><dt>Federated run time</dt><dd>{fed.seconds.toFixed(0)} s for {fed.rounds} rounds</dd></div>
          </dl>
        </Card>
      </div>
    </>
  )
}

export const MODEL_COLORS = {
  'Federated ANN': 'var(--s1)',
  'Centralized ANN (tuned)': 'var(--s2)',
  'Centralized RF': 'var(--s3)',
  'Federated LR': 'var(--s4)',
}
export const colorFor = (m) => MODEL_COLORS[m] || 'var(--s1)'
