import dataset from '../data/dataset.json'
import results from '../data/results.json'
import { Card } from './ui.jsx'
import Figure from './Figure.jsx'
import { int, annParams } from '../format.js'

const split = (f) => dataset.splits.find((s) => s.file === f)
const VALIDATION = 178545 // combined validation slices of the three clients (from the saved evaluation files)

function Explain({ title = 'How to read it', items }) {
  return (
    <div className="explain">
      <h4>{title}</h4>
      <ol>
        {items.map(([head, text]) => (
          <li key={head}><b>{head}</b> {text}</li>
        ))}
      </ol>
    </div>
  )
}

export function SystemArchCard() {
  const c = split('client_A.parquet')
  return (
    <Card title="System architecture" subtitle="Training data stays at each organization; only model weights travel">
      <Figure src="/diagrams/system_architecture.png" alt="Federated learning system architecture" caption="Figure: system architecture of the federated framework" />
      <Explain items={[
        ['Organizations A, B and C (left).', `Each holds its own ${int(c.rows)} network flows. It fits its own feature scaler and trains the shared model locally for ${results.federated.ann.local_epochs} epochs per round. The raw flows never leave the organization.`],
        ['Blue arrows: upload.', 'Each client sends only its updated weights and the number of samples n_k it trained on.'],
        ['Server (centre).', 'Combines the three weight sets with FedAvg, a sample-weighted average, so a client with more data counts for more. It only ever sees weights.'],
        ['Red arrows: broadcast.', 'The new global model goes back to every organization for the next round.'],
        ['Validation (top right).', `After every round the server scores the global model on ${int(VALIDATION)} validation flows and keeps the round with the best PR-AUC as best.pt.`],
        ['Final test (bottom right).', `Run once on ${int(results.test_set.flows)} held-out flows, using the decision threshold that was chosen on validation data.`],
      ]} />
    </Card>
  )
}

export function AnnArchCard() {
  const layers = [[77, 312], [312, 289], [289, 201], [201, 1]]
  return (
    <Card title="ANN architecture" subtitle="Feed-forward network (MLP) for binary intrusion detection: 77 → 312 → 289 → 201 → 1">
      <Figure src="/diagrams/ann_architecture.png" alt="ANN architecture with layer sizes and training settings" caption="Figure: architecture and key settings of the tuned ANN" />
      <Explain items={[
        ['Input layer (77).', 'One neuron per flow feature. The features are standardized with a scaler fitted on training rows only.'],
        ['Three hidden layers (312, 289, 201).', 'Each is a fully connected layer followed by ReLU and dropout 0.099. The sizes were chosen by an Optuna search (trial 14 of 40) that maximised the validation PR-AUC of the attack class.'],
        ['Output layer (1).', 'A single logit; a sigmoid turns it into the probability that the flow is an attack. A flow is called an attack when this probability exceeds the threshold chosen on validation data: 0.645 for the centralized model and 0.551 for the federated one.'],
        ['Parameters.', `Weights plus biases per layer: ${layers.map(([i, o]) => int(i * o + o)).join(' + ')} = ${int(annParams(77, [312, 289, 201]))} trainable parameters. This small size is why a federated round only moves about 0.66 MiB per client.`],
        ['Training (right panel).', 'BCEWithLogitsLoss, AdamW (learning rate 2.03e-4, weight decay 1.49e-6), batch size 2048, gradient clipping at norm 4.10 and early stopping on validation PR-AUC. The same architecture and settings are used for the centralized and the federated ANN, which keeps the comparison fair.'],
      ]} />
    </Card>
  )
}

export function DataPipelineCard() {
  const train = split('train.parquet'), test = split('test.parquet'), c = split('client_A.parquet')
  return (
    <Card title="Data pipeline and splits" subtitle="From the eight raw CICIDS2017 files to the three clients and the held-out test set">
      <Figure src="/diagrams/data_pipeline.png" alt="Data pipeline and splits" caption="Figure: data pipeline and splits" />
      <Explain items={[
        ['8 raw files.', `${int(dataset.raw_rows)} flows from the cleaned CICIDS2017 release, one file per capture period.`],
        ['Cleaned data.', `Exact duplicate rows are removed (${int(dataset.duplicates_removed)}), leaving ${int(dataset.rows)} flows. All attack types become one class: Benign = 0, Attack = 1.`],
        ['Stratified split (seed 42).', `80% / 20% with the attack share kept equal in both parts (${((test.attacks / test.rows) * 100).toFixed(2)}% attacks). The ${int(test.rows)}-flow test set is held out and never used for training, tuning or threshold selection.`],
        ['Train set.', `${int(train.rows)} flows, divided at random (IID) into three equal client files of ${int(c.rows)} flows each.`],
        ['Clients A, B, C.', `Each client keeps about 90% of its rows for training and about 10% for validation. The three validation slices together give ${int(VALIDATION)} flows, which are used for model selection and threshold choice.`],
      ]} />
    </Card>
  )
}

export function RoundFlowCard() {
  const f = results.federated.ann
  return (
    <Card title="One federated communication round" subtitle={`The five steps repeated for ${f.rounds} rounds`}>
      <Figure src="/diagrams/federated_round_flow.png" alt="One federated round: broadcast, local training, upload, aggregate, validate" caption="Figure: one federated communication round" />
      <Explain items={[
        ['1 Broadcast.', 'The server sends the current global weights to clients A, B and C.'],
        ['2 Local training.', `Each client trains on its own data for ${f.local_epochs} epochs with AdamW and gradient clipping.`],
        ['3 Upload.', 'Each client returns its updated weights and its sample count n_k. No raw data is sent.'],
        ['4 Aggregate.', 'FedAvg computes w(t+1) = Σ (n_k / n) · w_k(t+1): the average of the client weights, weighted by how many samples each client used.'],
        ['5 Validate.', `The server scores the new global model on the combined validation set and saves the best and the latest weights. The best round was round ${f.best_round} (validation PR-AUC ${f.best_val_pr_auc.toFixed(4)}).`],
      ]} />
    </Card>
  )
}
